#include "forge/cuda_utils.h"
#include "forge/execution_context.h"
#include "forge/kernels.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect_close(float actual, float expected, float tolerance, const char* label) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "FAIL " << label << ": actual=" << actual << " expected=" << expected << '\n';
    ++failures;
  }
}

template <typename T>
T* upload(const std::vector<T>& host) {
  T* device{};
  FORGE_CUDA(cudaMalloc(&device, host.size() * sizeof(T)));
  FORGE_CUDA(cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
  return device;
}

void test_rmsnorm() {
  constexpr int width = 13;
  std::vector<half> input(2 * width), weight(width);
  for (int i = 0; i < static_cast<int>(input.size()); ++i) {
    input[i] = __float2half(static_cast<float>(i - 4) / 8.0F);
  }
  for (int i = 0; i < width; ++i) weight[i] = __float2half(0.5F + i / 16.0F);
  auto* d_input = upload(input);
  auto* d_weight = upload(weight);
  half* d_output{};
  FORGE_CUDA(cudaMalloc(&d_output, input.size() * sizeof(half)));
  forge::launch_rmsnorm(d_input, d_weight, d_output, 2, width, 1e-6F, nullptr);
  std::vector<half> output(input.size());
  FORGE_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half), cudaMemcpyDeviceToHost));
  for (int row = 0; row < 2; ++row) {
    float sum = 0;
    for (int col = 0; col < width; ++col) {
      const float value = __half2float(input[row * width + col]);
      sum += value * value;
    }
    const float scale = 1.0F / std::sqrt(sum / width + 1e-6F);
    for (int col = 0; col < width; ++col) {
      const float expected = __half2float(input[row * width + col]) * scale * __half2float(weight[col]);
      expect_close(__half2float(output[row * width + col]), expected, 2e-3F, "rmsnorm");
    }
  }
  cudaFree(d_output); cudaFree(d_weight); cudaFree(d_input);
}

void test_embedding_residual_and_linear() {
  const std::vector<std::int32_t> tokens = {2, 0};
  std::vector<half> table(12);
  for (int i = 0; i < 12; ++i) table[i] = __float2half(static_cast<float>(i));
  auto* d_tokens = upload(tokens);
  auto* d_table = upload(table);
  half* d_hidden{};
  FORGE_CUDA(cudaMalloc(&d_hidden, 8 * sizeof(half)));
  forge::launch_embedding(d_tokens, d_table, d_hidden, 2, 4, nullptr);
  std::vector<half> hidden(8);
  FORGE_CUDA(cudaMemcpy(hidden.data(), d_hidden, 8 * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(hidden[0]), 8.0F, 0.0F, "embedding row");
  expect_close(__half2float(hidden[7]), 3.0F, 0.0F, "embedding second row");

  std::vector<half> update(8, __float2half(1.0F));
  std::vector<half> norm_weight(4, __float2half(1.0F));
  auto* d_update = upload(update);
  auto* d_norm_weight = upload(norm_weight);
  half* d_normalized{};
  FORGE_CUDA(cudaMalloc(&d_normalized, 8 * sizeof(half)));
  forge::launch_residual_rmsnorm(d_hidden, d_update, d_norm_weight, d_normalized,
                                 2, 4, 1e-6F, nullptr);
  forge::launch_add(d_hidden, d_update, 8, nullptr);
  FORGE_CUDA(cudaMemcpy(hidden.data(), d_hidden, 8 * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(hidden[0]), 10.0F, 0.0F, "fused residual then add");

  // A[2,3] * W[2,3]^T + bias[2].
  std::vector<half> a = {__float2half(1), __float2half(2), __float2half(3),
                         __float2half(4), __float2half(5), __float2half(6)};
  std::vector<half> w = {__float2half(1), __float2half(0), __float2half(-1),
                         __float2half(2), __float2half(1), __float2half(0)};
  std::vector<half> bias = {__float2half(0.5F), __float2half(-1.0F)};
  auto* d_a = upload(a);
  auto* d_w = upload(w);
  auto* d_bias = upload(bias);
  half* d_linear{};
  FORGE_CUDA(cudaMalloc(&d_linear, 4 * sizeof(half)));
  forge::ExecutionContext context(1U << 20U);
  context.linear_fp16(d_a, d_w, d_bias, d_linear, 2, 2, 3);
  context.linear_fp16(d_a, d_w, d_bias, d_linear, 2, 2, 3);
  context.synchronize();
  std::vector<half> linear(4);
  FORGE_CUDA(cudaMemcpy(linear.data(), d_linear, 4 * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(linear[0]), -1.5F, 2e-3F, "cuBLASLt row major output zero");
  expect_close(__half2float(linear[1]), 3.0F, 2e-3F, "cuBLASLt bias output one");
  expect_close(__half2float(linear[2]), -1.5F, 2e-3F, "cuBLASLt row major output two");
  expect_close(__half2float(linear[3]), 12.0F, 2e-3F, "cuBLASLt bias output three");
  if (context.linear_plan_count() != 1) { std::cerr << "FAIL cuBLASLt plan cache\n"; ++failures; }

  cudaFree(d_linear); cudaFree(d_bias); cudaFree(d_w); cudaFree(d_a);
  cudaFree(d_normalized); cudaFree(d_norm_weight); cudaFree(d_update);
  cudaFree(d_hidden); cudaFree(d_table); cudaFree(d_tokens);
}

void test_silu_and_argmax() {
  std::vector<half> gate = {__float2half(-1), __float2half(0), __float2half(1), __float2half(2)};
  std::vector<half> up = {__float2half(2), __float2half(3), __float2half(4), __float2half(5)};
  auto* d_gate = upload(gate);
  auto* d_up = upload(up);
  half* d_output{};
  FORGE_CUDA(cudaMalloc(&d_output, 4 * sizeof(half)));
  forge::launch_silu_mul(d_gate, d_up, d_output, 4, nullptr);
  std::vector<half> output(4);
  FORGE_CUDA(cudaMemcpy(output.data(), d_output, 4 * sizeof(half), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 4; ++i) {
    const float x = __half2float(gate[i]);
    expect_close(__half2float(output[i]), x / (1 + std::exp(-x)) * __half2float(up[i]), 2e-3F, "silu_mul");
  }
  std::int32_t* d_token{};
  FORGE_CUDA(cudaMalloc(&d_token, sizeof(std::int32_t)));
  forge::launch_argmax(d_output, d_token, 1, 4, nullptr);
  std::int32_t token = -1;
  FORGE_CUDA(cudaMemcpy(&token, d_token, sizeof(token), cudaMemcpyDeviceToHost));
  if (token != 3) { std::cerr << "FAIL argmax\n"; ++failures; }
  cudaFree(d_token); cudaFree(d_output); cudaFree(d_up); cudaFree(d_gate);
}

void test_causal_softmax() {
  std::vector<half> scores = {
      __float2half(1), __float2half(9), __float2half(9),
      __float2half(1), __float2half(2), __float2half(9),
      __float2half(1), __float2half(2), __float2half(3)};
  const std::vector<std::int32_t> lengths = {3};
  auto* d_scores = upload(scores);
  auto* d_lengths = upload(lengths);
  forge::launch_causal_softmax(d_scores, d_lengths, 1, 1, 3, 3, nullptr);
  FORGE_CUDA(cudaMemcpy(scores.data(), d_scores, scores.size() * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(scores[0]), 1.0F, 1e-3F, "causal row zero");
  expect_close(__half2float(scores[1]), 0.0F, 1e-3F, "causal mask");
  const float denom = std::exp(1.0F) + std::exp(2.0F) + std::exp(3.0F);
  expect_close(__half2float(scores[8]), std::exp(3.0F) / denom, 2e-3F, "causal full row");
  cudaFree(d_lengths); cudaFree(d_scores);
}

void test_rope_and_cache_write() {
  constexpr int query_heads = 2, kv_heads = 1, head_dim = 4, block_tokens = 4;
  std::vector<half> query(2 * query_heads * head_dim);
  std::vector<half> key(2 * kv_heads * head_dim);
  std::vector<half> value(2 * kv_heads * head_dim);
  for (int i = 0; i < static_cast<int>(query.size()); ++i) query[i] = __float2half(0.1F * (i + 1));
  for (int i = 0; i < static_cast<int>(key.size()); ++i) key[i] = __float2half(0.2F * (i + 1));
  for (int i = 0; i < static_cast<int>(value.size()); ++i) value[i] = __float2half(1.0F + i);
  const auto original_query = query;
  const auto original_key = key;
  const std::vector<std::int32_t> table = {0};
  auto* d_query = upload(query);
  auto* d_key = upload(key);
  auto* d_value = upload(value);
  auto* d_table = upload(table);
  half* d_cache{};
  FORGE_CUDA(cudaMalloc(&d_cache, 2 * kv_heads * block_tokens * head_dim * sizeof(half)));
  FORGE_CUDA(cudaMemset(d_cache, 0, 2 * kv_heads * block_tokens * head_dim * sizeof(half)));
  forge::launch_rope_cache_write_prefill(d_query, d_key, d_value, d_cache, d_table,
                                         2, query_heads, kv_heads, head_dim,
                                         block_tokens, 0, 1, 10000.0F, nullptr);
  FORGE_CUDA(cudaMemcpy(query.data(), d_query, query.size() * sizeof(half), cudaMemcpyDeviceToHost));
  std::vector<half> cache(2 * kv_heads * block_tokens * head_dim);
  FORGE_CUDA(cudaMemcpy(cache.data(), d_cache, cache.size() * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(query[0]), __half2float(original_query[0]), 1e-3F, "RoPE position zero");
  const int q_base = query_heads * head_dim;
  const float first = __half2float(original_query[q_base]);
  const float second = __half2float(original_query[q_base + 2]);
  expect_close(__half2float(query[q_base]), first * std::cos(1.0F) - second * std::sin(1.0F),
               2e-3F, "RoPE position one");
  expect_close(__half2float(cache[0]), __half2float(original_key[0]), 1e-3F, "prefill K cache write");
  const int value_plane = kv_heads * block_tokens * head_dim;
  expect_close(__half2float(cache[value_plane]), __half2float(value[0]), 1e-3F, "prefill V cache write");

  std::vector<half> decode_q(query_heads * head_dim, __float2half(0.5F));
  std::vector<half> decode_k(kv_heads * head_dim, __float2half(0.25F));
  std::vector<half> decode_v(kv_heads * head_dim, __float2half(3.0F));
  const std::vector<std::int32_t> position = {2};
  auto* d_decode_q = upload(decode_q);
  auto* d_decode_k = upload(decode_k);
  auto* d_decode_v = upload(decode_v);
  auto* d_position = upload(position);
  forge::launch_rope_cache_write(d_decode_q, d_decode_k, d_decode_v, d_cache,
                                 d_position, d_table, 1, query_heads, kv_heads,
                                 head_dim, 1, block_tokens, 0, 1, 10000.0F, nullptr);
  FORGE_CUDA(cudaMemcpy(cache.data(), d_cache, cache.size() * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(cache[value_plane + 2 * head_dim]), 3.0F, 1e-3F,
               "decode V cache write");
  cudaFree(d_position); cudaFree(d_decode_v); cudaFree(d_decode_k); cudaFree(d_decode_q);
  cudaFree(d_cache); cudaFree(d_table); cudaFree(d_value); cudaFree(d_key); cudaFree(d_query);
}

void test_paged_prefill_attention() {
  constexpr int blocks = 2, layers = 1, kv_heads = 1, block_tokens = 2, head_dim = 4;
  std::vector<half> cache(blocks * layers * 2 * kv_heads * block_tokens * head_dim,
                          __float2half(0.0F));
  const auto index = [=](int block, int kv, int offset, int dim) {
    return (((block * 2 + kv) * block_tokens + offset) * head_dim + dim);
  };
  const int physical_for_token[] = {1, 1, 0};
  const int offset_for_token[] = {0, 1, 0};
  const float keys[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}};
  const float values[3][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {2, 4, 6, 8}};
  for (int token = 0; token < 3; ++token) {
    for (int dim = 0; dim < 4; ++dim) {
      cache[index(physical_for_token[token], 0, offset_for_token[token], dim)] = __float2half(keys[token][dim]);
      cache[index(physical_for_token[token], 1, offset_for_token[token], dim)] = __float2half(values[token][dim]);
    }
  }
  std::vector<half> query(3 * 2 * 4, __float2half(0.0F));
  for (int token = 0; token < 3; ++token) {
    query[(token * 2) * 4] = __float2half(1.0F);
    query[(token * 2 + 1) * 4 + 1] = __float2half(1.0F);
  }
  // Make token two/head zero score the third key more highly than the first two.
  query[(2 * 2) * 4 + 1] = __float2half(1.0F);
  const std::vector<std::int32_t> table = {1, 0};
  auto* d_cache = upload(cache);
  auto* d_query = upload(query);
  auto* d_table = upload(table);
  half* d_output{};
  FORGE_CUDA(cudaMalloc(&d_output, query.size() * sizeof(half)));
  forge::launch_paged_prefill_attention(d_query, d_cache, d_output, d_table,
                                        3, 2, 1, 4, 2, 0, 1, nullptr);
  std::vector<half> output(query.size());
  FORGE_CUDA(cudaMemcpy(output.data(), d_output, output.size() * sizeof(half), cudaMemcpyDeviceToHost));
  for (int dim = 0; dim < 4; ++dim) {
    expect_close(__half2float(output[dim]), values[0][dim], 2e-3F, "prefill causal first token");
  }
  const float denom = 2.0F * std::exp(0.5F) + std::exp(1.0F);
  const float expected_dim0 = (std::exp(0.5F) * values[0][0] +
                               std::exp(0.5F) * values[1][0] +
                               std::exp(1.0F) * values[2][0]) / denom;
  expect_close(__half2float(output[(2 * 2) * 4]), expected_dim0, 3e-3F,
               "prefill paged/noncontiguous weighting");
  const std::vector<std::int32_t> lengths = {3};
  auto* d_lengths = upload(lengths);
  half* d_decode_output{};
  FORGE_CUDA(cudaMalloc(&d_decode_output, 2 * 4 * sizeof(half)));
  forge::launch_paged_decode_attention(d_query + 2 * 2 * 4, d_cache, d_decode_output,
                                       d_lengths, d_table, 1, 2, 1, 4, 2, 2, 0, 1, nullptr);
  std::vector<half> decode_output(8);
  FORGE_CUDA(cudaMemcpy(decode_output.data(), d_decode_output, 8 * sizeof(half), cudaMemcpyDeviceToHost));
  expect_close(__half2float(decode_output[0]), expected_dim0, 3e-3F,
               "decode and prefill paged attention agree");
  cudaFree(d_decode_output); cudaFree(d_lengths);
  cudaFree(d_output); cudaFree(d_table); cudaFree(d_query); cudaFree(d_cache);
}

}  // namespace

int main() {
  test_rmsnorm();
  test_embedding_residual_and_linear();
  test_silu_and_argmax();
  test_causal_softmax();
  test_rope_and_cache_write();
  test_paged_prefill_attention();
  FORGE_CUDA(cudaDeviceSynchronize());
  if (failures == 0) std::cout << "all CUDA tests passed\n";
  return failures == 0 ? 0 : 1;
}
