#include "forge/kernels.h"

#include "forge/cuda_utils.h"
#include "forge/error.h"

#include <cfloat>
#include <cmath>

namespace forge {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxContext = 2048;

__device__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ float block_sum(float value) {
  __shared__ float warps[32];
  __shared__ float result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? warps[lane] : 0.0F;
  if (warp == 0) {
    value = warp_sum(value);
    if (lane == 0) result = value;
  }
  __syncthreads();
  return result;
}

__device__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  return value;
}

__device__ float block_max(float value) {
  __shared__ float warps[32];
  __shared__ float result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) warps[warp] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? warps[lane] : -FLT_MAX;
  if (warp == 0) {
    value = warp_max(value);
    if (lane == 0) result = value;
  }
  __syncthreads();
  return result;
}

__global__ void embedding_kernel(const std::int32_t* tokens, const half* table, half* output,
                                 int token_count, int hidden_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = token_count * hidden_size;
  if (index < elements) {
    const int token_index = index / hidden_size;
    const int column = index % hidden_size;
    output[index] = table[static_cast<std::int64_t>(tokens[token_index]) * hidden_size + column];
  }
}

__global__ void rmsnorm_kernel(const half* input, const half* weight, half* output,
                               int hidden_size, float epsilon) {
  const auto row = blockIdx.x;
  const auto offset = static_cast<std::int64_t>(row) * hidden_size;
  float squares = 0.0F;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float value = __half2float(input[offset + i]);
    squares += value * value;
  }
  const float scale = rsqrtf(block_sum(squares) / hidden_size + epsilon);
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    output[offset + i] = __float2half(__half2float(input[offset + i]) * scale * __half2float(weight[i]));
  }
}

__global__ void residual_rmsnorm_kernel(half* residual, const half* update, const half* weight,
                                        half* output, int hidden_size, float epsilon) {
  const auto offset = static_cast<std::int64_t>(blockIdx.x) * hidden_size;
  float squares = 0.0F;
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    const float value = __half2float(residual[offset + i]) + __half2float(update[offset + i]);
    residual[offset + i] = __float2half(value);
    squares += value * value;
  }
  const float scale = rsqrtf(block_sum(squares) / hidden_size + epsilon);
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    output[offset + i] = __float2half(__half2float(residual[offset + i]) * scale * __half2float(weight[i]));
  }
}

__global__ void silu_mul_kernel(const half* gate, const half* up, half* output, std::int64_t n) {
  const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float x = __half2float(gate[i]);
    output[i] = __float2half((x / (1.0F + expf(-x))) * __half2float(up[i]));
  }
}

__global__ void add_kernel(half* destination, const half* update, std::int64_t n) {
  const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) destination[i] = __hadd(destination[i], update[i]);
}

__device__ std::int64_t cache_index(int physical_block, int layer, int kv, int head,
                                    int token_offset, int dim, int num_layers,
                                    int kv_heads, int block_tokens, int head_dim) {
  auto index = static_cast<std::int64_t>(physical_block) * num_layers + layer;
  index = index * 2 + kv;
  index = index * kv_heads + head;
  index = index * block_tokens + token_offset;
  return index * head_dim + dim;
}

__global__ void rope_cache_kernel(half* query, half* key, const half* value, half* cache,
                                  const std::int32_t* positions, const std::int32_t* tables,
                                  int query_heads, int kv_heads, int head_dim, int max_blocks,
                                  int block_tokens, int layer, int num_layers, float theta) {
  const int batch_index = blockIdx.x;
  const int position = positions[batch_index];
  const int logical_block = position / block_tokens;
  const int token_offset = position % block_tokens;
  const int physical_block = tables[batch_index * max_blocks + logical_block];
  const int half_dim = head_dim / 2;

  for (int index = threadIdx.x; index < query_heads * half_dim; index += blockDim.x) {
    const int head = index / half_dim;
    const int dim = index % half_dim;
    const float frequency = powf(theta, -2.0F * dim / head_dim);
    float s{}, c{};
    sincosf(position * frequency, &s, &c);
    const auto base = (static_cast<std::int64_t>(batch_index) * query_heads + head) * head_dim;
    const float first = __half2float(query[base + dim]);
    const float second = __half2float(query[base + dim + half_dim]);
    query[base + dim] = __float2half(first * c - second * s);
    query[base + dim + half_dim] = __float2half(second * c + first * s);
  }
  for (int index = threadIdx.x; index < kv_heads * half_dim; index += blockDim.x) {
    const int head = index / half_dim;
    const int dim = index % half_dim;
    const float frequency = powf(theta, -2.0F * dim / head_dim);
    float s{}, c{};
    sincosf(position * frequency, &s, &c);
    const auto base = (static_cast<std::int64_t>(batch_index) * kv_heads + head) * head_dim;
    const float first = __half2float(key[base + dim]);
    const float second = __half2float(key[base + dim + half_dim]);
    const float rotated_first = first * c - second * s;
    const float rotated_second = second * c + first * s;
    key[base + dim] = __float2half(rotated_first);
    key[base + dim + half_dim] = __float2half(rotated_second);
    const auto first_cache = cache_index(physical_block, layer, 0, head, token_offset, dim,
                                         num_layers, kv_heads, block_tokens, head_dim);
    const auto second_cache = cache_index(physical_block, layer, 0, head, token_offset,
                                          dim + half_dim, num_layers, kv_heads, block_tokens, head_dim);
    cache[first_cache] = __float2half(rotated_first);
    cache[second_cache] = __float2half(rotated_second);
    cache[cache_index(physical_block, layer, 1, head, token_offset, dim,
                      num_layers, kv_heads, block_tokens, head_dim)] = value[base + dim];
    cache[cache_index(physical_block, layer, 1, head, token_offset, dim + half_dim,
                      num_layers, kv_heads, block_tokens, head_dim)] = value[base + dim + half_dim];
  }
}

__global__ void rope_cache_prefill_kernel(
    half* query, half* key, const half* value, half* cache,
    const std::int32_t* table, int query_heads, int kv_heads, int head_dim,
    int block_tokens, int layer, int num_layers, float theta) {
  const int position = blockIdx.x;
  const int physical_block = table[position / block_tokens];
  const int token_offset = position % block_tokens;
  const int half_dim = head_dim / 2;
  for (int index = threadIdx.x; index < query_heads * half_dim; index += blockDim.x) {
    const int head = index / half_dim;
    const int dim = index % half_dim;
    const float frequency = powf(theta, -2.0F * dim / head_dim);
    float s{}, c{};
    sincosf(position * frequency, &s, &c);
    const auto base = (static_cast<std::int64_t>(position) * query_heads + head) * head_dim;
    const float first = __half2float(query[base + dim]);
    const float second = __half2float(query[base + dim + half_dim]);
    query[base + dim] = __float2half(first * c - second * s);
    query[base + dim + half_dim] = __float2half(second * c + first * s);
  }
  for (int index = threadIdx.x; index < kv_heads * half_dim; index += blockDim.x) {
    const int head = index / half_dim;
    const int dim = index % half_dim;
    const float frequency = powf(theta, -2.0F * dim / head_dim);
    float s{}, c{};
    sincosf(position * frequency, &s, &c);
    const auto base = (static_cast<std::int64_t>(position) * kv_heads + head) * head_dim;
    const float first = __half2float(key[base + dim]);
    const float second = __half2float(key[base + dim + half_dim]);
    const float rotated_first = first * c - second * s;
    const float rotated_second = second * c + first * s;
    key[base + dim] = __float2half(rotated_first);
    key[base + dim + half_dim] = __float2half(rotated_second);
    cache[cache_index(physical_block, layer, 0, head, token_offset, dim,
                      num_layers, kv_heads, block_tokens, head_dim)] = __float2half(rotated_first);
    cache[cache_index(physical_block, layer, 0, head, token_offset, dim + half_dim,
                      num_layers, kv_heads, block_tokens, head_dim)] = __float2half(rotated_second);
    cache[cache_index(physical_block, layer, 1, head, token_offset, dim,
                      num_layers, kv_heads, block_tokens, head_dim)] = value[base + dim];
    cache[cache_index(physical_block, layer, 1, head, token_offset, dim + half_dim,
                      num_layers, kv_heads, block_tokens, head_dim)] = value[base + dim + half_dim];
  }
}

__global__ void paged_attention_kernel(const half* query, const half* cache, half* output,
                                       const std::int32_t* lengths, const std::int32_t* tables,
                                       int query_heads, int kv_heads, int head_dim, int max_blocks,
                                       int block_tokens, int layer, int num_layers) {
  __shared__ float logits[kMaxContext];
  const int batch_index = blockIdx.x;
  const int query_head = blockIdx.y;
  const int kv_head = query_head / (query_heads / kv_heads);
  const int length = lengths[batch_index];
  const auto q_base = (static_cast<std::int64_t>(batch_index) * query_heads + query_head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    const int physical = tables[batch_index * max_blocks + token / block_tokens];
    float dot = 0.0F;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += __half2float(query[q_base + dim]) *
             __half2float(cache[cache_index(physical, layer, 0, kv_head, token % block_tokens,
                                            dim, num_layers, kv_heads, block_tokens, head_dim)]);
    }
    logits[token] = dot * scale;
  }
  __syncthreads();
  float local_max = -FLT_MAX;
  for (int token = threadIdx.x; token < length; token += blockDim.x) local_max = fmaxf(local_max, logits[token]);
  const float maximum = block_max(local_max);
  float local_sum = 0.0F;
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    logits[token] = expf(logits[token] - maximum);
    local_sum += logits[token];
  }
  const float inverse_sum = 1.0F / block_sum(local_sum);
  __syncthreads();
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float value = 0.0F;
    for (int token = 0; token < length; ++token) {
      const int physical = tables[batch_index * max_blocks + token / block_tokens];
      value += logits[token] * inverse_sum *
               __half2float(cache[cache_index(physical, layer, 1, kv_head, token % block_tokens,
                                              dim, num_layers, kv_heads, block_tokens, head_dim)]);
    }
    output[q_base + dim] = __float2half(value);
  }
}

__global__ void paged_prefill_attention_kernel(
    const half* query, const half* cache, half* output, const std::int32_t* table,
    int query_heads, int kv_heads, int head_dim, int block_tokens,
    int layer, int num_layers) {
  __shared__ float logits[kMaxContext];
  const int query_token = blockIdx.x;
  const int query_head = blockIdx.y;
  const int kv_head = query_head / (query_heads / kv_heads);
  const int length = query_token + 1;
  const auto q_base = (static_cast<std::int64_t>(query_token) * query_heads + query_head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    const int physical = table[token / block_tokens];
    float dot = 0.0F;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += __half2float(query[q_base + dim]) *
             __half2float(cache[cache_index(physical, layer, 0, kv_head, token % block_tokens,
                                            dim, num_layers, kv_heads, block_tokens, head_dim)]);
    }
    logits[token] = dot * scale;
  }
  __syncthreads();
  float local_max = -FLT_MAX;
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    local_max = fmaxf(local_max, logits[token]);
  }
  const float maximum = block_max(local_max);
  float local_sum = 0.0F;
  for (int token = threadIdx.x; token < length; token += blockDim.x) {
    logits[token] = expf(logits[token] - maximum);
    local_sum += logits[token];
  }
  const float inverse_sum = 1.0F / block_sum(local_sum);
  __syncthreads();
  for (int dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
    float value = 0.0F;
    for (int token = 0; token < length; ++token) {
      const int physical = table[token / block_tokens];
      value += logits[token] * inverse_sum *
               __half2float(cache[cache_index(physical, layer, 1, kv_head, token % block_tokens,
                                              dim, num_layers, kv_heads, block_tokens, head_dim)]);
    }
    output[q_base + dim] = __float2half(value);
  }
}

__global__ void causal_softmax_kernel(half* scores, const std::int32_t* lengths,
                                      int heads, int query_length, int key_length) {
  const int batch_index = blockIdx.x / (heads * query_length);
  const int within = blockIdx.x % (heads * query_length);
  const int query = within % query_length;
  const int valid = min(lengths[batch_index], query + 1);
  const auto base = static_cast<std::int64_t>(blockIdx.x) * key_length;
  float local_max = -FLT_MAX;
  for (int key = threadIdx.x; key < valid; key += blockDim.x) local_max = fmaxf(local_max, __half2float(scores[base + key]));
  const float maximum = block_max(local_max);
  float local_sum = 0.0F;
  for (int key = threadIdx.x; key < valid; key += blockDim.x) local_sum += expf(__half2float(scores[base + key]) - maximum);
  const float inverse = 1.0F / block_sum(local_sum);
  for (int key = threadIdx.x; key < key_length; key += blockDim.x) {
    scores[base + key] = key < valid ? __float2half(expf(__half2float(scores[base + key]) - maximum) * inverse)
                                     : __float2half(0.0F);
  }
}

struct Candidate { float value; int index; };

__global__ void argmax_kernel(const half* logits, std::int32_t* tokens, int vocab_size) {
  __shared__ Candidate candidates[kThreads];
  Candidate best{-FLT_MAX, 0};
  const auto base = static_cast<std::int64_t>(blockIdx.x) * vocab_size;
  for (int index = threadIdx.x; index < vocab_size; index += blockDim.x) {
    const float value = __half2float(logits[base + index]);
    if (value > best.value || (value == best.value && index < best.index)) best = {value, index};
  }
  candidates[threadIdx.x] = best;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const auto other = candidates[threadIdx.x + stride];
      if (other.value > candidates[threadIdx.x].value ||
          (other.value == candidates[threadIdx.x].value && other.index < candidates[threadIdx.x].index)) {
        candidates[threadIdx.x] = other;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) tokens[blockIdx.x] = candidates[0].index;
}

void check_launch() { FORGE_CUDA(cudaGetLastError()); }

}  // namespace

void launch_embedding(const std::int32_t* tokens, const half* table, half* output,
                      int token_count, int hidden_size, cudaStream_t stream) {
  const int n = token_count * hidden_size;
  embedding_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, stream>>>(tokens, table, output, token_count, hidden_size);
  check_launch();
}

void launch_rmsnorm(const half* input, const half* weight, half* output, int rows,
                    int hidden_size, float epsilon, cudaStream_t stream) {
  rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(input, weight, output, hidden_size, epsilon);
  check_launch();
}

void launch_residual_rmsnorm(half* residual, const half* update, const half* weight,
                             half* normalized, int rows, int hidden_size,
                             float epsilon, cudaStream_t stream) {
  residual_rmsnorm_kernel<<<rows, kThreads, 0, stream>>>(residual, update, weight, normalized, hidden_size, epsilon);
  check_launch();
}

void launch_silu_mul(const half* gate, const half* up, half* output, std::int64_t elements,
                     cudaStream_t stream) {
  silu_mul_kernel<<<static_cast<unsigned>((elements + kThreads - 1) / kThreads), kThreads, 0, stream>>>(gate, up, output, elements);
  check_launch();
}

void launch_add(half* destination, const half* update, std::int64_t elements, cudaStream_t stream) {
  add_kernel<<<static_cast<unsigned>((elements + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      destination, update, elements);
  check_launch();
}

void launch_rope_cache_write(half* query, half* key, const half* value, half* kv_cache,
                             const std::int32_t* positions, const std::int32_t* block_table,
                             int batch, int query_heads, int kv_heads, int head_dim, int max_blocks,
                             int block_tokens, int layer, int num_layers, float rope_theta,
                             cudaStream_t stream) {
  rope_cache_kernel<<<batch, kThreads, 0, stream>>>(query, key, value, kv_cache, positions, block_table,
      query_heads, kv_heads, head_dim, max_blocks, block_tokens, layer, num_layers, rope_theta);
  check_launch();
}

void launch_rope_cache_write_prefill(
    half* query, half* key, const half* value, half* kv_cache,
    const std::int32_t* block_table, int token_count, int query_heads,
    int kv_heads, int head_dim, int block_tokens, int layer,
    int num_layers, float rope_theta, cudaStream_t stream) {
  rope_cache_prefill_kernel<<<token_count, kThreads, 0, stream>>>(
      query, key, value, kv_cache, block_table, query_heads, kv_heads, head_dim,
      block_tokens, layer, num_layers, rope_theta);
  check_launch();
}

void launch_paged_decode_attention(const half* query, const half* kv_cache, half* output,
                                   const std::int32_t* context_lengths, const std::int32_t* block_table,
                                   int batch, int query_heads, int kv_heads, int head_dim, int max_blocks,
                                   int block_tokens, int layer, int num_layers, cudaStream_t stream) {
  check(head_dim <= kThreads, "paged attention head dimension exceeds kernel limit");
  paged_attention_kernel<<<dim3(batch, query_heads), kThreads, 0, stream>>>(query, kv_cache, output,
      context_lengths, block_table, query_heads, kv_heads, head_dim, max_blocks,
      block_tokens, layer, num_layers);
  check_launch();
}

void launch_paged_prefill_attention(
    const half* query, const half* kv_cache, half* output,
    const std::int32_t* block_table, int token_count, int query_heads,
    int kv_heads, int head_dim, int block_tokens, int layer,
    int num_layers, cudaStream_t stream) {
  check(token_count > 0 && token_count <= kMaxContext, "prefill length exceeds kernel limit");
  check(head_dim <= kThreads, "prefill attention head dimension exceeds kernel limit");
  paged_prefill_attention_kernel<<<dim3(token_count, query_heads), kThreads, 0, stream>>>(
      query, kv_cache, output, block_table, query_heads, kv_heads, head_dim,
      block_tokens, layer, num_layers);
  check_launch();
}

void launch_causal_softmax(half* scores, const std::int32_t* lengths, int batch, int heads,
                           int query_length, int key_length, cudaStream_t stream) {
  check(key_length <= kMaxContext, "softmax key length exceeds kernel limit");
  causal_softmax_kernel<<<batch * heads * query_length, kThreads, 0, stream>>>(
      scores, lengths, heads, query_length, key_length);
  check_launch();
}

void launch_argmax(const half* logits, std::int32_t* tokens, int batch, int vocab_size,
                   cudaStream_t stream) {
  argmax_kernel<<<batch, kThreads, 0, stream>>>(logits, tokens, vocab_size);
  check_launch();
}

}  // namespace forge
