#include "forge/qwen_model.h"

#include "forge/cuda_utils.h"
#include "forge/error.h"
#include "forge/kernels.h"
#include "forge/nvtx.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <string>

namespace forge {
namespace {

std::size_t activation_bytes(const ModelConfig& c, std::uint32_t rows,
                             std::uint32_t max_batch, std::uint32_t max_blocks) {
  const auto half_count = static_cast<std::size_t>(rows) *
      (5ULL * c.hidden_size + 2ULL * c.num_kv_heads * c.head_dim() +
       3ULL * c.intermediate_size) + static_cast<std::size_t>(max_batch) * c.vocab_size;
  const auto integer_count = static_cast<std::size_t>(rows) +
      static_cast<std::size_t>(max_batch) * (3ULL + max_blocks);
  return half_count * sizeof(half) + integer_count * sizeof(std::int32_t) + 4096U;
}

std::size_t kv_capacity(const ModelConfig& c, std::uint32_t blocks,
                        std::uint32_t block_tokens) {
  return static_cast<std::size_t>(blocks) * c.num_layers * 2ULL * c.num_kv_heads *
         block_tokens * c.head_dim() * sizeof(half);
}

}  // namespace

QwenModel::QwenModel(const ModelFile& file, std::uint32_t max_batch,
                     std::uint32_t max_model_length, std::uint32_t kv_blocks,
                     std::uint32_t block_tokens)
    : max_batch_(max_batch), max_model_length_(max_model_length),
      max_blocks_per_sequence_((max_model_length + block_tokens - 1U) / block_tokens),
      block_tokens_(block_tokens), activation_rows_(std::max(max_batch, max_model_length)),
      context_(), weights_(file, context_.stream()),
      activations_(activation_bytes(file.config(), activation_rows_, max_batch_,
                                    max_blocks_per_sequence_)),
      kv_cache_(kv_capacity(file.config(), kv_blocks, block_tokens)) {
  check(max_batch > 0 && max_model_length > 0 && max_model_length <= 2048,
        "Qwen v1 supports batch > 0 and maximum length <= 2048");
  check(kv_blocks > 0, "KV cache needs at least one block");
  const auto& c = config();
  kv_bytes_ = kv_cache_.capacity();
  FORGE_CUDA(cudaMemsetAsync(kv_cache_.data(), 0, kv_bytes_, context_.stream()));

  device_tokens_ = static_cast<std::int32_t*>(activations_.allocate(activation_rows_ * sizeof(std::int32_t)));
  device_positions_ = static_cast<std::int32_t*>(activations_.allocate(max_batch_ * sizeof(std::int32_t)));
  device_lengths_ = static_cast<std::int32_t*>(activations_.allocate(max_batch_ * sizeof(std::int32_t)));
  device_tables_ = static_cast<std::int32_t*>(activations_.allocate(
      static_cast<std::size_t>(max_batch_) * max_blocks_per_sequence_ * sizeof(std::int32_t)));
  device_output_tokens_ = static_cast<std::int32_t*>(activations_.allocate(max_batch_ * sizeof(std::int32_t)));
  hidden_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.hidden_size);
  normalized_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.hidden_size);
  query_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.hidden_size);
  const auto kv_width = c.num_kv_heads * c.head_dim();
  key_ = allocate_half(static_cast<std::size_t>(activation_rows_) * kv_width);
  value_ = allocate_half(static_cast<std::size_t>(activation_rows_) * kv_width);
  attention_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.hidden_size);
  update_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.hidden_size);
  gate_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.intermediate_size);
  up_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.intermediate_size);
  mlp_ = allocate_half(static_cast<std::size_t>(activation_rows_) * c.intermediate_size);
  logits_ = allocate_half(static_cast<std::size_t>(max_batch_) * c.vocab_size);
  context_.synchronize();
}

half* QwenModel::allocate_half(std::size_t elements) {
  return static_cast<half*>(activations_.allocate(elements * sizeof(half)));
}

const half* QwenModel::weight(const std::string& name) const {
  return static_cast<const half*>(weights_.tensor(name).data);
}

std::vector<std::int32_t> QwenModel::forward(
    std::span<const std::int32_t> tokens, std::span<const std::int32_t> positions,
    std::span<const std::int32_t> context_lengths,
    std::span<const std::int32_t> block_tables) {
  const auto batch = static_cast<int>(tokens.size());
  check(batch > 0 && batch <= static_cast<int>(max_batch_), "invalid forward batch size");
  check(positions.size() == tokens.size() && context_lengths.size() == tokens.size(),
        "token metadata batch mismatch");
  check(block_tables.size() == tokens.size() * max_blocks_per_sequence_, "block table shape mismatch");
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    check(tokens[i] >= 0 && tokens[i] < static_cast<std::int32_t>(config().vocab_size), "token id out of range");
    check(positions[i] >= 0 && positions[i] < static_cast<std::int32_t>(max_model_length_), "position out of range");
    check(context_lengths[i] == positions[i] + 1, "decode context length must equal position + 1");
  }
  const auto stream = context_.stream();
  FORGE_CUDA(cudaMemcpyAsync(device_tokens_, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream));
  FORGE_CUDA(cudaMemcpyAsync(device_positions_, positions.data(), positions.size_bytes(), cudaMemcpyHostToDevice, stream));
  FORGE_CUDA(cudaMemcpyAsync(device_lengths_, context_lengths.data(), context_lengths.size_bytes(), cudaMemcpyHostToDevice, stream));
  FORGE_CUDA(cudaMemcpyAsync(device_tables_, block_tables.data(), block_tables.size_bytes(), cudaMemcpyHostToDevice, stream));

  const auto& c = config();
  const auto hidden_elements = static_cast<std::int64_t>(batch) * c.hidden_size;
  const auto intermediate_elements = static_cast<std::int64_t>(batch) * c.intermediate_size;
  const auto kv_width = c.num_kv_heads * c.head_dim();
  launch_embedding(device_tokens_, weight("model.embed_tokens.weight"), hidden_, batch, c.hidden_size, stream);
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    const NvtxRange layer_range("qwen.decoder_layer", 0xff4c78a8U);
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    launch_rmsnorm(hidden_, weight(prefix + "input_layernorm.weight"), normalized_,
                   batch, c.hidden_size, c.rms_norm_eps, stream);
    context_.linear_fp16(normalized_, weight(prefix + "self_attn.q_proj.weight"),
                         weight(prefix + "self_attn.q_proj.bias"), query_, batch, c.hidden_size, c.hidden_size);
    context_.linear_fp16(normalized_, weight(prefix + "self_attn.k_proj.weight"),
                         weight(prefix + "self_attn.k_proj.bias"), key_, batch, kv_width, c.hidden_size);
    context_.linear_fp16(normalized_, weight(prefix + "self_attn.v_proj.weight"),
                         weight(prefix + "self_attn.v_proj.bias"), value_, batch, kv_width, c.hidden_size);
    launch_rope_cache_write(query_, key_, value_, static_cast<half*>(kv_cache_.data()), device_positions_,
                            device_tables_, batch, c.num_attention_heads, c.num_kv_heads, c.head_dim(),
                            max_blocks_per_sequence_, block_tokens_, layer, c.num_layers, c.rope_theta, stream);
    launch_paged_decode_attention(query_, static_cast<const half*>(kv_cache_.data()), attention_, device_lengths_,
                                  device_tables_, batch, c.num_attention_heads, c.num_kv_heads, c.head_dim(),
                                  max_blocks_per_sequence_, block_tokens_, layer, c.num_layers, stream);
    context_.linear_fp16(attention_, weight(prefix + "self_attn.o_proj.weight"), nullptr,
                         update_, batch, c.hidden_size, c.hidden_size);
    launch_residual_rmsnorm(hidden_, update_, weight(prefix + "post_attention_layernorm.weight"),
                            normalized_, batch, c.hidden_size, c.rms_norm_eps, stream);
    context_.linear_fp16(normalized_, weight(prefix + "mlp.gate_proj.weight"), nullptr,
                         gate_, batch, c.intermediate_size, c.hidden_size);
    context_.linear_fp16(normalized_, weight(prefix + "mlp.up_proj.weight"), nullptr,
                         up_, batch, c.intermediate_size, c.hidden_size);
    launch_silu_mul(gate_, up_, mlp_, intermediate_elements, stream);
    context_.linear_fp16(mlp_, weight(prefix + "mlp.down_proj.weight"), nullptr,
                         update_, batch, c.hidden_size, c.intermediate_size);
    launch_add(hidden_, update_, hidden_elements, stream);
  }
  launch_rmsnorm(hidden_, weight("model.norm.weight"), normalized_, batch,
                 c.hidden_size, c.rms_norm_eps, stream);
  {
    const NvtxRange sampling_range("qwen.lm_head_argmax", 0xffe45756U);
    context_.linear_fp16(normalized_, weight("lm_head.weight"), nullptr,
                         logits_, batch, c.vocab_size, c.hidden_size);
    launch_argmax(logits_, device_output_tokens_, batch, c.vocab_size, stream);
  }
  std::vector<std::int32_t> output(tokens.size());
  FORGE_CUDA(cudaMemcpyAsync(output.data(), device_output_tokens_, output.size() * sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost, stream));
  context_.synchronize();
  return output;
}

std::int32_t QwenModel::prefill(std::span<const std::int32_t> tokens,
                                std::span<const std::int32_t> block_table) {
  const auto rows = static_cast<int>(tokens.size());
  check(rows > 0 && rows <= static_cast<int>(max_model_length_), "invalid prefill length");
  check(block_table.size() == max_blocks_per_sequence_, "prefill block table shape mismatch");
  for (const auto token : tokens) {
    check(token >= 0 && token < static_cast<std::int32_t>(config().vocab_size), "token id out of range");
  }
  const auto stream = context_.stream();
  FORGE_CUDA(cudaMemcpyAsync(device_tokens_, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream));
  FORGE_CUDA(cudaMemcpyAsync(device_tables_, block_table.data(), block_table.size_bytes(),
                            cudaMemcpyHostToDevice, stream));

  const auto& c = config();
  const auto hidden_elements = static_cast<std::int64_t>(rows) * c.hidden_size;
  const auto intermediate_elements = static_cast<std::int64_t>(rows) * c.intermediate_size;
  const auto kv_width = c.num_kv_heads * c.head_dim();
  launch_embedding(device_tokens_, weight("model.embed_tokens.weight"), hidden_, rows, c.hidden_size, stream);
  for (std::uint32_t layer = 0; layer < c.num_layers; ++layer) {
    const NvtxRange layer_range("qwen.prefill_decoder_layer", 0xff72b7b2U);
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    launch_rmsnorm(hidden_, weight(prefix + "input_layernorm.weight"), normalized_,
                   rows, c.hidden_size, c.rms_norm_eps, stream);
    context_.linear_fp16(normalized_, weight(prefix + "self_attn.q_proj.weight"),
                         weight(prefix + "self_attn.q_proj.bias"), query_, rows, c.hidden_size, c.hidden_size);
    context_.linear_fp16(normalized_, weight(prefix + "self_attn.k_proj.weight"),
                         weight(prefix + "self_attn.k_proj.bias"), key_, rows, kv_width, c.hidden_size);
    context_.linear_fp16(normalized_, weight(prefix + "self_attn.v_proj.weight"),
                         weight(prefix + "self_attn.v_proj.bias"), value_, rows, kv_width, c.hidden_size);
    launch_rope_cache_write_prefill(query_, key_, value_, static_cast<half*>(kv_cache_.data()),
                                    device_tables_, rows, c.num_attention_heads, c.num_kv_heads,
                                    c.head_dim(), block_tokens_, layer, c.num_layers, c.rope_theta, stream);
    launch_paged_prefill_attention(query_, static_cast<const half*>(kv_cache_.data()), attention_,
                                   device_tables_, rows, c.num_attention_heads, c.num_kv_heads,
                                   c.head_dim(), block_tokens_, layer, c.num_layers, stream);
    context_.linear_fp16(attention_, weight(prefix + "self_attn.o_proj.weight"), nullptr,
                         update_, rows, c.hidden_size, c.hidden_size);
    launch_residual_rmsnorm(hidden_, update_, weight(prefix + "post_attention_layernorm.weight"),
                            normalized_, rows, c.hidden_size, c.rms_norm_eps, stream);
    context_.linear_fp16(normalized_, weight(prefix + "mlp.gate_proj.weight"), nullptr,
                         gate_, rows, c.intermediate_size, c.hidden_size);
    context_.linear_fp16(normalized_, weight(prefix + "mlp.up_proj.weight"), nullptr,
                         up_, rows, c.intermediate_size, c.hidden_size);
    launch_silu_mul(gate_, up_, mlp_, intermediate_elements, stream);
    context_.linear_fp16(mlp_, weight(prefix + "mlp.down_proj.weight"), nullptr,
                         update_, rows, c.hidden_size, c.intermediate_size);
    launch_add(hidden_, update_, hidden_elements, stream);
  }
  launch_rmsnorm(hidden_, weight("model.norm.weight"), normalized_, rows,
                 c.hidden_size, c.rms_norm_eps, stream);
  context_.linear_fp16(normalized_ + static_cast<std::int64_t>(rows - 1) * c.hidden_size,
                       weight("lm_head.weight"), nullptr, logits_, 1, c.vocab_size, c.hidden_size);
  launch_argmax(logits_, device_output_tokens_, 1, c.vocab_size, stream);
  std::int32_t output{};
  FORGE_CUDA(cudaMemcpyAsync(&output, device_output_tokens_, sizeof(output),
                            cudaMemcpyDeviceToHost, stream));
  context_.synchronize();
  return output;
}

std::vector<float> QwenModel::copy_logits(std::uint32_t batch_index) {
  check(batch_index < max_batch_, "logit batch index is out of range");
  const auto count = config().vocab_size;
  std::vector<half> host(count);
  const auto* source = logits_ + static_cast<std::uint64_t>(batch_index) * count;
  FORGE_CUDA(cudaMemcpyAsync(host.data(), source, host.size() * sizeof(half),
                            cudaMemcpyDeviceToHost, context_.stream()));
  context_.synchronize();
  std::vector<float> result(count);
  std::transform(host.begin(), host.end(), result.begin(),
                 [](half value) { return __half2float(value); });
  return result;
}

}  // namespace forge
