#pragma once

#include <cstdint>

namespace forge {

enum class ModelType : std::uint32_t { qwen2 = 1, gemma3_text = 2 };
enum class ActivationType : std::uint32_t { silu = 1, gelu_pytorch_tanh = 2 };

struct ModelConfig {
  std::uint32_t vocab_size{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t num_layers{};
  std::uint32_t num_attention_heads{};
  std::uint32_t num_kv_heads{};
  std::uint32_t attention_head_dim{};
  std::uint32_t max_position_embeddings{};
  std::uint32_t eos_token_id{};
  ModelType model_type{ModelType::qwen2};
  ActivationType activation{ActivationType::silu};
  std::uint32_t sliding_window{};
  std::uint32_t sliding_window_pattern{};
  float rope_theta{};
  float rope_local_theta{};
  float rms_norm_eps{};
  float query_pre_attn_scalar{};
  float embedding_scale{1.0F};
  float attn_logit_softcapping{};
  float final_logit_softcapping{};
  float norm_weight_offset{};

  [[nodiscard]] std::uint32_t head_dim() const {
    return attention_head_dim != 0 ? attention_head_dim : hidden_size / num_attention_heads;
  }
  void validate() const;
};

}  // namespace forge
