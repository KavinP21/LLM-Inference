#pragma once

#include <cstdint>

namespace forge {

struct ModelConfig {
  std::uint32_t vocab_size{};
  std::uint32_t hidden_size{};
  std::uint32_t intermediate_size{};
  std::uint32_t num_layers{};
  std::uint32_t num_attention_heads{};
  std::uint32_t num_kv_heads{};
  std::uint32_t max_position_embeddings{};
  std::uint32_t eos_token_id{};
  float rope_theta{};
  float rms_norm_eps{};

  [[nodiscard]] std::uint32_t head_dim() const {
    return hidden_size / num_attention_heads;
  }
  void validate() const;
};

}  // namespace forge

