#pragma once

#include "forge/cuda_arena.h"
#include "forge/execution_context.h"
#include "forge/model_weights.h"

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <span>
#include <vector>

namespace forge {

class QwenModel {
 public:
  QwenModel(const ModelFile& file, std::uint32_t max_batch, std::uint32_t max_model_length,
            std::uint32_t kv_blocks, std::uint32_t block_tokens = 16);
  ~QwenModel() = default;
  QwenModel(const QwenModel&) = delete;
  QwenModel& operator=(const QwenModel&) = delete;

  // Executes one autoregressive token for each sequence and returns greedy next tokens.
  [[nodiscard]] std::vector<std::int32_t> forward(
      std::span<const std::int32_t> tokens, std::span<const std::int32_t> positions,
      std::span<const std::int32_t> context_lengths,
      std::span<const std::int32_t> block_tables);
  [[nodiscard]] std::int32_t prefill(std::span<const std::int32_t> tokens,
                                     std::span<const std::int32_t> block_table);
  [[nodiscard]] std::vector<float> copy_logits(std::uint32_t batch_index = 0);

  [[nodiscard]] const ModelConfig& config() const { return weights_.config(); }
  [[nodiscard]] std::uint64_t kv_bytes() const { return kv_bytes_; }
  [[nodiscard]] std::uint64_t model_bytes() const { return weights_.device_bytes(); }

 private:
  const half* weight(const std::string& name) const;
  half* allocate_half(std::size_t elements);

  std::uint32_t max_batch_{};
  std::uint32_t max_model_length_{};
  std::uint32_t max_blocks_per_sequence_{};
  std::uint32_t block_tokens_{};
  std::uint32_t activation_rows_{};
  ExecutionContext context_;
  ModelWeights weights_;
  CudaArena activations_;
  CudaArena kv_cache_;
  std::uint64_t kv_bytes_{};
  std::int32_t* device_tokens_{};
  std::int32_t* device_positions_{};
  std::int32_t* device_lengths_{};
  std::int32_t* device_tables_{};
  std::int32_t* device_output_tokens_{};
  half* hidden_{};
  half* normalized_{};
  half* query_{};
  half* key_{};
  half* value_{};
  half* attention_{};
  half* update_{};
  half* gate_{};
  half* up_{};
  half* mlp_{};
  half* logits_{};
};

}  // namespace forge
