#pragma once

#include "forge/model_file.h"
#include "forge/qwen_model.h"
#include "forge/scheduler.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace forge {

struct TokenEvent {
  RequestId request_id{};
  std::int32_t token{};
  bool finished{};
};

struct EngineStats {
  SchedulerStats scheduler;
  KVCacheStats kv_cache;
  std::uint64_t model_bytes{};
  std::uint64_t kv_device_bytes{};
};

class Engine {
 public:
  Engine(const std::filesystem::path& model_path, std::uint32_t max_num_sequences = 16,
         std::uint32_t max_model_length = 2048,
         std::uint64_t kv_cache_bytes = 512ULL << 20U);

  [[nodiscard]] RequestId submit(std::span<const std::int32_t> input_ids,
                                 std::uint32_t max_new_tokens,
                                 std::span<const std::int32_t> eos_token_ids);
  [[nodiscard]] std::vector<TokenEvent> step();
  void cancel(RequestId id) { scheduler_.cancel(id); }
  void forget(RequestId id) { scheduler_.forget(id); }
  [[nodiscard]] std::vector<std::int32_t> generate(
      std::span<const std::int32_t> input_ids, std::uint32_t max_new_tokens,
      std::span<const std::int32_t> eos_token_ids);
  // Test-only path. Requires an idle engine and releases all temporary KV blocks before returning.
  [[nodiscard]] std::vector<float> debug_prefill_logits(
      std::span<const std::int32_t> input_ids);
  [[nodiscard]] const Sequence& sequence(RequestId id) const { return scheduler_.sequence(id); }
  [[nodiscard]] EngineStats stats() const;

 private:
  void append_padded_table(RequestId id, std::vector<std::int32_t>& destination) const;

  ModelFile file_;
  KVBlockPool cache_;
  Scheduler scheduler_;
  QwenModel model_;
  std::uint32_t max_model_length_{};
  std::uint32_t max_blocks_per_sequence_{};
  std::vector<std::int32_t> token_scratch_;
  std::vector<std::int32_t> position_scratch_;
  std::vector<std::int32_t> length_scratch_;
  std::vector<std::int32_t> table_scratch_;
};

}  // namespace forge
