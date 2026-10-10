#pragma once

#include "forge/kv_cache.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace forge {

enum class SequenceState : std::uint8_t { waiting, prefilling, running, completed, cancelled, rejected };

struct Sequence {
  RequestId id{};
  SequenceState state{SequenceState::waiting};
  std::vector<std::int32_t> prompt_tokens;
  std::vector<std::int32_t> output_tokens;
  std::unordered_set<std::int32_t> eos_token_ids;
  std::uint32_t max_new_tokens{};

  [[nodiscard]] std::uint32_t live_tokens() const {
    return static_cast<std::uint32_t>(prompt_tokens.size() + output_tokens.size());
  }
  [[nodiscard]] std::uint32_t maximum_tokens() const {
    return static_cast<std::uint32_t>(prompt_tokens.size()) + max_new_tokens;
  }
};

struct Schedule {
  std::optional<RequestId> prefill;
  std::vector<RequestId> decode;
};

struct SchedulerStats {
  std::uint64_t submitted{};
  std::uint64_t completed{};
  std::uint64_t cancelled{};
  std::uint64_t rejected{};
  std::uint32_t waiting{};
  std::uint32_t running{};
};

class Scheduler {
 public:
  Scheduler(std::uint32_t max_sequences, std::uint32_t max_model_length, KVBlockPool& cache);

  [[nodiscard]] RequestId submit(std::span<const std::int32_t> prompt,
                                 std::uint32_t max_new_tokens,
                                 std::span<const std::int32_t> eos_token_ids);
  [[nodiscard]] Schedule next();
  void finish_prefill(RequestId id);
  [[nodiscard]] bool append_token(RequestId id, std::int32_t token);
  void cancel(RequestId id);
  void forget(RequestId id);
  [[nodiscard]] const Sequence& sequence(RequestId id) const;
  [[nodiscard]] SchedulerStats stats() const;

 private:
  void finish(Sequence& sequence, SequenceState terminal_state);
  Sequence& mutable_sequence(RequestId id);

  std::uint32_t max_sequences_{};
  std::uint32_t max_model_length_{};
  KVBlockPool& cache_;
  RequestId next_id_{1};
  std::deque<RequestId> waiting_;
  std::vector<RequestId> running_;
  std::unordered_map<RequestId, Sequence> sequences_;
  SchedulerStats totals_{};
};

}  // namespace forge

