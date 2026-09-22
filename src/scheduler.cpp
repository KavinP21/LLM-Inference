#include "forge/scheduler.h"

#include "forge/error.h"

#include <algorithm>

namespace forge {

Scheduler::Scheduler(std::uint32_t max_sequences, std::uint32_t max_model_length,
                     KVBlockPool& cache)
    : max_sequences_(max_sequences), max_model_length_(max_model_length), cache_(cache) {
  check(max_sequences > 0 && max_model_length > 0, "invalid scheduler limits");
}

RequestId Scheduler::submit(std::span<const std::int32_t> prompt, std::uint32_t max_new_tokens,
                            std::span<const std::int32_t> eos_token_ids) {
  const auto id = next_id_++;
  ++totals_.submitted;
  Sequence sequence;
  sequence.id = id;
  sequence.prompt_tokens.assign(prompt.begin(), prompt.end());
  sequence.max_new_tokens = max_new_tokens;
  sequence.eos_token_ids.insert(eos_token_ids.begin(), eos_token_ids.end());
  const bool invalid = prompt.empty() || max_new_tokens == 0 ||
                       prompt.size() > max_model_length_ ||
                       static_cast<std::uint64_t>(prompt.size()) + max_new_tokens > max_model_length_;
  const auto active = std::count_if(sequences_.begin(), sequences_.end(), [](const auto& item) {
    const auto state = item.second.state;
    return state == SequenceState::waiting || state == SequenceState::prefilling ||
           state == SequenceState::running;
  });
  if (invalid || active >= max_sequences_ || !cache_.reserve(id, sequence.maximum_tokens())) {
    sequence.state = SequenceState::rejected;
    ++totals_.rejected;
  } else {
    waiting_.push_back(id);
  }
  sequences_.emplace(id, std::move(sequence));
  return id;
}

Schedule Scheduler::next() {
  Schedule schedule;
  schedule.decode = running_;
  if (!waiting_.empty()) {
    const auto id = waiting_.front();
    waiting_.pop_front();
    auto& sequence = mutable_sequence(id);
    check(sequence.state == SequenceState::waiting, "invalid waiting sequence state");
    sequence.state = SequenceState::prefilling;
    check(cache_.ensure_tokens(id, static_cast<std::uint32_t>(sequence.prompt_tokens.size())),
          "reserved KV allocation unexpectedly failed");
    schedule.prefill = id;
  }
  return schedule;
}

void Scheduler::finish_prefill(RequestId id) {
  auto& sequence = mutable_sequence(id);
  check(sequence.state == SequenceState::prefilling, "sequence is not prefilling");
  sequence.state = SequenceState::running;
  running_.push_back(id);
}

bool Scheduler::append_token(RequestId id, std::int32_t token) {
  auto& sequence = mutable_sequence(id);
  check(sequence.state == SequenceState::running, "sequence is not running");
  sequence.output_tokens.push_back(token);
  const auto stop = sequence.eos_token_ids.contains(token) ||
                    sequence.output_tokens.size() >= sequence.max_new_tokens ||
                    sequence.live_tokens() >= max_model_length_;
  if (stop) {
    finish(sequence, SequenceState::completed);
    return true;
  }
  check(cache_.ensure_tokens(id, sequence.live_tokens()), "reserved KV allocation unexpectedly failed");
  return false;
}

void Scheduler::cancel(RequestId id) {
  auto& sequence = mutable_sequence(id);
  if (sequence.state == SequenceState::completed || sequence.state == SequenceState::cancelled ||
      sequence.state == SequenceState::rejected) return;
  waiting_.erase(std::remove(waiting_.begin(), waiting_.end(), id), waiting_.end());
  finish(sequence, SequenceState::cancelled);
}

const Sequence& Scheduler::sequence(RequestId id) const {
  const auto it = sequences_.find(id);
  check(it != sequences_.end(), "unknown request id");
  return it->second;
}

SchedulerStats Scheduler::stats() const {
  auto result = totals_;
  result.waiting = static_cast<std::uint32_t>(waiting_.size());
  result.running = static_cast<std::uint32_t>(std::count_if(
      sequences_.begin(), sequences_.end(), [](const auto& item) {
        return item.second.state == SequenceState::running ||
               item.second.state == SequenceState::prefilling;
      }));
  return result;
}

void Scheduler::finish(Sequence& sequence, SequenceState terminal_state) {
  check(terminal_state == SequenceState::completed || terminal_state == SequenceState::cancelled,
        "invalid terminal state");
  running_.erase(std::remove(running_.begin(), running_.end(), sequence.id), running_.end());
  cache_.release(sequence.id);
  sequence.state = terminal_state;
  if (terminal_state == SequenceState::completed) ++totals_.completed;
  else ++totals_.cancelled;
}

Sequence& Scheduler::mutable_sequence(RequestId id) {
  const auto it = sequences_.find(id);
  check(it != sequences_.end(), "unknown request id");
  return it->second;
}

}  // namespace forge
