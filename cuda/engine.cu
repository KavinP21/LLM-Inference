#include "forge/engine.h"

#include "forge/error.h"
#include "forge/nvtx.h"

#include <algorithm>
#include <limits>

namespace forge {

Engine::Engine(const std::filesystem::path& model_path, std::uint32_t max_num_sequences,
               std::uint32_t max_model_length, std::uint64_t kv_cache_bytes)
    : file_(model_path),
      cache_(kv_cache_bytes, file_.config().num_layers, file_.config().num_kv_heads,
             file_.config().head_dim()),
      scheduler_(max_num_sequences, max_model_length, cache_),
      model_(file_, max_num_sequences, max_model_length, cache_.stats().total_blocks),
      max_model_length_(max_model_length),
      max_blocks_per_sequence_((max_model_length + cache_.block_tokens() - 1U) /
                               cache_.block_tokens()) {
  check(max_model_length <= file_.config().max_position_embeddings,
        "runtime length exceeds model position limit");
  token_scratch_.reserve(max_num_sequences);
  position_scratch_.reserve(max_num_sequences);
  length_scratch_.reserve(max_num_sequences);
  table_scratch_.reserve(static_cast<std::size_t>(max_num_sequences) * max_blocks_per_sequence_);
}

RequestId Engine::submit(std::span<const std::int32_t> input_ids,
                         std::uint32_t max_new_tokens,
                         std::span<const std::int32_t> eos_token_ids) {
  const auto id = scheduler_.submit(input_ids, max_new_tokens, eos_token_ids);
  check(scheduler_.sequence(id).state != SequenceState::rejected,
        "request rejected: invalid length, concurrency limit, or insufficient KV capacity");
  return id;
}

void Engine::append_padded_table(RequestId id, std::vector<std::int32_t>& destination) const {
  const auto& source = cache_.block_table(id);
  check(source.size() <= max_blocks_per_sequence_, "KV block table exceeds configured length");
  destination.insert(destination.end(), source.begin(), source.end());
  destination.insert(destination.end(), max_blocks_per_sequence_ - source.size(), -1);
}

std::vector<TokenEvent> Engine::step() {
  const auto schedule = scheduler_.next();
  std::vector<TokenEvent> events;
  events.reserve(schedule.decode.size() + (schedule.prefill.has_value() ? 1U : 0U));
  try {

  if (!schedule.decode.empty()) {
    const NvtxRange decode_range("engine.decode", 0xff54a24bU);
    token_scratch_.clear();
    position_scratch_.clear();
    length_scratch_.clear();
    table_scratch_.clear();
    for (const auto id : schedule.decode) {
      const auto& sequence = scheduler_.sequence(id);
      check(!sequence.output_tokens.empty(), "running sequence has no generated input token");
      token_scratch_.push_back(sequence.output_tokens.back());
      position_scratch_.push_back(static_cast<std::int32_t>(sequence.live_tokens() - 1U));
      length_scratch_.push_back(static_cast<std::int32_t>(sequence.live_tokens()));
      append_padded_table(id, table_scratch_);
    }
    const auto generated = model_.forward(token_scratch_, position_scratch_, length_scratch_,
                                          table_scratch_);
    for (std::size_t i = 0; i < generated.size(); ++i) {
      const bool finished = scheduler_.append_token(schedule.decode[i], generated[i]);
      events.push_back({schedule.decode[i], generated[i], finished});
    }
  }

  if (schedule.prefill.has_value()) {
    const NvtxRange prefill_range("engine.prefill", 0xfff58518U);
    const auto id = *schedule.prefill;
    const auto& sequence = scheduler_.sequence(id);
    table_scratch_.clear();
    append_padded_table(id, table_scratch_);
    const auto generated = model_.prefill(sequence.prompt_tokens, table_scratch_);
    scheduler_.finish_prefill(id);
    const bool finished = scheduler_.append_token(id, generated);
    events.push_back({id, generated, finished});
  }
  } catch (...) {
    for (const auto id : schedule.decode) scheduler_.cancel(id);
    if (schedule.prefill.has_value()) scheduler_.cancel(*schedule.prefill);
    throw;
  }
  return events;
}

std::vector<std::int32_t> Engine::generate(
    std::span<const std::int32_t> input_ids, std::uint32_t max_new_tokens,
    std::span<const std::int32_t> eos_token_ids) {
  const auto id = submit(input_ids, max_new_tokens, eos_token_ids);
  while (true) {
    static_cast<void>(step());
    const auto& sequence = scheduler_.sequence(id);
    if (sequence.state == SequenceState::completed) return sequence.output_tokens;
    check(sequence.state != SequenceState::cancelled && sequence.state != SequenceState::rejected,
          "generation terminated without output");
  }
}

std::vector<float> Engine::debug_prefill_logits(std::span<const std::int32_t> input_ids) {
  const auto scheduler_stats = scheduler_.stats();
  check(scheduler_stats.waiting == 0 && scheduler_stats.running == 0,
        "debug logits require an idle engine");
  check(!input_ids.empty() && input_ids.size() <= max_model_length_,
        "invalid debug prompt length");
  constexpr auto debug_id = std::numeric_limits<RequestId>::max();
  check(cache_.reserve(debug_id, static_cast<std::uint32_t>(input_ids.size())),
        "insufficient KV cache for debug logits");
  try {
    check(cache_.ensure_tokens(debug_id, static_cast<std::uint32_t>(input_ids.size())),
          "debug KV allocation unexpectedly failed");
    table_scratch_.clear();
    append_padded_table(debug_id, table_scratch_);
    static_cast<void>(model_.prefill(input_ids, table_scratch_));
    auto logits = model_.copy_logits();
    cache_.release(debug_id);
    return logits;
  } catch (...) {
    cache_.release(debug_id);
    throw;
  }
}

EngineStats Engine::stats() const {
  return {scheduler_.stats(), cache_.stats(), model_.model_bytes(),
          model_.kv_bytes()};
}

}  // namespace forge
