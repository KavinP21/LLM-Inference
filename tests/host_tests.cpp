#include "forge/error.h"
#include "forge/kv_cache.h"
#include "forge/model_file.h"
#include "forge/scheduler.h"
#include "forge/sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <span>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
void expect_throws(Function&& function, std::string_view message) {
  try {
    function();
    expect(false, message);
  } catch (const forge::Error&) {
  }
}

void test_sha256() {
  constexpr std::string_view input = "abc";
  const auto bytes = std::as_bytes(std::span(input.data(), input.size()));
  const auto digest = forge::sha256(bytes);
  constexpr std::array<std::uint8_t, 32> expected = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
      0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
      0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad};
  expect(digest == expected, "SHA-256 known-answer test");

  const auto digest_text = [](std::span<const std::byte> value) {
    const auto hash = forge::sha256(value);
    std::ostringstream output;
    for (const auto byte : hash) {
      output << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    }
    return output.str();
  };
  for (const auto& [length, expected_text] : {
           std::pair{55U, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"},
           std::pair{56U, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"},
           std::pair{64U, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"},
           std::pair{65U, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"}}) {
    const std::string repeated(length, 'a');
    expect(digest_text(std::as_bytes(std::span(repeated.data(), repeated.size()))) == expected_text,
           "SHA-256 padding-boundary test");
  }
}

void test_cache_boundaries() {
  // One block is 2 layers * K/V * 2 heads * 16 tokens * 8 dims * 2 bytes = 2048.
  forge::KVBlockPool pool(2048U * 8U, 2, 2, 8);
  expect(pool.reserve(1, 33), "reserve three blocks");
  expect(pool.ensure_tokens(1, 15), "allocate first block");
  expect(pool.block_table(1).size() == 1, "15 tokens use one block");
  expect(pool.ensure_tokens(1, 16), "fill first block");
  expect(pool.block_table(1).size() == 1, "16 tokens use one block");
  expect(pool.ensure_tokens(1, 17), "cross block boundary");
  expect(pool.block_table(1).size() == 2, "17 tokens use two blocks");
  expect(pool.stats().internal_fragmentation_tokens == 15, "fragmentation is exact");
  pool.release(1);
  expect(pool.stats().allocated_blocks == 0, "release returns physical blocks");
  expect(pool.stats().reserved_blocks == 0, "release returns reservations");
}

void test_reservation_isolation() {
  forge::KVBlockPool pool(2048U * 4U, 2, 2, 8);
  expect(pool.reserve(1, 32), "first reservation");
  expect(pool.reserve(2, 32), "second reservation");
  expect(pool.stats().total_blocks == 4, "reservations do not inflate physical capacity");
  expect(!pool.reserve(3, 1), "reserved capacity cannot be overcommitted");
  expect(pool.ensure_tokens(1, 32), "first allocation consumes own reservation");
  expect(pool.ensure_tokens(2, 32), "second allocation remains guaranteed");
}

void test_scheduler_lifecycle() {
  forge::KVBlockPool pool(2048U * 16U, 2, 2, 8);
  forge::Scheduler scheduler(4, 64, pool);
  const std::array<std::int32_t, 3> prompt = {1, 2, 3};
  const std::array<std::int32_t, 1> eos = {9};
  const auto first = scheduler.submit(prompt, 3, eos);
  auto schedule = scheduler.next();
  expect(schedule.prefill == first, "FCFS request selected for prefill");
  scheduler.finish_prefill(first);
  expect(!scheduler.append_token(first, 4), "ordinary token keeps request running");
  expect(scheduler.append_token(first, 9), "EOS completes request");
  expect(scheduler.sequence(first).state == forge::SequenceState::completed, "terminal state recorded");
  expect(pool.stats().allocated_blocks == 0, "completion releases cache");

  const auto second = scheduler.submit(prompt, 3, eos);
  scheduler.cancel(second);
  expect(scheduler.sequence(second).state == forge::SequenceState::cancelled, "waiting cancellation");
  expect(!scheduler.next().prefill.has_value(), "cancelled request removed from queue");
}

void test_scheduler_limits_and_reuse() {
  forge::KVBlockPool pool(2048U * 4U, 2, 2, 8);
  forge::Scheduler scheduler(1, 32, pool);
  const std::array<std::int32_t, 2> prompt = {1, 2};
  const std::array<std::int32_t, 1> eos = {99};
  const auto first = scheduler.submit(prompt, 2, eos);
  const auto first_schedule = scheduler.next();
  expect(first_schedule.prefill == first, "first request starts prefill");
  const auto rejected_while_prefilling = scheduler.submit(prompt, 2, eos);
  expect(scheduler.sequence(rejected_while_prefilling).state == forge::SequenceState::rejected,
         "prefilling request counts against concurrency limit");
  scheduler.cancel(first);
  expect(pool.stats().allocated_blocks == 0 && pool.stats().reserved_blocks == 0,
         "cancelling prefill releases allocated and reserved blocks");
  const auto replacement = scheduler.submit(prompt, 2, eos);
  expect(scheduler.sequence(replacement).state == forge::SequenceState::waiting,
         "capacity is reusable after cancellation");

  std::vector<std::int32_t> oversized(33, 1);
  const auto too_long = scheduler.submit(oversized, 1, eos);
  expect(scheduler.sequence(too_long).state == forge::SequenceState::rejected,
         "oversized prompt rejected without integer truncation");
  const std::array<std::int32_t, 0> empty{};
  const auto empty_request = scheduler.submit(empty, 1, eos);
  expect(scheduler.sequence(empty_request).state == forge::SequenceState::rejected,
         "empty prompt rejected");
  expect_throws([&] { static_cast<void>(scheduler.sequence(999999)); },
                "unknown sequence lookup throws");
}

void test_scheduler_forget_terminal_history() {
  forge::KVBlockPool pool(2048U * 4U, 2, 2, 8);
  forge::Scheduler scheduler(2, 32, pool);
  const std::array<std::int32_t, 2> prompt{1, 2};
  const std::array<std::int32_t, 0> no_eos{};
  const auto id = scheduler.submit(prompt, 2, no_eos);
  expect_throws([&] { scheduler.forget(id); }, "cannot forget active request");
  scheduler.cancel(id);
  scheduler.forget(id);
  expect_throws([&] { static_cast<void>(scheduler.sequence(id)); }, "forgotten request is unavailable");
  expect(scheduler.stats().cancelled == 1, "forget preserves lifetime totals");
  expect(pool.stats().allocated_blocks == 0, "forget leaves cache reclaimed");
}

void test_scheduler_fcfs_continuous_admission() {
  forge::KVBlockPool pool(2048U * 16U, 2, 2, 8);
  forge::Scheduler scheduler(3, 64, pool);
  const std::array<std::int32_t, 2> prompt = {1, 2};
  const std::array<std::int32_t, 0> no_eos{};
  const auto first = scheduler.submit(prompt, 4, no_eos);
  const auto second = scheduler.submit(prompt, 4, no_eos);
  const auto third = scheduler.submit(prompt, 4, no_eos);
  auto schedule = scheduler.next();
  expect(schedule.prefill == first && schedule.decode.empty(), "FCFS first admission");
  scheduler.finish_prefill(first);
  schedule = scheduler.next();
  expect(schedule.prefill == second && schedule.decode == std::vector<forge::RequestId>{first},
         "new prefill joins existing decode iteration");
  scheduler.finish_prefill(second);
  schedule = scheduler.next();
  expect(schedule.prefill == third &&
             schedule.decode == std::vector<forge::RequestId>({first, second}),
         "waiting requests cannot starve behind decoders");
  scheduler.finish_prefill(third);
  scheduler.cancel(first);
  scheduler.cancel(second);
  scheduler.cancel(third);
  expect(pool.stats().allocated_blocks == 0 && pool.stats().reserved_blocks == 0,
         "continuous batch teardown releases every block");
}

void test_model_file(const char* path) {
  const forge::ModelFile model(path);
  expect(model.config().hidden_size == 8, "model config round trip");
  expect(model.tensors().size() == 2, "model tensor count round trip");
  expect(model.tensor("model.embed_tokens.weight").shape == std::vector<std::uint32_t>({32, 8}),
         "model tensor shape round trip");
  expect(model.tensor_bytes("model.norm.weight").size() == 16, "model tensor extent round trip");
}

}  // namespace

int main(int argc, char** argv) {
  test_sha256();
  test_cache_boundaries();
  test_reservation_isolation();
  test_scheduler_lifecycle();
  test_scheduler_limits_and_reuse();
  test_scheduler_fcfs_continuous_admission();
  test_scheduler_forget_terminal_history();
  if (argc == 2) test_model_file(argv[1]);
  if (failures == 0) std::cout << "all host tests passed\n";
  return failures == 0 ? 0 : 1;
}
