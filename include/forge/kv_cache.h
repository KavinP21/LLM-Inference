#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace forge {

using RequestId = std::uint64_t;

struct KVCacheStats {
  std::uint32_t total_blocks{};
  std::uint32_t allocated_blocks{};
  std::uint32_t reserved_blocks{};
  std::uint32_t free_blocks{};
  std::uint64_t live_tokens{};
  std::uint64_t internal_fragmentation_tokens{};
  std::uint64_t allocation_failures{};
  std::uint64_t bytes_per_block{};

  [[nodiscard]] double occupancy() const;
};

class KVBlockPool {
 public:
  KVBlockPool(std::uint64_t capacity_bytes, std::uint32_t num_layers,
              std::uint32_t num_kv_heads, std::uint32_t head_dim,
              std::uint32_t block_tokens = 16, std::uint32_t element_bytes = 2);

  [[nodiscard]] bool reserve(RequestId request_id, std::uint32_t maximum_tokens);
  [[nodiscard]] bool ensure_tokens(RequestId request_id, std::uint32_t live_tokens);
  void release(RequestId request_id);

  [[nodiscard]] const std::vector<std::uint32_t>& block_table(RequestId request_id) const;
  [[nodiscard]] bool contains(RequestId request_id) const;
  [[nodiscard]] KVCacheStats stats() const;
  [[nodiscard]] std::uint32_t block_tokens() const { return block_tokens_; }
  [[nodiscard]] std::uint64_t bytes_per_block() const { return bytes_per_block_; }

 private:
  struct Allocation {
    std::uint32_t reservation{};
    std::uint32_t live_tokens{};
    std::vector<std::uint32_t> blocks;
  };

  std::uint32_t block_tokens_{};
  std::uint64_t bytes_per_block_{};
  std::vector<std::uint32_t> free_blocks_;
  std::unordered_map<RequestId, Allocation> allocations_;
  std::uint32_t reserved_unallocated_{};
  std::uint64_t allocation_failures_{};
};

}  // namespace forge

