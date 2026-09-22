#include "forge/kv_cache.h"

#include "forge/error.h"

#include <algorithm>
#include <limits>

namespace forge {
namespace {

std::uint32_t blocks_for(std::uint32_t tokens, std::uint32_t block_tokens) {
  return tokens == 0 ? 0 : 1U + (tokens - 1U) / block_tokens;
}

}  // namespace

double KVCacheStats::occupancy() const {
  return total_blocks == 0 ? 0.0 : static_cast<double>(allocated_blocks) / total_blocks;
}

KVBlockPool::KVBlockPool(std::uint64_t capacity_bytes, std::uint32_t num_layers,
                         std::uint32_t num_kv_heads, std::uint32_t head_dim,
                         std::uint32_t block_tokens, std::uint32_t element_bytes)
    : block_tokens_(block_tokens) {
  check(num_layers > 0 && num_kv_heads > 0 && head_dim > 0, "invalid KV dimensions");
  check(block_tokens > 0 && element_bytes > 0, "invalid KV block layout");
  bytes_per_block_ = static_cast<std::uint64_t>(num_layers) * 2U * num_kv_heads *
                     block_tokens * head_dim * element_bytes;
  check(bytes_per_block_ > 0, "KV block size overflow");
  const auto block_count64 = capacity_bytes / bytes_per_block_;
  check(block_count64 > 0, "KV cache is too small for one block");
  check(block_count64 <= std::numeric_limits<std::uint32_t>::max(), "too many KV blocks");
  const auto block_count = static_cast<std::uint32_t>(block_count64);
  free_blocks_.reserve(block_count);
  for (std::uint32_t i = block_count; i > 0; --i) free_blocks_.push_back(i - 1U);
}

bool KVBlockPool::reserve(RequestId request_id, std::uint32_t maximum_tokens) {
  check(maximum_tokens > 0, "cannot reserve an empty sequence");
  check(!contains(request_id), "duplicate KV reservation");
  const auto needed = blocks_for(maximum_tokens, block_tokens_);
  const auto available = static_cast<std::uint32_t>(free_blocks_.size()) - reserved_unallocated_;
  if (needed > available) {
    ++allocation_failures_;
    return false;
  }
  allocations_.emplace(request_id, Allocation{needed, 0, {}});
  reserved_unallocated_ += needed;
  return true;
}

bool KVBlockPool::ensure_tokens(RequestId request_id, std::uint32_t live_tokens) {
  auto it = allocations_.find(request_id);
  check(it != allocations_.end(), "request has no KV reservation");
  auto& allocation = it->second;
  const auto needed = blocks_for(live_tokens, block_tokens_);
  check(needed <= allocation.reservation, "request exceeded its KV reservation");
  check(live_tokens >= allocation.live_tokens, "KV token count cannot shrink");
  if (needed > allocation.blocks.size()) {
    const auto additional = needed - static_cast<std::uint32_t>(allocation.blocks.size());
    if (additional > free_blocks_.size()) {
      ++allocation_failures_;
      return false;
    }
    for (std::uint32_t i = 0; i < additional; ++i) {
      allocation.blocks.push_back(free_blocks_.back());
      free_blocks_.pop_back();
    }
    reserved_unallocated_ -= additional;
  }
  allocation.live_tokens = live_tokens;
  return true;
}

void KVBlockPool::release(RequestId request_id) {
  auto it = allocations_.find(request_id);
  if (it == allocations_.end()) return;
  auto& allocation = it->second;
  free_blocks_.insert(free_blocks_.end(), allocation.blocks.begin(), allocation.blocks.end());
  reserved_unallocated_ -= allocation.reservation - static_cast<std::uint32_t>(allocation.blocks.size());
  allocations_.erase(it);
}

const std::vector<std::uint32_t>& KVBlockPool::block_table(RequestId request_id) const {
  const auto it = allocations_.find(request_id);
  check(it != allocations_.end(), "request has no KV allocation");
  return it->second.blocks;
}

bool KVBlockPool::contains(RequestId request_id) const {
  return allocations_.contains(request_id);
}

KVCacheStats KVBlockPool::stats() const {
  KVCacheStats result;
  result.total_blocks = static_cast<std::uint32_t>(free_blocks_.size());
  for (const auto& [_, allocation] : allocations_) {
    result.total_blocks += static_cast<std::uint32_t>(allocation.blocks.size());
    result.allocated_blocks += static_cast<std::uint32_t>(allocation.blocks.size());
    result.live_tokens += allocation.live_tokens;
    if (!allocation.blocks.empty()) {
      result.internal_fragmentation_tokens += allocation.blocks.size() * block_tokens_ - allocation.live_tokens;
    }
  }
  result.reserved_blocks = reserved_unallocated_;
  result.free_blocks = static_cast<std::uint32_t>(free_blocks_.size());
  result.allocation_failures = allocation_failures_;
  result.bytes_per_block = bytes_per_block_;
  return result;
}

}  // namespace forge
