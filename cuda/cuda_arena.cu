#include "forge/cuda_arena.h"

#include "forge/cuda_utils.h"
#include "forge/error.h"

#include <cstdint>
#include <utility>

namespace forge {

CudaArena::CudaArena(std::size_t capacity) : capacity_(capacity) {
  check(capacity > 0, "CUDA arena capacity must be non-zero");
  FORGE_CUDA(cudaMalloc(&data_, capacity));
}

CudaArena::~CudaArena() {
  if (data_ != nullptr) cudaFree(data_);
}

CudaArena::CudaArena(CudaArena&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      capacity_(std::exchange(other.capacity_, 0)), used_(std::exchange(other.used_, 0)) {}

CudaArena& CudaArena::operator=(CudaArena&& other) noexcept {
  if (this == &other) return *this;
  if (data_ != nullptr) cudaFree(data_);
  data_ = std::exchange(other.data_, nullptr);
  capacity_ = std::exchange(other.capacity_, 0);
  used_ = std::exchange(other.used_, 0);
  return *this;
}

void* CudaArena::allocate(std::size_t bytes, std::size_t alignment) {
  check(alignment > 0 && (alignment & (alignment - 1)) == 0, "arena alignment must be a power of two");
  const auto aligned = (used_ + alignment - 1) & ~(alignment - 1);
  check(aligned <= capacity_ && bytes <= capacity_ - aligned, "CUDA arena exhausted");
  auto* result = static_cast<std::byte*>(data_) + aligned;
  used_ = aligned + bytes;
  return result;
}

}  // namespace forge

