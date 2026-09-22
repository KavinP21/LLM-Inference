#pragma once

#include <cstddef>
#include <cuda_runtime_api.h>

namespace forge {

class CudaArena {
 public:
  explicit CudaArena(std::size_t capacity);
  ~CudaArena();
  CudaArena(const CudaArena&) = delete;
  CudaArena& operator=(const CudaArena&) = delete;
  CudaArena(CudaArena&& other) noexcept;
  CudaArena& operator=(CudaArena&& other) noexcept;

  void* allocate(std::size_t bytes, std::size_t alignment = 256);
  void reset() { used_ = 0; }
  [[nodiscard]] void* data() const { return data_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }
  [[nodiscard]] std::size_t used() const { return used_; }

 private:
  void* data_{};
  std::size_t capacity_{};
  std::size_t used_{};
};

}  // namespace forge

