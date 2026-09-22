#pragma once

#include "forge/cuda_arena.h"

#include <cstddef>
#include <cstdint>
#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <unordered_map>

namespace forge {

class ExecutionContext {
 public:
  explicit ExecutionContext(std::size_t workspace_bytes = 32U << 20U);
  ~ExecutionContext();
  ExecutionContext(const ExecutionContext&) = delete;
  ExecutionContext& operator=(const ExecutionContext&) = delete;

  // Row-major D[m,n] = A[m,k] * B[n,k]^T, optionally adding FP16 bias[n].
  void linear_fp16(const void* a, const void* b, const void* bias, void* d,
                   std::int64_t m, std::int64_t n, std::int64_t k);
  void synchronize() const;

  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  [[nodiscard]] cublasLtHandle_t blas() const { return blas_; }
  [[nodiscard]] std::size_t linear_plan_count() const { return linear_plans_.size(); }

 private:
  struct LinearKey {
    std::int64_t m{};
    std::int64_t n{};
    std::int64_t k{};
    bool bias{};
    bool operator==(const LinearKey&) const = default;
  };
  struct LinearKeyHash {
    std::size_t operator()(const LinearKey& key) const noexcept;
  };
  struct LinearPlan;

  LinearPlan& linear_plan(const LinearKey& key);

  cudaStream_t stream_{};
  cublasLtHandle_t blas_{};
  CudaArena workspace_;
  std::unordered_map<LinearKey, std::unique_ptr<LinearPlan>, LinearKeyHash> linear_plans_;
};

}  // namespace forge
