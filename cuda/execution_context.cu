#include "forge/execution_context.h"

#include "forge/cuda_utils.h"
#include "forge/error.h"

#include <functional>
#include <utility>

namespace forge {

struct ExecutionContext::LinearPlan {
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a_layout{};
  cublasLtMatrixLayout_t b_layout{};
  cublasLtMatrixLayout_t d_layout{};
  cublasLtMatmulAlgo_t algorithm{};

  ~LinearPlan() {
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
};

std::size_t ExecutionContext::LinearKeyHash::operator()(const LinearKey& key) const noexcept {
  auto hash = std::hash<std::int64_t>{}(key.m);
  hash ^= std::hash<std::int64_t>{}(key.n) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
  hash ^= std::hash<std::int64_t>{}(key.k) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
  hash ^= std::hash<bool>{}(key.bias) + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
  return hash;
}

ExecutionContext::ExecutionContext(std::size_t workspace_bytes) : workspace_(workspace_bytes) {
  FORGE_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  try {
    FORGE_CUBLAS(cublasLtCreate(&blas_));
  } catch (...) {
    cudaStreamDestroy(stream_);
    throw;
  }
}

ExecutionContext::~ExecutionContext() {
  // Plans own descriptors created from the handle and must die before the handle.
  linear_plans_.clear();
  if (blas_ != nullptr) cublasLtDestroy(blas_);
  if (stream_ != nullptr) cudaStreamDestroy(stream_);
}

ExecutionContext::LinearPlan& ExecutionContext::linear_plan(const LinearKey& key) {
  if (const auto found = linear_plans_.find(key); found != linear_plans_.end()) return *found->second;

  auto plan = std::make_unique<LinearPlan>();
  cublasLtMatmulPreference_t preference{};
  try {
    FORGE_CUBLAS(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t trans_b = CUBLAS_OP_T;
    FORGE_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    if (key.bias) {
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      FORGE_CUBLAS(cublasLtMatmulDescSetAttribute(
          plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    }
    FORGE_CUBLAS(cublasLtMatrixLayoutCreate(&plan->a_layout, CUDA_R_16F, key.m, key.k, key.k));
    FORGE_CUBLAS(cublasLtMatrixLayoutCreate(&plan->b_layout, CUDA_R_16F, key.n, key.k, key.k));
    FORGE_CUBLAS(cublasLtMatrixLayoutCreate(&plan->d_layout, CUDA_R_16F, key.m, key.n, key.n));
    const cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    for (auto layout : {plan->a_layout, plan->b_layout, plan->d_layout}) {
      FORGE_CUBLAS(cublasLtMatrixLayoutSetAttribute(
          layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major, sizeof(row_major)));
    }
    FORGE_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
    const auto workspace_bytes = workspace_.capacity();
    FORGE_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes, sizeof(workspace_bytes)));
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    FORGE_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
        blas_, plan->operation, plan->a_layout, plan->b_layout, plan->d_layout,
        plan->d_layout, preference, 1, &heuristic, &returned));
    check(returned == 1, "cuBLASLt found no algorithm for linear layer");
    plan->algorithm = heuristic.algo;
  } catch (...) {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    throw;
  }
  cublasLtMatmulPreferenceDestroy(preference);
  auto [inserted, _] = linear_plans_.emplace(key, std::move(plan));
  return *inserted->second;
}

void ExecutionContext::linear_fp16(const void* a, const void* b, const void* bias, void* d,
                                   std::int64_t m, std::int64_t n, std::int64_t k) {
  check(a != nullptr && b != nullptr && d != nullptr, "linear layer received a null tensor");
  check(m > 0 && n > 0 && k > 0, "linear layer dimensions must be positive");
  auto& plan = linear_plan({m, n, k, bias != nullptr});
  if (bias != nullptr) {
    FORGE_CUBLAS(cublasLtMatmulDescSetAttribute(
        plan.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  FORGE_CUBLAS(cublasLtMatmul(
      blas_, plan.operation, &alpha, a, plan.a_layout, b, plan.b_layout, &beta,
      d, plan.d_layout, d, plan.d_layout, &plan.algorithm,
      workspace_.data(), workspace_.capacity(), stream_));
}

void ExecutionContext::synchronize() const { FORGE_CUDA(cudaStreamSynchronize(stream_)); }

}  // namespace forge
