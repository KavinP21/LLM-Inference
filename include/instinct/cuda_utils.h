#pragma once

#include "instinct/error.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <string>

namespace instinct {

inline void cuda_check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw Error(std::string(expression) + ": " + cudaGetErrorString(status));
  }
}

inline void cublas_check(cublasStatus_t status, const char* expression) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw Error(std::string(expression) + ": cuBLAS status " + std::to_string(status));
  }
}

#define INSTINCT_CUDA(expr) ::instinct::cuda_check((expr), #expr)
#define INSTINCT_CUBLAS(expr) ::instinct::cublas_check((expr), #expr)

}  // namespace instinct

