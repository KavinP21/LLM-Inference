#pragma once

#include "forge/error.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <string>

namespace forge {

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

#define FORGE_CUDA(expr) ::forge::cuda_check((expr), #expr)
#define FORGE_CUBLAS(expr) ::forge::cublas_check((expr), #expr)

}  // namespace forge

