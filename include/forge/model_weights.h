#pragma once

#include "forge/cuda_arena.h"
#include "forge/model_file.h"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace forge {

struct DeviceTensor {
  const void* data{};
  DType dtype{};
  std::vector<std::uint32_t> shape;
  std::uint64_t nbytes{};
};

class ModelWeights {
 public:
  ModelWeights(const ModelFile& file, cudaStream_t stream);

  [[nodiscard]] const DeviceTensor& tensor(const std::string& name) const;
  [[nodiscard]] const ModelConfig& config() const { return config_; }
  [[nodiscard]] std::size_t device_bytes() const { return arena_.used(); }

 private:
  static std::size_t required_bytes(const ModelFile& file);
  ModelConfig config_{};
  CudaArena arena_;
  std::unordered_map<std::string, DeviceTensor> tensors_;
};

}  // namespace forge
