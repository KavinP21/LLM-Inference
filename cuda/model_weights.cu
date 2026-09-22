#include "forge/model_weights.h"

#include "forge/cuda_utils.h"
#include "forge/error.h"
#include "forge/qwen_validation.h"

#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace forge {
namespace {

}  // namespace

std::size_t ModelWeights::required_bytes(const ModelFile& file) {
  std::size_t total = 0;
  std::unordered_set<std::uint64_t> seen_offsets;
  for (const auto& tensor : file.tensors()) {
    if (seen_offsets.insert(tensor.offset).second) {
      total = (total + 255U) / 256U * 256U + tensor.nbytes;
    }
  }
  return total;
}

ModelWeights::ModelWeights(const ModelFile& file, cudaStream_t stream)
    : config_(file.config()), arena_(required_bytes(file)) {
  validate_qwen2_weights(file);
  std::unordered_map<std::uint64_t, void*> device_by_offset;
  for (const auto& info : file.tensors()) {
    check(info.dtype == DType::fp16, "v1 requires FP16 model tensors: " + info.name);
    void* destination{};
    if (const auto alias = device_by_offset.find(info.offset); alias != device_by_offset.end()) {
      destination = alias->second;
    } else {
      destination = arena_.allocate(info.nbytes);
      const auto source = file.tensor_bytes(info.name);
      FORGE_CUDA(cudaMemcpyAsync(destination, source.data(), source.size(), cudaMemcpyHostToDevice, stream));
      device_by_offset.emplace(info.offset, destination);
    }
    tensors_.emplace(info.name, DeviceTensor{destination, info.dtype, info.shape, info.nbytes});
  }
  FORGE_CUDA(cudaStreamSynchronize(stream));
}

const DeviceTensor& ModelWeights::tensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  check(it != tensors_.end(), "missing device tensor: " + name);
  return it->second;
}

}  // namespace forge
