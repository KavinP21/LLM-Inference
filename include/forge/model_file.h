#pragma once

#include "forge/model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace forge {

enum class DType : std::uint8_t { fp16 = 1, fp32 = 2, int32 = 3, int8 = 4 };

struct QuantizationSpec {
  std::string scale_name;
  std::uint8_t scheme{};
  std::uint8_t axis{};
};

struct TensorInfo {
  std::string name;
  DType dtype{};
  std::vector<std::uint32_t> shape;
  std::uint64_t offset{};
  std::uint64_t nbytes{};
};

class ModelFile {
 public:
  explicit ModelFile(const std::filesystem::path& path);
  ~ModelFile();
  ModelFile(const ModelFile&) = delete;
  ModelFile& operator=(const ModelFile&) = delete;

  [[nodiscard]] const ModelConfig& config() const { return config_; }
  [[nodiscard]] std::uint32_t format_version() const { return format_version_; }
  [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return tensors_; }
  [[nodiscard]] const std::unordered_map<std::string, QuantizationSpec>& quantization() const {
    return quantization_;
  }
  [[nodiscard]] const TensorInfo& tensor(const std::string& name) const;
  [[nodiscard]] std::span<const std::byte> tensor_bytes(const std::string& name) const;
  [[nodiscard]] const std::array<std::uint8_t, 32>& data_digest() const { return data_digest_; }

 private:
  std::uint32_t format_version_{};
  ModelConfig config_{};
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, std::size_t> by_name_;
  std::unordered_map<std::string, QuantizationSpec> quantization_;
  int file_descriptor_{-1};
  const std::byte* mapping_{};
  std::size_t file_size_{};
  std::uint64_t data_start_{};
  std::array<std::uint8_t, 32> data_digest_{};
};

}  // namespace forge
