#include "forge/model_file.h"

#include "forge/error.h"
#include "forge/sha256.h"

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace forge {
namespace {

constexpr std::array<char, 8> kMagic = {'F', 'O', 'R', 'G', 'E', 'L', 'L', 'M'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kFixedHeaderBytes = 96;

template <typename T>
T read_scalar(std::span<const std::byte> bytes, std::size_t& cursor) {
  static_assert(std::is_trivially_copyable_v<T>);
  check(cursor <= bytes.size() && sizeof(T) <= bytes.size() - cursor, "truncated model header");
  T value{};
  std::memcpy(&value, bytes.data() + cursor, sizeof(T));
  cursor += sizeof(T);
  return value;
}

std::uint64_t element_count(const TensorInfo& tensor) {
  std::uint64_t count = 1;
  for (auto dim : tensor.shape) {
    check(dim > 0, "tensor has zero dimension: " + tensor.name);
    check(count <= std::numeric_limits<std::uint64_t>::max() / dim,
          "tensor element count overflow: " + tensor.name);
    count *= dim;
  }
  return count;
}

std::size_t dtype_bytes(DType dtype) {
  switch (dtype) {
    case DType::fp16: return 2;
    case DType::fp32: return 4;
    case DType::int32: return 4;
  }
  throw Error("unknown tensor dtype");
}

}  // namespace

void ModelConfig::validate() const {
  check(vocab_size > 0 && hidden_size > 0 && intermediate_size > 0 && num_layers > 0,
        "model dimensions must be non-zero");
  check(vocab_size <= 1'000'000U && hidden_size <= 65'536U &&
            intermediate_size <= 262'144U && num_layers <= 1'024U,
        "model dimensions exceed runtime safety limits");
  check(num_attention_heads > 0 && num_kv_heads > 0, "attention head counts must be non-zero");
  check(hidden_size % num_attention_heads == 0, "hidden size must divide attention heads");
  check(num_attention_heads % num_kv_heads == 0, "query heads must divide KV heads");
  check(head_dim() % 2 == 0, "RoPE requires an even head dimension");
  check(max_position_embeddings > 0, "maximum position count must be non-zero");
  check(max_position_embeddings <= 16'777'216U, "position limit exceeds runtime safety limit");
  check(eos_token_id < vocab_size, "EOS token id is outside the vocabulary");
  check(rope_theta > 0.0F && rms_norm_eps > 0.0F, "invalid RoPE or RMSNorm configuration");
}

ModelFile::ModelFile(const std::filesystem::path& path) {
  file_descriptor_ = ::open(path.c_str(), O_RDONLY);
  check(file_descriptor_ >= 0, "cannot open model file: " + path.string());
  struct stat status {};
  if (::fstat(file_descriptor_, &status) != 0 || status.st_size < 0) {
    ::close(file_descriptor_);
    file_descriptor_ = -1;
    throw Error("cannot determine model file size: " + path.string());
  }
  file_size_ = static_cast<std::size_t>(status.st_size);
  if (file_size_ < kFixedHeaderBytes) {
    ::close(file_descriptor_);
    file_descriptor_ = -1;
    throw Error("model file is too small");
  }
  void* mapped = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_descriptor_, 0);
  if (mapped == MAP_FAILED) {
    ::close(file_descriptor_);
    file_descriptor_ = -1;
    throw Error("cannot memory-map model file: " + path.string());
  }
  mapping_ = static_cast<const std::byte*>(mapped);

  try {
  const auto all = std::span<const std::byte>(mapping_, file_size_);
  std::size_t cursor = 0;
  for (char expected : kMagic) check(read_scalar<char>(all, cursor) == expected, "invalid model magic");
  check(read_scalar<std::uint32_t>(all, cursor) == kVersion, "unsupported model format version");
  const auto metadata_bytes = read_scalar<std::uint32_t>(all, cursor);
  data_start_ = read_scalar<std::uint64_t>(all, cursor);
  const auto data_bytes = read_scalar<std::uint64_t>(all, cursor);
  std::array<std::uint8_t, 32> metadata_digest{};
  for (auto& byte : metadata_digest) byte = read_scalar<std::uint8_t>(all, cursor);
  for (auto& byte : data_digest_) byte = read_scalar<std::uint8_t>(all, cursor);
  check(cursor == kFixedHeaderBytes, "internal header size mismatch");
  check(kFixedHeaderBytes + metadata_bytes <= file_size_, "metadata extends beyond file");
  check(data_start_ % 256U == 0U, "tensor data is not 256-byte aligned");
  check(data_start_ <= file_size_ && data_bytes == file_size_ - data_start_,
        "invalid tensor data extent");

  const auto metadata = all.subspan(kFixedHeaderBytes, metadata_bytes);
  check(sha256(metadata) == metadata_digest, "metadata checksum mismatch");
  check(sha256(all.subspan(data_start_, data_bytes)) == data_digest_, "tensor data checksum mismatch");

  cursor = 0;
  config_.vocab_size = read_scalar<std::uint32_t>(metadata, cursor);
  config_.hidden_size = read_scalar<std::uint32_t>(metadata, cursor);
  config_.intermediate_size = read_scalar<std::uint32_t>(metadata, cursor);
  config_.num_layers = read_scalar<std::uint32_t>(metadata, cursor);
  config_.num_attention_heads = read_scalar<std::uint32_t>(metadata, cursor);
  config_.num_kv_heads = read_scalar<std::uint32_t>(metadata, cursor);
  config_.max_position_embeddings = read_scalar<std::uint32_t>(metadata, cursor);
  config_.eos_token_id = read_scalar<std::uint32_t>(metadata, cursor);
  config_.rope_theta = read_scalar<float>(metadata, cursor);
  config_.rms_norm_eps = read_scalar<float>(metadata, cursor);
  config_.validate();
  const auto tensor_count = read_scalar<std::uint32_t>(metadata, cursor);
  check(tensor_count < 10000U, "implausible tensor count");

  tensors_.reserve(tensor_count);
  for (std::uint32_t index = 0; index < tensor_count; ++index) {
    const auto name_bytes = read_scalar<std::uint16_t>(metadata, cursor);
    const auto dtype_raw = read_scalar<std::uint8_t>(metadata, cursor);
    const auto rank = read_scalar<std::uint8_t>(metadata, cursor);
    TensorInfo info;
    info.offset = read_scalar<std::uint64_t>(metadata, cursor);
    info.nbytes = read_scalar<std::uint64_t>(metadata, cursor);
    check(rank > 0 && rank <= 8, "invalid tensor rank");
    info.shape.reserve(rank);
    for (std::uint8_t dim = 0; dim < rank; ++dim) {
      info.shape.push_back(read_scalar<std::uint32_t>(metadata, cursor));
    }
    check(cursor <= metadata.size() && name_bytes <= metadata.size() - cursor, "truncated tensor name");
    info.name.assign(reinterpret_cast<const char*>(metadata.data() + cursor), name_bytes);
    cursor += name_bytes;
    info.dtype = static_cast<DType>(dtype_raw);
    check(info.nbytes == element_count(info) * dtype_bytes(info.dtype),
          "tensor byte count does not match shape: " + info.name);
    check(info.offset % 256U == 0U, "unaligned tensor: " + info.name);
    check(info.offset <= data_bytes && info.nbytes <= data_bytes - info.offset,
          "tensor outside data section: " + info.name);
    check(by_name_.emplace(info.name, tensors_.size()).second, "duplicate tensor: " + info.name);
    tensors_.push_back(std::move(info));
  }
  check(cursor == metadata.size(), "unexpected trailing model metadata");
  std::vector<const TensorInfo*> by_offset;
  by_offset.reserve(tensors_.size());
  for (const auto& tensor : tensors_) by_offset.push_back(&tensor);
  std::sort(by_offset.begin(), by_offset.end(), [](const auto* left, const auto* right) {
    return left->offset < right->offset;
  });
  for (std::size_t index = 1; index < by_offset.size(); ++index) {
    const auto& previous = *by_offset[index - 1];
    const auto& current = *by_offset[index];
    if (current.offset == previous.offset) {
      check(current.nbytes == previous.nbytes && current.dtype == previous.dtype &&
                current.shape == previous.shape,
            "incompatible tensors share a data offset");
    } else {
      check(previous.offset + previous.nbytes <= current.offset,
            "tensor data regions partially overlap");
    }
  }
  } catch (...) {
    ::munmap(const_cast<std::byte*>(mapping_), file_size_);
    ::close(file_descriptor_);
    mapping_ = nullptr;
    file_descriptor_ = -1;
    file_size_ = 0;
    throw;
  }
}

ModelFile::~ModelFile() {
  if (mapping_ != nullptr) ::munmap(const_cast<std::byte*>(mapping_), file_size_);
  if (file_descriptor_ >= 0) ::close(file_descriptor_);
}

const TensorInfo& ModelFile::tensor(const std::string& name) const {
  const auto it = by_name_.find(name);
  check(it != by_name_.end(), "missing tensor: " + name);
  return tensors_[it->second];
}

std::span<const std::byte> ModelFile::tensor_bytes(const std::string& name) const {
  const auto& info = tensor(name);
  return std::span<const std::byte>(mapping_, file_size_).subspan(data_start_ + info.offset,
                                                                  info.nbytes);
}

}  // namespace forge
