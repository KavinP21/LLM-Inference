#include "forge/model_file.h"
#include "forge/qwen_validation.h"

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: forge-inspect-model MODEL.engine\n";
    return 2;
  }
  try {
    const forge::ModelFile model(argv[1]);
    forge::validate_qwen2_weights(model);
    const auto& config = model.config();
    std::cout << "format: Forge LLM v1\n"
              << "vocab_size: " << config.vocab_size << '\n'
              << "hidden_size: " << config.hidden_size << '\n'
              << "intermediate_size: " << config.intermediate_size << '\n'
              << "layers: " << config.num_layers << '\n'
              << "attention_heads: " << config.num_attention_heads << '\n'
              << "kv_heads: " << config.num_kv_heads << '\n'
              << "head_dim: " << config.head_dim() << '\n'
              << "max_positions: " << config.max_position_embeddings << '\n'
              << "tensors: " << model.tensors().size() << '\n'
              << "data_sha256: ";
    for (const auto byte : model.data_digest()) {
      std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    }
    std::cout << std::dec << '\n';
  } catch (const std::exception& error) {
    std::cerr << "invalid model: " << error.what() << '\n';
    return 1;
  }
  return 0;
}

