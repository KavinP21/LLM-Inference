#include "forge/qwen_validation.h"

#include "forge/error.h"

#include <initializer_list>
#include <string>
#include <vector>

namespace forge {

void validate_qwen2_weights(const ModelFile& file, bool allow_int8) {
  check(allow_int8 || file.quantization().empty(),
        "CUDA INT8 execution is not implemented; use an FP16 artifact");
  const auto& config = file.config();
  check(config.model_type == ModelType::qwen2,
        "CUDA Qwen execution requires a Qwen2 model artifact");
  const auto expect = [&file, allow_int8](const std::string& name,
                              std::initializer_list<std::uint32_t> shape) {
    const auto& tensor = file.tensor(name);
    const bool quantized = allow_int8 && name.starts_with("model.layers.") &&
                           name.ends_with("_proj.weight") && file.quantization().contains(name);
    check(tensor.dtype == DType::fp16 || (quantized && tensor.dtype == DType::int8),
          "Qwen execution requires an FP16 tensor: " + name);
    check(tensor.shape == std::vector<std::uint32_t>(shape),
          "unexpected shape for " + name);
  };
  expect("model.embed_tokens.weight", {config.vocab_size, config.hidden_size});
  expect("model.norm.weight", {config.hidden_size});
  expect("lm_head.weight", {config.vocab_size, config.hidden_size});
  for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    expect(prefix + "input_layernorm.weight", {config.hidden_size});
    expect(prefix + "post_attention_layernorm.weight", {config.hidden_size});
    expect(prefix + "self_attn.q_proj.weight", {config.hidden_size, config.hidden_size});
    expect(prefix + "self_attn.q_proj.bias", {config.hidden_size});
    const auto kv_width = config.num_kv_heads * config.head_dim();
    expect(prefix + "self_attn.k_proj.weight", {kv_width, config.hidden_size});
    expect(prefix + "self_attn.k_proj.bias", {kv_width});
    expect(prefix + "self_attn.v_proj.weight", {kv_width, config.hidden_size});
    expect(prefix + "self_attn.v_proj.bias", {kv_width});
    expect(prefix + "self_attn.o_proj.weight", {config.hidden_size, config.hidden_size});
    expect(prefix + "mlp.gate_proj.weight", {config.intermediate_size, config.hidden_size});
    expect(prefix + "mlp.up_proj.weight", {config.intermediate_size, config.hidden_size});
    expect(prefix + "mlp.down_proj.weight", {config.hidden_size, config.intermediate_size});
  }
}

}  // namespace forge
