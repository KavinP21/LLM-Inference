#include "forge/model_validation.h"

#include "forge/error.h"
#include "forge/qwen_validation.h"

#include <initializer_list>
#include <string>
#include <vector>
#include <unordered_set>

namespace forge {

void validate_gemma3_weights(const ModelFile& file) {
  const auto& config = file.config();
  check(config.model_type == ModelType::gemma3_text,
        "Gemma 3 validation requires a Gemma 3 text artifact");
  const auto expect = [&file](const std::string& name,
                              std::initializer_list<std::uint32_t> shape) {
    const auto& tensor = file.tensor(name);
    const bool quantized = name.starts_with("model.layers.") && name.ends_with("_proj.weight") &&
                           file.quantization().contains(name);
    check(tensor.dtype == DType::fp16 || (quantized && tensor.dtype == DType::int8),
          "Gemma 3 execution requires an FP16 tensor: " + name);
    check(tensor.shape == std::vector<std::uint32_t>(shape),
          "unexpected shape for " + name);
  };

  expect("model.embed_tokens.weight", {config.vocab_size, config.hidden_size});
  expect("model.norm.weight", {config.hidden_size});
  expect("lm_head.weight", {config.vocab_size, config.hidden_size});
  const auto query_width = config.num_attention_heads * config.head_dim();
  const auto kv_width = config.num_kv_heads * config.head_dim();
  for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    expect(prefix + "input_layernorm.weight", {config.hidden_size});
    expect(prefix + "post_attention_layernorm.weight", {config.hidden_size});
    expect(prefix + "pre_feedforward_layernorm.weight", {config.hidden_size});
    expect(prefix + "post_feedforward_layernorm.weight", {config.hidden_size});
    expect(prefix + "self_attn.q_proj.weight", {query_width, config.hidden_size});
    expect(prefix + "self_attn.k_proj.weight", {kv_width, config.hidden_size});
    expect(prefix + "self_attn.v_proj.weight", {kv_width, config.hidden_size});
    expect(prefix + "self_attn.o_proj.weight", {config.hidden_size, query_width});
    expect(prefix + "self_attn.q_norm.weight", {config.head_dim()});
    expect(prefix + "self_attn.k_norm.weight", {config.head_dim()});
    expect(prefix + "mlp.gate_proj.weight", {config.intermediate_size, config.hidden_size});
    expect(prefix + "mlp.up_proj.weight", {config.intermediate_size, config.hidden_size});
    expect(prefix + "mlp.down_proj.weight", {config.hidden_size, config.intermediate_size});
  }
}

void validate_model_weights(const ModelFile& file) {
  std::unordered_set<std::string> supported;
  for (std::uint32_t layer = 0; layer < file.config().num_layers; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    for (const auto* name : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) {
      supported.insert(prefix + name + ".weight");
    }
  }
  for (const auto& [name, spec] : file.quantization()) {
    check(supported.contains(name),
          "only internal projection weights may be quantized");
  }
  switch (file.config().model_type) {
    case ModelType::qwen2: validate_qwen2_weights(file, true); return;
    case ModelType::gemma3_text: validate_gemma3_weights(file); return;
  }
  throw Error("no tensor contract for model type");
}

}  // namespace forge
