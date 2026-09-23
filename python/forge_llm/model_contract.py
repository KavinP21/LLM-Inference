"""Architecture-specific tensor contracts independent of execution backends."""

from __future__ import annotations

import numpy as np

from .model_file import ModelFile


def _expect_fp16(file: ModelFile, name: str, shape: tuple[int, ...]) -> None:
    tensor = file.tensor_info(name)
    if tensor.dtype != np.dtype("<f2"):
        raise ValueError(f"model execution requires FP16 tensor: {name}")
    if tensor.shape != shape:
        raise ValueError(
            f"unexpected shape for {name}: expected {shape}, got {tensor.shape}"
        )


def validate_qwen2_weights(file: ModelFile) -> None:
    config = file.config
    if config.model_type != "qwen2":
        raise ValueError("Qwen2 validation requires a Qwen2 model artifact")
    _expect_fp16(
        file, "model.embed_tokens.weight", (config.vocab_size, config.hidden_size)
    )
    _expect_fp16(file, "model.norm.weight", (config.hidden_size,))
    _expect_fp16(file, "lm_head.weight", (config.vocab_size, config.hidden_size))
    query_width = config.num_attention_heads * config.attention_head_dim
    kv_width = config.num_key_value_heads * config.attention_head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        _expect_fp16(file, prefix + "input_layernorm.weight", (config.hidden_size,))
        _expect_fp16(
            file, prefix + "post_attention_layernorm.weight", (config.hidden_size,)
        )
        _expect_fp16(
            file,
            prefix + "self_attn.q_proj.weight",
            (query_width, config.hidden_size),
        )
        _expect_fp16(file, prefix + "self_attn.q_proj.bias", (query_width,))
        _expect_fp16(
            file,
            prefix + "self_attn.k_proj.weight",
            (kv_width, config.hidden_size),
        )
        _expect_fp16(file, prefix + "self_attn.k_proj.bias", (kv_width,))
        _expect_fp16(
            file,
            prefix + "self_attn.v_proj.weight",
            (kv_width, config.hidden_size),
        )
        _expect_fp16(file, prefix + "self_attn.v_proj.bias", (kv_width,))
        _expect_fp16(
            file,
            prefix + "self_attn.o_proj.weight",
            (config.hidden_size, query_width),
        )
        _expect_fp16(
            file,
            prefix + "mlp.gate_proj.weight",
            (config.intermediate_size, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "mlp.up_proj.weight",
            (config.intermediate_size, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "mlp.down_proj.weight",
            (config.hidden_size, config.intermediate_size),
        )


def validate_gemma3_weights(file: ModelFile) -> None:
    config = file.config
    if config.model_type != "gemma3_text":
        raise ValueError("Gemma 3 validation requires a Gemma 3 text artifact")
    _expect_fp16(
        file, "model.embed_tokens.weight", (config.vocab_size, config.hidden_size)
    )
    _expect_fp16(file, "model.norm.weight", (config.hidden_size,))
    _expect_fp16(file, "lm_head.weight", (config.vocab_size, config.hidden_size))
    query_width = config.num_attention_heads * config.attention_head_dim
    kv_width = config.num_key_value_heads * config.attention_head_dim
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        for norm in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ):
            _expect_fp16(file, prefix + norm, (config.hidden_size,))
        _expect_fp16(
            file,
            prefix + "self_attn.q_proj.weight",
            (query_width, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "self_attn.k_proj.weight",
            (kv_width, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "self_attn.v_proj.weight",
            (kv_width, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "self_attn.o_proj.weight",
            (config.hidden_size, query_width),
        )
        _expect_fp16(
            file, prefix + "self_attn.q_norm.weight", (config.attention_head_dim,)
        )
        _expect_fp16(
            file, prefix + "self_attn.k_norm.weight", (config.attention_head_dim,)
        )
        _expect_fp16(
            file,
            prefix + "mlp.gate_proj.weight",
            (config.intermediate_size, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "mlp.up_proj.weight",
            (config.intermediate_size, config.hidden_size),
        )
        _expect_fp16(
            file,
            prefix + "mlp.down_proj.weight",
            (config.hidden_size, config.intermediate_size),
        )


def validate_model_weights(file: ModelFile) -> None:
    validators = {
        "qwen2": validate_qwen2_weights,
        "gemma3_text": validate_gemma3_weights,
    }
    try:
        validator = validators[file.config.model_type]
    except KeyError as exc:
        raise ValueError(
            f"no tensor contract for model type {file.config.model_type!r}"
        ) from exc
    validator(file)
