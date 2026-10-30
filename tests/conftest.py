"""Shared test helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from instinct.format import ModelConfig, write_engine


def tiny_artifact(path: Path, family: str = "qwen2") -> None:
    config = ModelConfig(32, 8, 16, 2, 2, 1, 64, 2, 10_000.0, 1e-6)
    if family == "gemma3_text":
        from dataclasses import replace

        config = replace(
            config,
            model_type=family,
            head_dim=4,
            sliding_window=4,
            sliding_window_pattern=2,
            activation="gelu_pytorch_tanh",
            embedding_scale=8**0.5,
            norm_weight_offset=1.0,
        )
    random = np.random.default_rng(17)
    tensors = {
        "model.embed_tokens.weight": random.normal(0, 0.12, (32, 8)).astype(np.float16),
        "model.norm.weight": np.ones(8, dtype=np.float16),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        norms = ["input_layernorm", "post_attention_layernorm"]
        if family == "gemma3_text":
            norms += ["pre_feedforward_layernorm", "post_feedforward_layernorm"]
            for name in ["q_norm", "k_norm"]:
                tensors[prefix + f"self_attn.{name}.weight"] = np.zeros(
                    4, dtype=np.float16
                )
        for norm in norms:
            tensors[prefix + norm + ".weight"] = np.full(
                8, 0 if family == "gemma3_text" else 1, dtype=np.float16
            )
        for group, name, shape in [
            ("self_attn", "q", (8, 8)),
            ("self_attn", "k", (4, 8)),
            ("self_attn", "v", (4, 8)),
            ("self_attn", "o", (8, 8)),
            ("mlp", "gate", (16, 8)),
            ("mlp", "up", (16, 8)),
            ("mlp", "down", (8, 16)),
        ]:
            tensors[prefix + f"{group}.{name}_proj.weight"] = random.normal(
                0, 0.12, shape
            ).astype(np.float16)
            if family == "qwen2" and name in {"q", "k", "v"}:
                tensors[prefix + f"{group}.{name}_proj.bias"] = np.zeros(
                    shape[0], dtype=np.float16
                )
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    write_engine(path, config, tensors, {"lm_head.weight": "model.embed_tokens.weight"})
