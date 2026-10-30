"""Qwen2 and Gemma 3 text models from ``.engine`` files (FP16)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ..model_contract import validate_gemma3_weights, validate_qwen2_weights
from ..model_file import ModelFile
from ..paged_kv import KVGroup
from .base import Weights, mlx


class DenseModel:
    def __init__(self, path: str | Path) -> None:
        self.mx = mx = mlx()
        self.file = ModelFile(path)
        try:
            config = self.config = self.file.config
            if self.file.quantization:
                raise ValueError(
                    "INT8 .engine artifacts aren't supported by the MLX backend"
                )
            if config.model_type == "qwen2":
                validate_qwen2_weights(self.file)
            elif config.model_type == "gemma3_text":
                validate_gemma3_weights(self.file)
                if config.activation != "gelu_pytorch_tanh":
                    raise ValueError("Gemma 3 needs the gelu_pytorch_tanh activation")
            else:
                raise ValueError(f"no MLX adapter for model type {config.model_type!r}")
            self.model_type = config.model_type
            self.gemma = config.model_type == "gemma3_text"
            self.dtype = mx.float16
            self.vocab_size = config.vocab_size
            self.max_position_embeddings = config.max_position_embeddings
            self.eos_token_ids = (config.eos_token_id,)
            self.head_dim = config.attention_head_dim
            # Tensors that alias the same bytes (the tied output head) share one array.
            arrays: dict[tuple, Any] = {}
            tensors = {}
            for name, info in self.file.tensors.items():
                key = (info.offset, info.shape)
                if key not in arrays:
                    arrays[key] = mx.array(self.file.tensor_numpy(name))
                tensors[name] = arrays[key]
            if self.gemma:
                # Gemma scales by (offset + weight); fold the offset in once.
                for name in [n for n in tensors if n.endswith("norm.weight")]:
                    tensors[name] = (
                        tensors[name].astype(mx.float32) + config.norm_weight_offset
                    ).astype(mx.float16)
            mx.eval(*tensors.values())
            self.weights = Weights(mx, tensors, {})
            layers = range(config.num_hidden_layers)
            sliding = tuple(i for i in layers if config.is_sliding_layer(i))
            full = tuple(i for i in layers if not config.is_sliding_layer(i))
            heads = config.num_key_value_heads
            self.kv_groups = [
                group
                for group in (
                    KVGroup(
                        "sliding", sliding, heads, self.head_dim, config.sliding_window
                    ),
                    KVGroup("full", full, heads, self.head_dim),
                )
                if group.layers
            ]
        except Exception:
            self.file.close()
            raise

    def _norm(self, x: Any, name: str) -> Any:
        return self.mx.fast.rms_norm(x, self.weights[name], self.config.rms_norm_eps)

    def forward(self, batch: Any) -> Any:
        mx, w, config = self.mx, self.weights, self.config
        x = w.embed("model.embed_tokens", batch.tokens)
        if self.gemma:
            x = x * mx.array(config.embedding_scale, dtype=x.dtype)
        n, d = batch.count, self.head_dim
        scale = (config.query_pre_attn_scalar if self.gemma else d) ** -0.5
        for layer in range(config.num_hidden_layers):
            p = f"model.layers.{layer}."
            h = self._norm(x, p + "input_layernorm.weight")
            q = w.linear(p + "self_attn.q_proj", h).reshape(n, -1, d)
            k = w.linear(p + "self_attn.k_proj", h).reshape(n, -1, d)
            v = w.linear(p + "self_attn.v_proj", h).reshape(n, -1, d)
            base = config.rope_theta
            if self.gemma:
                q = self._norm(q, p + "self_attn.q_norm.weight")
                k = self._norm(k, p + "self_attn.k_norm.weight")
                if config.is_sliding_layer(layer):
                    base = config.rope_local_theta
            q = batch.rope(q, d, base=base)
            k = batch.rope(k, d, base=base)
            attention = batch.attend(
                layer, q, k, v, scale=scale, softcap=config.attn_logit_softcapping
            )
            out = w.linear(p + "self_attn.o_proj", attention)
            if self.gemma:
                x = x + self._norm(out, p + "post_attention_layernorm.weight")
                h = self._norm(x, p + "pre_feedforward_layernorm.weight")
                gate = w.linear(p + "mlp.gate_proj", h)
                gate = (
                    0.5
                    * gate
                    * (
                        1
                        + mx.tanh(math.sqrt(2 / math.pi) * (gate + 0.044715 * gate**3))
                    )
                )
                mlp = w.linear(
                    p + "mlp.down_proj", gate * w.linear(p + "mlp.up_proj", h)
                )
                x = x + self._norm(mlp, p + "post_feedforward_layernorm.weight")
            else:
                x = x + out
                h = self._norm(x, p + "post_attention_layernorm.weight")
                gate = w.linear(p + "mlp.gate_proj", h)
                mlp = w.linear(
                    p + "mlp.down_proj",
                    gate * mx.sigmoid(gate) * w.linear(p + "mlp.up_proj", h),
                )
                x = x + mlp
        if batch.prefill:
            x = x[-1:]
        logits = w.linear("lm_head", self._norm(x, "model.norm.weight")).astype(
            mx.float32
        )
        if config.final_logit_softcapping:
            softcap = config.final_logit_softcapping
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def close(self) -> None:
        self.weights.clear()
        self.file.close()
