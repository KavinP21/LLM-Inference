"""MLX execution backend for Qwen2-family decoder-only models.

The initial backend intentionally uses ordinary MLX operations.  It establishes
the numerical and ownership contracts that later custom Metal kernels must
preserve.  PyTorch and Transformers are not imported anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..model_file import ModelFile

if TYPE_CHECKING:
    import mlx.core as mx_types


def _mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "the MLX backend requires Apple Silicon and the 'mlx' extra: "
            "pip install 'forge-llm[mlx]'"
        ) from exc
    return mx


def mlx_build_info() -> dict[str, object]:
    mx = _mlx()
    from importlib.metadata import version

    metal = mx.device_info()
    return {
        "backend": "mlx",
        "mlx_version": version("mlx"),
        "device": metal.get("device_name", "Apple Metal GPU"),
        "architecture": metal.get("architecture", "unknown"),
    }


@dataclass
class LayerKV:
    key: "mx_types.array"
    value: "mx_types.array"


@dataclass
class QwenKVCache:
    layers: list[LayerKV]
    token_count: int


class MlxQwenModel:
    """Correctness-first Qwen2 implementation backed exclusively by MLX."""

    def __init__(self, model_path: str, *, max_model_length: int | None = None) -> None:
        self.mx = _mlx()
        self.file = ModelFile(model_path)
        self.config = self.file.config
        self.max_model_length = max_model_length or self.config.max_position_embeddings
        try:
            if not 0 < self.max_model_length <= self.config.max_position_embeddings:
                raise ValueError("runtime context limit exceeds the model context limit")
            self._validate_qwen_weights()
            self.weights = self._load_weights()
            self.rope_frequencies = 1.0 / (
                self.config.rope_theta
                ** (self.mx.arange(0, self.head_dim, 2, dtype=self.mx.float32) / self.head_dim)
            )
            self.mx.eval(self.rope_frequencies)
        except Exception:
            self.file.close()
            raise

    @property
    def head_dim(self) -> int:
        return self.config.hidden_size // self.config.num_attention_heads

    def _validate_qwen_weights(self) -> None:
        config = self.config

        def expect(name: str, shape: tuple[int, ...]) -> None:
            tensor = self.file.tensor_info(name)
            if tensor.dtype != np.dtype("<f2"):
                raise ValueError(f"MLX v1 requires FP16 tensor: {name}")
            if tensor.shape != shape:
                raise ValueError(
                    f"unexpected shape for {name}: expected {shape}, got {tensor.shape}"
                )

        expect("model.embed_tokens.weight", (config.vocab_size, config.hidden_size))
        expect("model.norm.weight", (config.hidden_size,))
        expect("lm_head.weight", (config.vocab_size, config.hidden_size))
        kv_width = config.num_key_value_heads * self.head_dim
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            expect(prefix + "input_layernorm.weight", (config.hidden_size,))
            expect(prefix + "post_attention_layernorm.weight", (config.hidden_size,))
            expect(prefix + "self_attn.q_proj.weight", (config.hidden_size, config.hidden_size))
            expect(prefix + "self_attn.q_proj.bias", (config.hidden_size,))
            expect(prefix + "self_attn.k_proj.weight", (kv_width, config.hidden_size))
            expect(prefix + "self_attn.k_proj.bias", (kv_width,))
            expect(prefix + "self_attn.v_proj.weight", (kv_width, config.hidden_size))
            expect(prefix + "self_attn.v_proj.bias", (kv_width,))
            expect(prefix + "self_attn.o_proj.weight", (config.hidden_size, config.hidden_size))
            expect(prefix + "mlp.gate_proj.weight", (config.intermediate_size, config.hidden_size))
            expect(prefix + "mlp.up_proj.weight", (config.intermediate_size, config.hidden_size))
            expect(prefix + "mlp.down_proj.weight", (config.hidden_size, config.intermediate_size))

    def _load_weights(self) -> dict[str, "mx_types.array"]:
        mx = self.mx
        result: dict[str, "mx_types.array"] = {}
        aliases: dict[tuple[int, tuple[int, ...], str], "mx_types.array"] = {}
        for name, info in self.file.tensors.items():
            key = (info.offset, info.shape, info.dtype.str)
            if key not in aliases:
                aliases[key] = mx.array(self.file.tensor_numpy(name))
            result[name] = aliases[key]
        # Materialize one device copy now. Generation must not lazily copy mapped
        # host weights during its first timed token.
        mx.eval(*aliases.values())
        return result

    def _linear(self, x, weight_name: str, bias_name: str | None = None):
        output = x @ self.weights[weight_name].T
        if bias_name is not None:
            output = output + self.weights[bias_name]
        return output

    def _rms_norm(self, x, weight_name: str):
        mx = self.mx
        variance = mx.mean(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True)
        normalized = x * mx.rsqrt(variance + self.config.rms_norm_eps).astype(x.dtype)
        return normalized * self.weights[weight_name]

    def _rope(self, tensor, positions):
        mx = self.mx
        half = self.head_dim // 2
        angles = positions.astype(mx.float32)[:, None] * self.rope_frequencies[None, :]
        cosine = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)[:, None, :]
        sine = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)[:, None, :]
        rotated = mx.concatenate([-tensor[..., half:], tensor[..., :half]], axis=-1)
        return tensor * cosine.astype(tensor.dtype) + rotated * sine.astype(tensor.dtype)

    def _attention(self, query, key, value, query_start: int):
        mx = self.mx
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        key = mx.repeat(key, groups, axis=1)
        value = mx.repeat(value, groups, axis=1)
        # [tokens, heads, dim] -> [heads, tokens, dim]
        q = mx.transpose(query, (1, 0, 2)).astype(mx.float32)
        k = mx.transpose(key, (1, 0, 2)).astype(mx.float32)
        v = mx.transpose(value, (1, 0, 2)).astype(mx.float32)
        scores = (q @ mx.transpose(k, (0, 2, 1))) * (self.head_dim ** -0.5)
        query_positions = mx.arange(query_start, query_start + query.shape[0])[:, None]
        key_positions = mx.arange(key.shape[0])[None, :]
        causal = key_positions <= query_positions
        scores = mx.where(causal[None, :, :], scores, mx.array(-1e9, dtype=mx.float32))
        probabilities = mx.softmax(scores, axis=-1)
        output = probabilities @ v
        return mx.transpose(output, (1, 0, 2)).reshape(query.shape[0], -1).astype(query.dtype)

    def forward(self, input_ids: list[int] | np.ndarray,
                cache: QwenKVCache | None = None) -> tuple["mx_types.array", QwenKVCache]:
        """Run one prompt chunk and return logits for every supplied token.

        ``cache`` may contain a preceding prompt/decode prefix.  The returned
        cache owns K/V arrays for the complete prefix plus ``input_ids``.
        """
        mx = self.mx
        token_array = np.asarray(input_ids, dtype=np.int32)
        if token_array.ndim != 1 or token_array.size == 0:
            raise ValueError("input_ids must be a non-empty one-dimensional sequence")
        if np.any(token_array < 0) or np.any(token_array >= self.config.vocab_size):
            raise ValueError("token id is outside the vocabulary")
        past_tokens = 0 if cache is None else cache.token_count
        if past_tokens + token_array.size > self.max_model_length:
            raise ValueError("request exceeds the configured context limit")
        if cache is not None and len(cache.layers) != self.config.num_hidden_layers:
            raise ValueError("KV cache layer count does not match the model")

        tokens = mx.array(token_array)
        hidden = self.weights["model.embed_tokens.weight"][tokens]
        positions = mx.arange(past_tokens, past_tokens + token_array.size, dtype=mx.int32)
        next_layers: list[LayerKV] = []
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            normalized = self._rms_norm(hidden, prefix + "input_layernorm.weight")
            query = self._linear(
                normalized,
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.q_proj.bias",
            ).reshape(token_array.size, self.config.num_attention_heads, self.head_dim)
            key = self._linear(
                normalized,
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.k_proj.bias",
            ).reshape(token_array.size, self.config.num_key_value_heads, self.head_dim)
            value = self._linear(
                normalized,
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.v_proj.bias",
            ).reshape(token_array.size, self.config.num_key_value_heads, self.head_dim)
            query = self._rope(query, positions)
            key = self._rope(key, positions)
            if cache is not None:
                key = mx.concatenate([cache.layers[layer].key, key], axis=0)
                value = mx.concatenate([cache.layers[layer].value, value], axis=0)
            attention = self._attention(query, key, value, past_tokens)
            hidden = hidden + self._linear(attention, prefix + "self_attn.o_proj.weight")

            normalized = self._rms_norm(hidden, prefix + "post_attention_layernorm.weight")
            gate = self._linear(normalized, prefix + "mlp.gate_proj.weight")
            up = self._linear(normalized, prefix + "mlp.up_proj.weight")
            mlp = (gate * mx.sigmoid(gate)) * up
            hidden = hidden + self._linear(mlp, prefix + "mlp.down_proj.weight")
            next_layers.append(LayerKV(key, value))

        normalized = self._rms_norm(hidden, "model.norm.weight")
        logits = self._linear(normalized, "lm_head.weight").astype(mx.float32)
        result_cache = QwenKVCache(next_layers, past_tokens + token_array.size)
        mx.eval(logits, *[item for layer in next_layers for item in (layer.key, layer.value)])
        return logits, result_cache

    def prefill_logits(self, input_ids: list[int] | np.ndarray) -> np.ndarray:
        logits, _ = self.forward(input_ids)
        return np.asarray(logits[-1])

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "MlxQwenModel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
