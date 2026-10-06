"""Gemma 3 text decoder adapter for the shared MLX paged runtime."""

from __future__ import annotations

import math

import numpy as np

from ..model_contract import validate_gemma3_weights
from ..paged_kv import MlxPagedKVStore
from .mlx import MlxQwenModel


class MlxGemma3Model(MlxQwenModel):
    """Text-only Gemma 3 execution with native sliding/global attention.

    The adapter deliberately reuses only architecture-neutral MLX primitives
    from the Qwen implementation. Gemma-specific normalization, Q/K norms,
    residual topology, RoPE bases, activation, and attention scaling live here.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.local_rope_frequencies = 1.0 / (
            self.config.rope_local_theta
            ** (
                self.mx.arange(0, self.head_dim, 2, dtype=self.mx.float32)
                / self.head_dim
            )
        )
        self.mx.eval(self.local_rope_frequencies)

    def _validate_weights(self) -> None:
        config = self.config
        validate_gemma3_weights(self.file)
        if config.activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma 3 requires the gelu_pytorch_tanh activation contract"
            )
        if config.attention_head_dim % 2:
            raise ValueError("Gemma 3 RoPE requires an even attention head dimension")

    def _rms_norm(self, x, weight_name: str):
        """Gemma RMSNorm multiplies by ``1 + weight`` before downcasting."""
        mx = self.mx
        value = x.astype(mx.float32)
        variance = mx.mean(value * value, axis=-1, keepdims=True)
        normalized = value * mx.rsqrt(variance + self.config.rms_norm_eps)
        normalized = normalized * (
            self.config.norm_weight_offset
            + self.weights[weight_name].astype(mx.float32)
        )
        return normalized.astype(x.dtype)

    def _rope_frequencies_for_layer(self, layer: int):
        if self.config.is_sliding_layer(layer):
            return self.local_rope_frequencies
        return self.rope_frequencies

    def _attention_scale(self, layer: int) -> float:
        del layer
        return self.config.query_pre_attn_scalar**-0.5

    def _attention_window(self, layer: int) -> int:
        return self.config.sliding_window if self.config.is_sliding_layer(layer) else 0

    def _attention_softcap(self, layer: int) -> float:
        del layer
        return self.config.attn_logit_softcapping

    def _gelu_tanh(self, x):
        mx = self.mx
        value = x.astype(mx.float32)
        coefficient = math.sqrt(2.0 / math.pi)
        result = (
            0.5
            * value
            * (1.0 + mx.tanh(coefficient * (value + 0.044715 * value * value * value)))
        )
        return result.astype(x.dtype)

    def _final_logits(self, hidden, *, decode: bool = False):
        mx = self.mx
        normalized = self._rms_norm(hidden, "model.norm.weight")
        linear = self._decode_linear if decode else self._linear
        logits = linear(normalized, "lm_head.weight").astype(mx.float32)
        softcap = self.config.final_logit_softcapping
        if softcap > 0.0:
            logits = mx.tanh(logits / softcap) * softcap
        return logits

    def _embed(self, token_array: np.ndarray):
        hidden = self.weights["model.embed_tokens.weight"][self.mx.array(token_array)]
        scale = self.mx.array(self.config.embedding_scale, dtype=hidden.dtype)
        return hidden * scale

    def _qkv(self, normalized, prefix: str, token_count: int, *, decode: bool = False):
        linear = self._decode_linear if decode else self._linear
        query = linear(normalized, prefix + "self_attn.q_proj.weight").reshape(
            token_count, self.config.num_attention_heads, self.head_dim
        )
        key = linear(normalized, prefix + "self_attn.k_proj.weight").reshape(
            token_count, self.config.num_key_value_heads, self.head_dim
        )
        value = linear(normalized, prefix + "self_attn.v_proj.weight").reshape(
            token_count, self.config.num_key_value_heads, self.head_dim
        )
        query = self._rms_norm(query, prefix + "self_attn.q_norm.weight")
        key = self._rms_norm(key, prefix + "self_attn.k_norm.weight")
        return query, key, value

    def _layer_body(
        self,
        hidden,
        *,
        layer: int,
        attention,
        decode: bool = False,
    ):
        prefix = f"model.layers.{layer}."
        linear = self._decode_linear if decode else self._linear
        attention_output = linear(attention, prefix + "self_attn.o_proj.weight")
        attention_output = self._rms_norm(
            attention_output, prefix + "post_attention_layernorm.weight"
        )
        hidden = hidden + attention_output
        normalized = self._rms_norm(hidden, prefix + "pre_feedforward_layernorm.weight")
        gate = linear(normalized, prefix + "mlp.gate_proj.weight")
        up = linear(normalized, prefix + "mlp.up_proj.weight")
        mlp_output = linear(self._gelu_tanh(gate) * up, prefix + "mlp.down_proj.weight")
        mlp_output = self._rms_norm(
            mlp_output, prefix + "post_feedforward_layernorm.weight"
        )
        return hidden + mlp_output

    def forward_paged_chunk(
        self,
        input_ids: list[int] | np.ndarray,
        *,
        start_position: int,
        block_table: tuple[int, ...],
        store: MlxPagedKVStore,
    ):
        mx = self.mx
        token_array = np.asarray(input_ids, dtype=np.int32)
        if token_array.ndim != 1 or token_array.size == 0:
            raise ValueError("input_ids must be a non-empty one-dimensional sequence")
        if (
            start_position < 0
            or start_position + token_array.size > self.max_model_length
        ):
            raise ValueError("prompt chunk exceeds the configured context limit")
        if np.any(token_array < 0) or np.any(token_array >= self.config.vocab_size):
            raise ValueError("token id is outside the vocabulary")

        hidden = self._embed(token_array)
        touched: set[int] = set()
        key_length = start_position + token_array.size
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            normalized = self._rms_norm(hidden, prefix + "input_layernorm.weight")
            query, key, value = self._qkv(normalized, prefix, token_array.size)
            query, key, written = self._rope_write_prefill(
                query,
                key,
                value,
                layer=layer,
                block_table=block_table,
                start_position=start_position,
                store=store,
            )
            touched.update(written)
            attention = self._paged_attention(
                query, layer, block_table, key_length, start_position, store
            )
            hidden = self._layer_body(hidden, layer=layer, attention=attention)

        logits = self._final_logits(hidden)
        mx.eval(logits)
        store.materialize(touched)
        return logits

    def decode_paged_batch(
        self,
        input_ids: list[int],
        *,
        positions: list[int],
        block_tables: list[tuple[int, ...]],
        store: MlxPagedKVStore,
    ):
        mx = self.mx
        if not input_ids or not (len(input_ids) == len(positions) == len(block_tables)):
            raise ValueError("decode batch metadata does not align")
        if any(
            position < 0 or position >= self.max_model_length for position in positions
        ):
            raise ValueError("decode position is out of range")
        token_array = np.asarray(input_ids, dtype=np.int32)
        if np.any(token_array < 0) or np.any(token_array >= self.config.vocab_size):
            raise ValueError("token id is outside the vocabulary")

        batch = len(input_ids)
        hidden = self._embed(token_array)
        position_array = mx.array(positions, dtype=mx.int32)
        key_lengths = [position + 1 for position in positions]
        touched: set[int] = set()
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            normalized = self._rms_norm(hidden, prefix + "input_layernorm.weight")
            query, key, value = self._qkv(normalized, prefix, batch, decode=True)
            query, key, written = self._rope_write_decode(
                query,
                key,
                value,
                layer=layer,
                block_tables=block_tables,
                positions=positions,
                position_array=position_array,
                store=store,
            )
            touched.update(written)
            attention = self._paged_decode_attention_batch(
                query, layer, block_tables, key_lengths, store
            )
            hidden = self._layer_body(
                hidden, layer=layer, attention=attention, decode=True
            )

        logits = self._final_logits(hidden, decode=True)
        mx.eval(logits)
        store.materialize(touched)
        return logits

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma 3 uses the physical paged execution contract; use MlxEngine"
        )

    def prefill_logits(self, *args, **kwargs):
        raise NotImplementedError(
            "Gemma 3 uses the physical paged execution contract; use "
            "MlxEngine.debug_prefill_logits"
        )
