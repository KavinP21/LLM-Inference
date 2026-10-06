"""Shared MLX execution primitives and the Qwen2 model adapter.

The initial backend intentionally uses ordinary MLX operations.  It establishes
the numerical and ownership contracts that later custom Metal kernels must
preserve.  PyTorch and Transformers are not imported anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from ..model_contract import validate_qwen2_weights
from ..model_file import ModelFile
from ..paged_kv import MlxPagedKVStore
from .int8_kernels import Int8LinearKernels
from .metal_kernels import MetalKernelSuite

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
    key: mx_types.array
    value: mx_types.array


@dataclass
class QwenKVCache:
    layers: list[LayerKV]
    token_count: int


class MlxQwenModel:
    """Correctness-first Qwen2 implementation backed exclusively by MLX."""

    _fused_attention_head_dims = frozenset({64, 72, 80, 96, 128, 192, 256})

    def __init__(
        self,
        model_path: str,
        *,
        max_model_length: int | None = None,
        attention_tile_size: int = 1024,
        custom_metal: bool = True,
        metal_paged_attention: bool = True,
        int8_mode: str = "auto",
        decode_mode: str = "batched",
        _model_file: ModelFile | None = None,
    ) -> None:
        self.mx = _mlx()
        self.file = _model_file if _model_file is not None else ModelFile(model_path)
        self.config = self.file.config
        self.max_model_length = (
            self.config.max_position_embeddings
            if max_model_length is None
            else int(max_model_length)
        )
        self.attention_tile_size = attention_tile_size
        self.custom_metal = bool(custom_metal)
        self.metal_paged_attention = bool(metal_paged_attention and custom_metal)
        try:
            if int8_mode not in {"auto", "dequantize", "metal", "reconstruct"}:
                raise ValueError(
                    "int8_mode must be auto, dequantize, metal or reconstruct"
                )
            self.int8_mode = int8_mode
            if decode_mode not in {"batched", "rowwise"}:
                raise ValueError("decode_mode must be batched or rowwise")
            self.decode_mode = decode_mode
            if not 0 < self.max_model_length <= self.config.max_position_embeddings:
                raise ValueError(
                    "runtime context limit exceeds the model context limit"
                )
            if attention_tile_size <= 0:
                raise ValueError("attention_tile_size must be positive")
            self._validate_weights()
            self.weights = self._load_weights()
            self.rope_frequencies = 1.0 / (
                self.config.rope_theta
                ** (
                    self.mx.arange(0, self.head_dim, 2, dtype=self.mx.float32)
                    / self.head_dim
                )
            )
            self.mx.eval(self.rope_frequencies)
            self.metal = MetalKernelSuite(self.mx) if self.custom_metal else None
            self.int8 = (
                Int8LinearKernels(self.mx)
                if self.file.quantization and int8_mode != "dequantize"
                else None
            )
        except Exception:
            self.file.close()
            raise

    @property
    def head_dim(self) -> int:
        return self.config.attention_head_dim

    def _validate_weights(self) -> None:
        validate_qwen2_weights(self.file)

    def _load_weights(self) -> dict[str, mx_types.array]:
        mx = self.mx
        result: dict[str, mx_types.array] = {}
        aliases: dict[tuple[int, tuple[int, ...], str], mx_types.array] = {}
        for name, info in self.file.tensors.items():
            key = (info.offset, info.shape, info.dtype.str)
            if key not in aliases:
                aliases[key] = mx.array(self.file.tensor_numpy(name))
            result[name] = aliases[key]
        # Materialize one device copy now. Generation must not lazily copy mapped
        # host weights during its first timed token.
        mx.eval(*aliases.values())
        return result

    def _decode_linear(self, x, weight_name: str, bias_name: str | None = None):
        if x.ndim != 2:
            raise ValueError("decode projection inputs must have rank two")
        return self._linear(
            x, weight_name, bias_name, rowwise=self.decode_mode == "rowwise"
        )

    def _linear(
        self,
        x,
        weight_name: str,
        bias_name: str | None = None,
        *,
        rowwise: bool = False,
    ):
        """Optional single-row reductions, without splitting attention or prefill.

        A reconstructed matrix is shared by all row GEMVs in this call. This
        policy changes launch/weight-reuse efficiency, not dtype or tie handling.
        """

        def multiply(weight):
            if rowwise and x.shape[0] > 1:
                return self.mx.concatenate(
                    [x[i : i + 1] @ weight.T for i in range(x.shape[0])], axis=0
                )
            return x @ weight.T

        weight = self.weights[weight_name]
        spec = self.file.quantization.get(weight_name)
        if spec is None:
            output = multiply(weight)
        else:
            scales = self.weights[spec.scale_name]
            rows = 1 if rowwise else x.size // x.shape[-1]
            if (
                self.int8 is not None
                and self.int8_mode != "reconstruct"
                and rows <= self.int8.max_rows
            ):
                output = (
                    self.mx.concatenate(
                        [
                            self.int8.linear(x[i : i + 1], weight, scales)
                            for i in range(x.shape[0])
                        ],
                        axis=0,
                    )
                    if rowwise and x.shape[0] > 1
                    else self.int8.linear(x, weight, scales)
                )
            else:
                # Ephemeral FP16 reconstruction feeds MLX's optimized prefill
                # GEMM. It is never cached as a second resident weight copy.
                reconstructed = (
                    self.int8.reconstruct(weight, scales)
                    if self.int8 is not None
                    else (weight.astype(self.mx.float32) * scales[:, None]).astype(
                        self.mx.float16
                    )
                )
                output = multiply(reconstructed)
        if bias_name is not None:
            output = output + self.weights[bias_name]
        return output

    def _rms_norm(self, x, weight_name: str):
        mx = self.mx
        variance = mx.mean(
            x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True
        )
        normalized = x * mx.rsqrt(variance + self.config.rms_norm_eps).astype(x.dtype)
        return normalized * self.weights[weight_name]

    def _residual_rms_norm(self, residual, branch, weight_name: str):
        if self.metal is not None:
            return self.metal.residual_rms_norm(
                residual,
                branch,
                self.weights[weight_name],
                self.config.rms_norm_eps,
            )
        hidden = residual + branch
        return hidden, self._rms_norm(hidden, weight_name)

    def _swiglu(self, gate, up):
        if self.metal is not None:
            return self.metal.swiglu(gate, up)
        return (gate * self.mx.sigmoid(gate)) * up

    def _rope_frequencies_for_layer(self, layer: int):
        del layer
        return self.rope_frequencies

    def _attention_scale(self, layer: int) -> float:
        del layer
        return self.head_dim**-0.5

    def _attention_window(self, layer: int) -> int:
        del layer
        return 0

    def _attention_softcap(self, layer: int) -> float:
        del layer
        return 0.0

    def _rope_write_prefill(
        self,
        query,
        key,
        value,
        *,
        layer: int,
        block_table: tuple[int, ...],
        start_position: int,
        store: MlxPagedKVStore,
    ):
        if self.metal is None:
            positions = self.mx.arange(
                start_position,
                start_position + query.shape[0],
                dtype=self.mx.int32,
            )
            frequencies = self._rope_frequencies_for_layer(layer)
            query = self._rope(query, positions, frequencies)
            key = self._rope(key, positions, frequencies)
            touched = store.write_layer(layer, block_table, start_position, key, value)
            return query, key, touched
        physical_ids, pages = store.prepare_prefill_write(
            layer, block_table, start_position, int(query.shape[0])
        )
        query, key, pages = self.metal.rope_write_prefill(
            query,
            key,
            value,
            pages,
            self._rope_frequencies_for_layer(layer),
            start_position=start_position,
            first_logical_block=start_position // store.block_tokens,
            block_tokens=store.block_tokens,
        )
        store.commit_prefill_write(
            layer,
            physical_ids,
            pages,
            start_position,
            int(query.shape[0]),
        )
        return query, key, physical_ids

    def _rope_write_decode(
        self,
        query,
        key,
        value,
        *,
        layer: int,
        block_tables: list[tuple[int, ...]],
        positions: list[int],
        position_array,
        store: MlxPagedKVStore,
    ):
        if self.metal is None:
            frequencies = self._rope_frequencies_for_layer(layer)
            query = self._rope(query, position_array, frequencies)
            key = self._rope(key, position_array, frequencies)
            touched: set[int] = set()
            for index, table in enumerate(block_tables):
                touched.update(
                    store.write_layer(
                        layer,
                        table,
                        positions[index],
                        key[index : index + 1],
                        value[index : index + 1],
                    )
                )
            return query, key, tuple(touched)
        physical_ids, pages = store.prepare_decode_write(layer, block_tables, positions)
        query, key, pages = self.metal.rope_write_decode(
            query,
            key,
            value,
            pages,
            self._rope_frequencies_for_layer(layer),
            positions=position_array,
            block_tokens=store.block_tokens,
        )
        store.commit_decode_write(layer, physical_ids, pages, positions)
        return query, key, physical_ids

    def _rope(self, tensor, positions, frequencies=None):
        mx = self.mx
        frequencies = self.rope_frequencies if frequencies is None else frequencies
        half = self.head_dim // 2
        angles = positions.astype(mx.float32)[:, None] * frequencies[None, :]
        cosine = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)[:, None, :]
        sine = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)[:, None, :]
        rotated = mx.concatenate([-tensor[..., half:], tensor[..., :half]], axis=-1)
        return tensor * cosine.astype(tensor.dtype) + rotated * sine.astype(
            tensor.dtype
        )

    def _attention(self, query, key, value, query_start: int):
        mx = self.mx
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        key = mx.repeat(key, groups, axis=1)
        value = mx.repeat(value, groups, axis=1)
        # [tokens, heads, dim] -> [heads, tokens, dim]
        q = mx.transpose(query, (1, 0, 2)).astype(mx.float32)
        k = mx.transpose(key, (1, 0, 2)).astype(mx.float32)
        v = mx.transpose(value, (1, 0, 2)).astype(mx.float32)
        scores = (q @ mx.transpose(k, (0, 2, 1))) * (self.head_dim**-0.5)
        query_positions = mx.arange(query_start, query_start + query.shape[0])[:, None]
        key_positions = mx.arange(key.shape[0])[None, :]
        causal = key_positions <= query_positions
        scores = mx.where(causal[None, :, :], scores, mx.array(-1e9, dtype=mx.float32))
        probabilities = mx.softmax(scores, axis=-1)
        output = probabilities @ v
        return (
            mx.transpose(output, (1, 0, 2))
            .reshape(query.shape[0], -1)
            .astype(query.dtype)
        )

    def _tiled_attention(
        self,
        query,
        key,
        value,
        query_start: int,
        *,
        scale: float | None = None,
        window_size: int = 0,
        softcap: float = 0.0,
        key_start: int = 0,
    ):
        """Exact online-softmax attention with bounded score-tile storage."""
        mx = self.mx
        groups = self.config.num_attention_heads // self.config.num_key_value_heads
        key = mx.repeat(key, groups, axis=1)
        value = mx.repeat(value, groups, axis=1)
        q = mx.transpose(query, (1, 0, 2)).astype(mx.float32)
        q_length = int(query.shape[0])
        shape = (self.config.num_attention_heads, q_length, 1)
        running_max = mx.full(shape, -float("inf"), dtype=mx.float32)
        running_sum = mx.zeros(shape, dtype=mx.float32)
        running_output = mx.zeros(
            (self.config.num_attention_heads, q_length, self.head_dim),
            dtype=mx.float32,
        )
        query_positions = mx.arange(query_start, query_start + q_length)[:, None]

        for tile_start in range(0, int(key.shape[0]), self.attention_tile_size):
            tile_end = min(tile_start + self.attention_tile_size, int(key.shape[0]))
            tile_key = mx.transpose(key[tile_start:tile_end], (1, 0, 2)).astype(
                mx.float32
            )
            tile_value = mx.transpose(value[tile_start:tile_end], (1, 0, 2)).astype(
                mx.float32
            )
            scores = (q @ mx.transpose(tile_key, (0, 2, 1))) * (
                self.head_dim**-0.5 if scale is None else scale
            )
            if softcap > 0.0:
                scores = mx.tanh(scores / softcap) * softcap
            key_positions = mx.arange(key_start + tile_start, key_start + tile_end)[
                None, :
            ]
            causal = key_positions <= query_positions
            if window_size:
                causal = causal & (key_positions > query_positions - window_size)
            scores = mx.where(causal[None, :, :], scores, -float("inf"))

            tile_max = mx.max(scores, axis=-1, keepdims=True)
            next_max = mx.maximum(running_max, tile_max)
            previous_scale = mx.exp(running_max - next_max)
            probabilities = mx.exp(scores - next_max)
            running_output = (
                running_output * previous_scale + probabilities @ tile_value
            )
            running_sum = running_sum * previous_scale + mx.sum(
                probabilities, axis=-1, keepdims=True
            )
            running_max = next_max

        output = running_output / running_sum
        return mx.transpose(output, (1, 0, 2)).reshape(q_length, -1).astype(query.dtype)

    def _fast_attention(
        self, query, key, value, *, scale: float | None = None, mask="causal"
    ):
        """Use MLX's fused GQA attention without materializing the score matrix."""
        mx = self.mx
        q = mx.transpose(query, (1, 0, 2))[None, :, :, :]
        k = mx.transpose(key, (1, 0, 2))[None, :, :, :]
        v = mx.transpose(value, (1, 0, 2))[None, :, :, :]
        output = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.head_dim**-0.5 if scale is None else scale,
            mask=mask,
            force_fused=(
                int(key.shape[0]) >= 256
                and self.head_dim in self._fused_attention_head_dims
            ),
        )
        return mx.transpose(output[0], (1, 0, 2)).reshape(query.shape[0], -1)

    def _paged_attention(
        self,
        query,
        layer: int,
        block_table: tuple[int, ...],
        key_length: int,
        query_start: int,
        store: MlxPagedKVStore,
    ):
        scale = self._attention_scale(layer)
        window_size = self._attention_window(layer)
        softcap = self._attention_softcap(layer)
        key_start = max(0, query_start - window_size + 1) if window_size else 0
        key, value = store.gather_layer_range(layer, block_table, key_start, key_length)
        # MLX's fused kernel supports a documented set of production head
        # dimensions. Exact tiling remains the fallback for every other shape.
        if self.head_dim in self._fused_attention_head_dims and softcap <= 0.0:
            mask = "causal"
            if window_size:
                query_positions = self.mx.arange(
                    query_start, query_start + query.shape[0]
                )[:, None]
                key_positions = self.mx.arange(key_start, key_length)[None, :]
                mask = (key_positions <= query_positions) & (
                    key_positions > query_positions - window_size
                )
                mask = mask[None, None, :, :]
            return self._fast_attention(
                query, key, value, scale=scale, mask=mask
            ).astype(query.dtype)
        return self._tiled_attention(
            query,
            key,
            value,
            query_start,
            scale=scale,
            window_size=window_size,
            softcap=softcap,
            key_start=key_start,
        )

    def _paged_decode_attention_batch(
        self,
        query,
        layer: int,
        block_tables: list[tuple[int, ...]],
        key_lengths: list[int],
        store: MlxPagedKVStore,
    ):
        mx = self.mx
        scale = self._attention_scale(layer)
        window_size = self._attention_window(layer)
        softcap = self._attention_softcap(layer)
        if self.metal is not None and self.metal_paged_attention:
            pages, tables, lengths = store.pack_layer(
                layer,
                block_tables,
                key_lengths,
                window_size=window_size,
            )
            output = self.metal.paged_decode_attention(
                query,
                pages,
                tables,
                lengths,
                block_tokens=store.block_tokens,
                scale=scale,
                window_size=window_size,
                softcap=softcap,
            )
            return output.reshape(query.shape[0], -1).astype(query.dtype)
        if self.head_dim not in self._fused_attention_head_dims or softcap > 0.0:
            outputs = []
            for index, (table, length) in enumerate(zip(block_tables, key_lengths)):
                key, value = store.gather_layer(layer, table, length)
                outputs.append(
                    self._tiled_attention(
                        query[index : index + 1],
                        key,
                        value,
                        length - 1,
                        scale=scale,
                        window_size=window_size,
                        softcap=softcap,
                    )
                )
            return mx.concatenate(outputs, axis=0)

        gathered = []
        effective_lengths = []
        for table, length in zip(block_tables, key_lengths):
            key, value = store.gather_layer(layer, table, length)
            if window_size and length > window_size:
                key = key[-window_size:]
                value = value[-window_size:]
            gathered.append((key, value))
            effective_lengths.append(int(key.shape[0]))
        maximum = max(effective_lengths)
        padded_keys = []
        padded_values = []
        for (key, value), length in zip(gathered, effective_lengths):
            padding = maximum - length
            if padding:
                zeros = mx.zeros(
                    (padding, self.config.num_key_value_heads, self.head_dim),
                    dtype=key.dtype,
                )
                key = mx.concatenate([key, zeros], axis=0)
                value = mx.concatenate([value, zeros], axis=0)
            padded_keys.append(mx.transpose(key, (1, 0, 2)))
            padded_values.append(mx.transpose(value, (1, 0, 2)))
        key_batch = mx.stack(padded_keys, axis=0)
        value_batch = mx.stack(padded_values, axis=0)
        query_batch = query[:, :, None, :]
        valid = mx.arange(maximum)[None, :] < mx.array(effective_lengths)[:, None]
        mask = valid[:, None, None, :]
        output = mx.fast.scaled_dot_product_attention(
            query_batch,
            key_batch,
            value_batch,
            scale=scale,
            mask=mask,
            force_fused=maximum >= 256,
        )
        return output[:, :, 0, :].reshape(query.shape[0], -1).astype(query.dtype)

    def forward_paged_chunk(
        self,
        input_ids: list[int] | np.ndarray,
        *,
        start_position: int,
        block_table: tuple[int, ...],
        store: MlxPagedKVStore,
    ):
        """Execute one append-only prompt chunk against a physical paged cache."""
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

        tokens = mx.array(token_array)
        hidden = self.weights["model.embed_tokens.weight"][tokens]
        touched: set[int] = set()
        key_length = start_position + token_array.size
        normalized = self._rms_norm(hidden, "model.layers.0.input_layernorm.weight")

        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
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
            attention_output = self._linear(
                attention, prefix + "self_attn.o_proj.weight"
            )
            hidden, normalized = self._residual_rms_norm(
                hidden,
                attention_output,
                prefix + "post_attention_layernorm.weight",
            )
            gate = self._linear(normalized, prefix + "mlp.gate_proj.weight")
            up = self._linear(normalized, prefix + "mlp.up_proj.weight")
            mlp_output = self._linear(
                self._swiglu(gate, up), prefix + "mlp.down_proj.weight"
            )
            next_norm = (
                f"model.layers.{layer + 1}.input_layernorm.weight"
                if layer + 1 < self.config.num_hidden_layers
                else "model.norm.weight"
            )
            hidden, normalized = self._residual_rms_norm(hidden, mlp_output, next_norm)

        logits = self._linear(normalized, "lm_head.weight").astype(mx.float32)
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
        """Batch projections/MLPs and attention for one token from each request."""
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
        hidden = self.weights["model.embed_tokens.weight"][mx.array(token_array)]
        position_array = mx.array(positions, dtype=mx.int32)
        key_lengths = [position + 1 for position in positions]
        touched: set[int] = set()
        linear = self._decode_linear
        normalized = self._rms_norm(hidden, "model.layers.0.input_layernorm.weight")

        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            query = linear(
                normalized,
                prefix + "self_attn.q_proj.weight",
                prefix + "self_attn.q_proj.bias",
            ).reshape(batch, self.config.num_attention_heads, self.head_dim)
            key = linear(
                normalized,
                prefix + "self_attn.k_proj.weight",
                prefix + "self_attn.k_proj.bias",
            ).reshape(batch, self.config.num_key_value_heads, self.head_dim)
            value = linear(
                normalized,
                prefix + "self_attn.v_proj.weight",
                prefix + "self_attn.v_proj.bias",
            ).reshape(batch, self.config.num_key_value_heads, self.head_dim)
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
            attention_output = linear(attention, prefix + "self_attn.o_proj.weight")
            hidden, normalized = self._residual_rms_norm(
                hidden,
                attention_output,
                prefix + "post_attention_layernorm.weight",
            )
            gate = linear(normalized, prefix + "mlp.gate_proj.weight")
            up = linear(normalized, prefix + "mlp.up_proj.weight")
            mlp_output = linear(self._swiglu(gate, up), prefix + "mlp.down_proj.weight")
            next_norm = (
                f"model.layers.{layer + 1}.input_layernorm.weight"
                if layer + 1 < self.config.num_hidden_layers
                else "model.norm.weight"
            )
            hidden, normalized = self._residual_rms_norm(hidden, mlp_output, next_norm)

        logits = linear(normalized, "lm_head.weight").astype(mx.float32)
        mx.eval(logits)
        store.materialize(touched)
        return logits

    def forward(
        self, input_ids: list[int] | np.ndarray, cache: QwenKVCache | None = None
    ) -> tuple[mx_types.array, QwenKVCache]:
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
        positions = mx.arange(
            past_tokens, past_tokens + token_array.size, dtype=mx.int32
        )
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
            hidden = hidden + self._linear(
                attention, prefix + "self_attn.o_proj.weight"
            )

            normalized = self._rms_norm(
                hidden, prefix + "post_attention_layernorm.weight"
            )
            gate = self._linear(normalized, prefix + "mlp.gate_proj.weight")
            up = self._linear(normalized, prefix + "mlp.up_proj.weight")
            mlp = (gate * mx.sigmoid(gate)) * up
            hidden = hidden + self._linear(mlp, prefix + "mlp.down_proj.weight")
            next_layers.append(LayerKV(key, value))

        normalized = self._rms_norm(hidden, "model.norm.weight")
        logits = self._linear(normalized, "lm_head.weight").astype(mx.float32)
        result_cache = QwenKVCache(next_layers, past_tokens + token_array.size)
        mx.eval(
            logits,
            *[item for layer in next_layers for item in (layer.key, layer.value)],
        )
        return logits, result_cache

    def prefill_logits(self, input_ids: list[int] | np.ndarray) -> np.ndarray:
        logits, _ = self.forward(input_ids)
        return np.asarray(logits[-1])

    def close(self) -> None:
        # Closing an engine must release its resident weights even if callers
        # retain the closed Engine object. MLX may keep freed buffers in its cache.
        self.weights.clear()
        self.file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
