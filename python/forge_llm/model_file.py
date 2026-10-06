"""Safe, memory-mapped reader for Forge model containers.

This mirrors the C++ ``forge::ModelFile`` contract so every execution backend
consumes the same validated artifact.  NumPy views point directly into the
read-only mapping; a backend decides when and how to copy them to its device.
"""

from __future__ import annotations

import hashlib
import math
import mmap
import struct
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np

from .format import (
    ACTIVATIONS_BY_ID,
    ALIGNMENT,
    CONFIG_V1,
    CONFIG_V2,
    HEADER,
    MAGIC,
    MODEL_TYPES_BY_ID,
    QUANTIZATION,
    TENSOR,
    VERSION,
    ModelConfig,
    QuantizationSpec,
)

_DTYPES: dict[int, np.dtype] = {
    1: np.dtype("<f2"),
    2: np.dtype("<f4"),
    3: np.dtype("<i4"),
    4: np.dtype("i1"),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]
    offset: int
    nbytes: int


class ModelFile:
    """Validated read-only view of a Forge model artifact."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._handle = self.path.open("rb")
        try:
            self._mapping = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
            self._parse()
        except Exception:
            mapping = getattr(self, "_mapping", None)
            if mapping is not None:
                mapping.close()
            self._handle.close()
            raise

    def _parse(self) -> None:
        if len(self._mapping) < HEADER.size:
            raise ValueError("model file is smaller than the fixed header")
        (
            magic,
            version,
            metadata_size,
            data_start,
            data_size,
            metadata_digest,
            data_digest,
        ) = HEADER.unpack_from(self._mapping)
        if magic != MAGIC:
            raise ValueError("invalid model magic")
        if not 1 <= version <= VERSION:
            raise ValueError(f"unsupported model format version {version}")
        self.version = version
        metadata_end = HEADER.size + metadata_size
        if metadata_end > len(self._mapping):
            raise ValueError("metadata extends beyond the model file")
        if data_start % ALIGNMENT:
            raise ValueError("tensor data is not aligned")
        if data_start < metadata_end or data_start + data_size != len(self._mapping):
            raise ValueError("invalid tensor data extent")

        metadata = self._mapping[HEADER.size : metadata_end]
        if hashlib.sha256(metadata).digest() != metadata_digest:
            raise ValueError("metadata checksum mismatch")
        data = memoryview(self._mapping)[data_start : data_start + data_size]
        try:
            actual_data_digest = hashlib.sha256(data).digest()
        finally:
            data.release()
        if actual_data_digest != data_digest:
            raise ValueError("tensor data checksum mismatch")

        config_struct = CONFIG_V1 if version == 1 else CONFIG_V2
        if len(metadata) < config_struct.size:
            raise ValueError("truncated model configuration")
        values = config_struct.unpack_from(metadata)
        if version == 1:
            self.config = ModelConfig(*values[:-1])
        else:
            (
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                max_position_embeddings,
                eos_token_id,
                model_type_id,
                activation_id,
                sliding_window,
                sliding_window_pattern,
                rope_theta,
                rope_local_theta,
                rms_norm_eps,
                query_pre_attn_scalar,
                embedding_scale,
                attn_logit_softcapping,
                final_logit_softcapping,
                norm_weight_offset,
                tensor_count,
            ) = values
            try:
                model_type = MODEL_TYPES_BY_ID[model_type_id]
                activation = ACTIVATIONS_BY_ID[activation_id]
            except KeyError as exc:
                raise ValueError(
                    f"unknown model configuration enum {exc.args[0]}"
                ) from exc
            self.config = ModelConfig(
                vocab_size,
                hidden_size,
                intermediate_size,
                num_hidden_layers,
                num_attention_heads,
                num_key_value_heads,
                max_position_embeddings,
                eos_token_id,
                rope_theta,
                rms_norm_eps,
                model_type,
                head_dim,
                sliding_window,
                sliding_window_pattern,
                activation,
                rope_local_theta,
                query_pre_attn_scalar,
                embedding_scale,
                attn_logit_softcapping,
                final_logit_softcapping,
                norm_weight_offset,
            )
        self._validate_config(self.config)
        tensor_count = values[-1]
        if tensor_count >= 10_000:
            raise ValueError("implausible tensor count")

        cursor = config_struct.size
        tensors: dict[str, TensorInfo] = {}
        ordered: list[TensorInfo] = []
        for _ in range(tensor_count):
            if cursor + TENSOR.size > len(metadata):
                raise ValueError("truncated tensor metadata")
            name_size, dtype_id, rank, offset, nbytes = TENSOR.unpack_from(
                metadata, cursor
            )
            cursor += TENSOR.size
            if not 1 <= rank <= 8:
                raise ValueError("invalid tensor rank")
            shape_size = rank * 4
            if cursor + shape_size + name_size > len(metadata):
                raise ValueError("truncated tensor shape or name")
            shape = struct.unpack_from(f"<{rank}I", metadata, cursor)
            cursor += shape_size
            try:
                name = metadata[cursor : cursor + name_size].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("tensor name is not valid UTF-8") from exc
            cursor += name_size
            if not name or name in tensors:
                raise ValueError(f"empty or duplicate tensor name: {name!r}")
            try:
                dtype = _DTYPES[dtype_id]
            except KeyError as exc:
                raise ValueError(f"unknown dtype id {dtype_id} for {name}") from exc
            if any(dimension == 0 for dimension in shape):
                raise ValueError(f"zero tensor dimension: {name}")
            elements = 1
            for dimension in shape:
                elements *= dimension
            if elements * dtype.itemsize != nbytes:
                raise ValueError(f"tensor byte count does not match shape: {name}")
            if offset % ALIGNMENT:
                raise ValueError(f"unaligned tensor: {name}")
            if offset > data_size or nbytes > data_size - offset:
                raise ValueError(f"tensor outside data section: {name}")
            info = TensorInfo(name, dtype, tuple(shape), offset, nbytes)
            tensors[name] = info
            ordered.append(info)
        quantization: dict[str, QuantizationSpec] = {}
        if version == 3:
            if cursor + 4 > len(metadata):
                raise ValueError("truncated quantization count")
            count = struct.unpack_from("<I", metadata, cursor)[0]
            cursor += 4
            if not 0 < count <= tensor_count:
                raise ValueError("invalid quantization count")
            for _ in range(count):
                if cursor + QUANTIZATION.size > len(metadata):
                    raise ValueError("truncated quantization descriptor")
                name_size, scheme, axis, scale_size = QUANTIZATION.unpack_from(
                    metadata, cursor
                )
                cursor += QUANTIZATION.size
                if cursor + name_size + scale_size > len(metadata):
                    raise ValueError("truncated quantization names")
                name = metadata[cursor : cursor + name_size].decode("utf-8")
                cursor += name_size
                scale_name = metadata[cursor : cursor + scale_size].decode("utf-8")
                cursor += scale_size
                if not name or not scale_name or name in quantization:
                    raise ValueError("empty or duplicate quantization descriptor")
                if scheme != 1 or axis != 0:
                    raise ValueError("unsupported quantization scheme or axis")
                quantization[name] = QuantizationSpec(scale_name, scheme, axis)
        if cursor != len(metadata):
            raise ValueError("unexpected trailing model metadata")

        by_offset = sorted(ordered, key=lambda tensor: tensor.offset)
        for previous, current in pairwise(by_offset):
            if current.offset == previous.offset:
                if (current.nbytes, current.dtype, current.shape) != (
                    previous.nbytes,
                    previous.dtype,
                    previous.shape,
                ):
                    raise ValueError("incompatible tensors share a data offset")
            elif previous.offset + previous.nbytes > current.offset:
                raise ValueError("tensor data regions partially overlap")

        self.data_start = data_start
        self.data_size = data_size
        self.data_sha256 = data_digest.hex()
        self.tensors = tensors
        self.quantization = quantization
        self._validate_quantization()

    def _validate_quantization(self) -> None:
        for name, spec in self.quantization.items():
            weight = self.tensor_info(name)
            scale = self.tensor_info(spec.scale_name)
            if weight.dtype != np.dtype("i1") or len(weight.shape) != 2:
                raise ValueError(
                    f"INT8 descriptor requires a rank-2 signed INT8 weight: {name}"
                )
            if scale.dtype != np.dtype("<f4") or scale.shape != (weight.shape[0],):
                raise ValueError(
                    f"INT8 scales require one FP32 value per output channel: {name}"
                )
            scales = self.tensor_numpy(spec.scale_name)
            valid = bool(np.all(np.isfinite(scales) & (scales > 0)))
            del scales
            if not valid:
                raise ValueError(f"INT8 scales must be positive and finite: {name}")
            weight_data = self.tensor_numpy(name)
            valid = bool(np.all(weight_data != -128))
            del weight_data
            if not valid:
                raise ValueError(
                    f"symmetric INT8 weights must be in [-127,127]: {name}"
                )
        for name, tensor in self.tensors.items():
            if tensor.dtype == np.dtype("i1") and name not in self.quantization:
                raise ValueError(f"INT8 tensor lacks quantization metadata: {name}")

    @staticmethod
    def _validate_config(config: ModelConfig) -> None:
        dimensions = (
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("model dimensions must be positive")
        if (
            config.vocab_size > 1_000_000
            or config.hidden_size > 65_536
            or config.intermediate_size > 262_144
            or config.num_hidden_layers > 1_024
            or config.max_position_embeddings > 16_777_216
        ):
            raise ValueError("model dimensions exceed runtime safety limits")
        if config.model_type == "qwen2" and (
            config.hidden_size % config.num_attention_heads
            or config.attention_head_dim * config.num_attention_heads
            != config.hidden_size
        ):
            raise ValueError("Qwen hidden size must match its attention width")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("query heads must divide KV heads")
        if config.attention_head_dim <= 0 or config.attention_head_dim % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= config.eos_token_id < config.vocab_size:
            raise ValueError("EOS token is outside the vocabulary")
        floats = (
            config.rope_theta,
            config.rope_local_theta or config.rope_theta,
            config.rms_norm_eps,
            config.query_pre_attn_scalar or config.attention_head_dim,
            config.embedding_scale,
            config.attn_logit_softcapping,
            config.final_logit_softcapping,
            config.norm_weight_offset,
        )
        if any(not math.isfinite(value) for value in floats):
            raise ValueError("model configuration contains a non-finite float")
        if (
            config.rope_theta <= 0
            or (config.rope_local_theta or config.rope_theta) <= 0
            or config.rms_norm_eps <= 0
            or (config.query_pre_attn_scalar or config.attention_head_dim) <= 0
            or config.embedding_scale <= 0
            or config.attn_logit_softcapping < 0
            or config.final_logit_softcapping < 0
        ):
            raise ValueError("invalid RoPE or RMSNorm configuration")
        if config.model_type == "gemma3_text":
            if (
                config.activation != "gelu_pytorch_tanh"
                or config.sliding_window <= 0
                or config.sliding_window_pattern <= 0
                or config.norm_weight_offset != 1.0
            ):
                raise ValueError("unsupported Gemma 3 architecture configuration")
        elif config.model_type != "qwen2" or config.activation != "silu":
            raise ValueError(f"unsupported model type {config.model_type!r}")

    def tensor_info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor: {name}") from exc

    def tensor_numpy(self, name: str) -> np.ndarray:
        info = self.tensor_info(name)
        return np.ndarray(
            info.shape,
            dtype=info.dtype,
            buffer=self._mapping,
            offset=self.data_start + info.offset,
        )

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
            self._handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
