"""Safe, memory-mapped reader for Forge model containers.

This mirrors the C++ ``forge::ModelFile`` contract so every execution backend
consumes the same validated artifact.  NumPy views point directly into the
read-only mapping; a backend decides when and how to copy them to its device.
"""

from __future__ import annotations

import hashlib
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .format import ALIGNMENT, CONFIG, HEADER, MAGIC, TENSOR, VERSION, ModelConfig

_DTYPES: dict[int, np.dtype] = {
    1: np.dtype("<f2"),
    2: np.dtype("<f4"),
    3: np.dtype("<i4"),
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
        (magic, version, metadata_size, data_start, data_size,
         metadata_digest, data_digest) = HEADER.unpack_from(self._mapping)
        if magic != MAGIC:
            raise ValueError("invalid model magic")
        if version != VERSION:
            raise ValueError(f"unsupported model format version {version}")
        metadata_end = HEADER.size + metadata_size
        if metadata_end > len(self._mapping):
            raise ValueError("metadata extends beyond the model file")
        if data_start % ALIGNMENT:
            raise ValueError("tensor data is not aligned")
        if data_start < metadata_end or data_start + data_size != len(self._mapping):
            raise ValueError("invalid tensor data extent")

        metadata = self._mapping[HEADER.size:metadata_end]
        if hashlib.sha256(metadata).digest() != metadata_digest:
            raise ValueError("metadata checksum mismatch")
        data = memoryview(self._mapping)[data_start:data_start + data_size]
        try:
            actual_data_digest = hashlib.sha256(data).digest()
        finally:
            data.release()
        if actual_data_digest != data_digest:
            raise ValueError("tensor data checksum mismatch")

        if len(metadata) < CONFIG.size:
            raise ValueError("truncated model configuration")
        values = CONFIG.unpack_from(metadata)
        self.config = ModelConfig(*values[:-1])
        self._validate_config(self.config)
        tensor_count = values[-1]
        if tensor_count >= 10_000:
            raise ValueError("implausible tensor count")

        cursor = CONFIG.size
        tensors: dict[str, TensorInfo] = {}
        ordered: list[TensorInfo] = []
        for _ in range(tensor_count):
            if cursor + TENSOR.size > len(metadata):
                raise ValueError("truncated tensor metadata")
            name_size, dtype_id, rank, offset, nbytes = TENSOR.unpack_from(metadata, cursor)
            cursor += TENSOR.size
            if not 1 <= rank <= 8:
                raise ValueError("invalid tensor rank")
            shape_size = rank * 4
            if cursor + shape_size + name_size > len(metadata):
                raise ValueError("truncated tensor shape or name")
            shape = struct.unpack_from(f"<{rank}I", metadata, cursor)
            cursor += shape_size
            try:
                name = metadata[cursor:cursor + name_size].decode("utf-8")
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
        if cursor != len(metadata):
            raise ValueError("unexpected trailing model metadata")

        by_offset = sorted(ordered, key=lambda tensor: tensor.offset)
        for previous, current in zip(by_offset, by_offset[1:]):
            if current.offset == previous.offset:
                if (current.nbytes, current.dtype, current.shape) != (
                        previous.nbytes, previous.dtype, previous.shape):
                    raise ValueError("incompatible tensors share a data offset")
            elif previous.offset + previous.nbytes > current.offset:
                raise ValueError("tensor data regions partially overlap")

        self.data_start = data_start
        self.data_size = data_size
        self.data_sha256 = data_digest.hex()
        self.tensors = tensors

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
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden size must divide attention heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("query heads must divide KV heads")
        if (config.hidden_size // config.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= config.eos_token_id < config.vocab_size:
            raise ValueError("EOS token is outside the vocabulary")
        if config.rope_theta <= 0 or config.rms_norm_eps <= 0:
            raise ValueError("invalid RoPE or RMSNorm configuration")

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

    def __enter__(self) -> "ModelFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
