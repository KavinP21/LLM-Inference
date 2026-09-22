from __future__ import annotations

import hashlib
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

MAGIC = b"FORGELLM"
VERSION = 1
ALIGNMENT = 256
HEADER = struct.Struct("<8sIIQQ32s32s")
CONFIG = struct.Struct("<8I2fI")
TENSOR = struct.Struct("<HBBQQ")
DTYPES = {np.dtype("<f2"): 1, np.dtype("<f4"): 2, np.dtype("<i4"): 3}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    eos_token_id: int
    rope_theta: float
    rms_norm_eps: float

    @classmethod
    def from_huggingface(cls, config: object) -> "ModelConfig":
        eos = getattr(config, "eos_token_id")
        if isinstance(eos, (list, tuple)):
            eos = eos[0]
        # Transformers 5 moved RoPE fields into ``rope_parameters`` while
        # older Qwen2 configs expose ``rope_theta`` directly.
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            raise ValueError("Hugging Face config does not define rope_theta")
        return cls(
            vocab_size=int(getattr(config, "vocab_size")),
            hidden_size=int(getattr(config, "hidden_size")),
            intermediate_size=int(getattr(config, "intermediate_size")),
            num_hidden_layers=int(getattr(config, "num_hidden_layers")),
            num_attention_heads=int(getattr(config, "num_attention_heads")),
            num_key_value_heads=int(getattr(config, "num_key_value_heads")),
            max_position_embeddings=int(getattr(config, "max_position_embeddings")),
            eos_token_id=int(eos),
            rope_theta=float(rope_theta),
            rms_norm_eps=float(getattr(config, "rms_norm_eps")),
        )


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_engine(path: Path, config: ModelConfig, tensors: Mapping[str, np.ndarray],
                 aliases: Mapping[str, str] | None = None) -> dict:
    """Write a deterministic, checksummed Forge model container."""
    aliases = dict(aliases or {})
    for alias, target in aliases.items():
        if alias not in tensors or target not in tensors or alias == target:
            raise ValueError(f"invalid tensor alias {alias!r} -> {target!r}")
    entries: list[tuple[str, np.ndarray, int]] = []
    stored: dict[str, tuple[np.ndarray, int]] = {}
    data = io.BytesIO()
    for name in sorted(tensors):
        canonical = aliases.get(name, name)
        if canonical in stored:
            target_array, offset = stored[canonical]
            array = np.asarray(tensors[name])
            if array.shape != target_array.shape:
                raise ValueError(f"alias shape mismatch for {name}")
            entries.append((name, target_array, offset))
            continue
        array = np.asarray(tensors[canonical])
        if array.dtype.kind == "f":
            array = array.astype("<f2", copy=False)
        elif array.dtype == np.int32:
            array = array.astype("<i4", copy=False)
        else:
            raise TypeError(f"unsupported dtype for {name}: {array.dtype}")
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        padding = _align(data.tell()) - data.tell()
        data.write(b"\0" * padding)
        offset = data.tell()
        data.write(array.tobytes(order="C"))
        stored[canonical] = (array, offset)
        entries.append((name, array, offset))

    metadata = io.BytesIO()
    metadata.write(CONFIG.pack(
        config.vocab_size, config.hidden_size, config.intermediate_size,
        config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads,
        config.max_position_embeddings, config.eos_token_id,
        config.rope_theta, config.rms_norm_eps, len(entries),
    ))
    for name, array, offset in entries:
        encoded = name.encode("utf-8")
        if len(encoded) > 65535 or array.ndim == 0 or array.ndim > 8:
            raise ValueError(f"invalid name or rank for tensor {name}")
        dtype_id = DTYPES.get(array.dtype)
        if dtype_id is None:
            raise TypeError(f"unsupported normalized dtype for {name}: {array.dtype}")
        metadata.write(TENSOR.pack(len(encoded), dtype_id, array.ndim, offset, array.nbytes))
        metadata.write(struct.pack(f"<{array.ndim}I", *array.shape))
        metadata.write(encoded)

    metadata_bytes = metadata.getvalue()
    data_bytes = data.getvalue()
    data_start = _align(HEADER.size + len(metadata_bytes))
    header = HEADER.pack(
        MAGIC, VERSION, len(metadata_bytes), data_start, len(data_bytes),
        hashlib.sha256(metadata_bytes).digest(), hashlib.sha256(data_bytes).digest(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(header)
        output.write(metadata_bytes)
        output.write(b"\0" * (data_start - output.tell()))
        output.write(data_bytes)

    return {
        "path": str(path),
        "format_version": VERSION,
        "tensors": len(entries),
        "bytes": path.stat().st_size,
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
    }


def write_manifest(path: Path, result: dict, source_model: str) -> None:
    payload = {"source_model": source_model, **result}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
