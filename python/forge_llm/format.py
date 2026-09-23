from __future__ import annotations

import hashlib
import io
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"FORGELLM"
VERSION = 2
ALIGNMENT = 256
HEADER = struct.Struct("<8sIIQQ32s32s")
CONFIG_V1 = struct.Struct("<8I2fI")
CONFIG_V2 = struct.Struct("<13I8fI")
CONFIG = CONFIG_V2
TENSOR = struct.Struct("<HBBQQ")
DTYPES = {np.dtype("<f2"): 1, np.dtype("<f4"): 2, np.dtype("<i4"): 3}
MODEL_TYPES = {"qwen2": 1, "gemma3_text": 2}
MODEL_TYPES_BY_ID = {value: key for key, value in MODEL_TYPES.items()}
ACTIVATIONS = {"silu": 1, "gelu_pytorch_tanh": 2}
ACTIVATIONS_BY_ID = {value: key for key, value in ACTIVATIONS.items()}


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
    model_type: str = "qwen2"
    head_dim: int = 0
    sliding_window: int = 0
    sliding_window_pattern: int = 0
    activation: str = "silu"
    rope_local_theta: float = 0.0
    query_pre_attn_scalar: float = 0.0
    embedding_scale: float = 1.0
    attn_logit_softcapping: float = 0.0
    final_logit_softcapping: float = 0.0
    norm_weight_offset: float = 0.0

    @property
    def attention_head_dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    def is_sliding_layer(self, layer: int) -> bool:
        return bool(
            self.sliding_window
            and self.sliding_window_pattern
            and (layer + 1) % self.sliding_window_pattern
        )

    @classmethod
    def from_huggingface(cls, config: object) -> ModelConfig:
        model_type = str(getattr(config, "model_type", ""))
        if model_type not in MODEL_TYPES:
            raise ValueError(f"unsupported model type {model_type!r}")
        eos = config.eos_token_id
        if isinstance(eos, (list, tuple)):
            if not eos:
                raise ValueError("Hugging Face config defines an empty EOS token list")
            eos = eos[0]
        if eos is None:
            raise ValueError("Hugging Face config does not define an EOS token")
        # Transformers 5 moved RoPE fields into ``rope_parameters`` while
        # older Qwen2 configs expose ``rope_theta`` directly.
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            if model_type == "gemma3_text":
                rope_theta = (rope_parameters.get("full_attention") or {}).get(
                    "rope_theta"
                )
            else:
                rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            raise ValueError("Hugging Face config does not define rope_theta")
        if model_type == "gemma3_text":
            local_theta = (rope_parameters.get("sliding_attention") or {}).get(
                "rope_theta"
            )
            local_theta = local_theta or getattr(
                config, "rope_local_base_freq", 10_000.0
            )
            pattern = int(
                getattr(
                    config,
                    "_sliding_window_pattern",
                    getattr(config, "sliding_window_pattern", 0),
                )
            )
            expected_types = (
                [
                    "sliding_attention" if (index + 1) % pattern else "full_attention"
                    for index in range(int(config.num_hidden_layers))
                ]
                if pattern
                else []
            )
            layer_types = list(getattr(config, "layer_types", []) or [])
            if not pattern or layer_types != expected_types:
                raise ValueError(
                    "Gemma 3 export requires its regular sliding-window layer pattern"
                )
        else:
            local_theta = float(rope_theta)
            pattern = 0
        return cls(
            vocab_size=int(config.vocab_size),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_hidden_layers=int(config.num_hidden_layers),
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            max_position_embeddings=int(config.max_position_embeddings),
            eos_token_id=int(eos),
            rope_theta=float(rope_theta),
            rms_norm_eps=float(config.rms_norm_eps),
            model_type=model_type,
            head_dim=int(
                getattr(config, "head_dim", 0)
                or int(config.hidden_size) // int(config.num_attention_heads)
            ),
            sliding_window=int(getattr(config, "sliding_window", 0) or 0),
            sliding_window_pattern=pattern,
            activation=str(getattr(config, "hidden_activation", "silu")),
            rope_local_theta=float(local_theta),
            query_pre_attn_scalar=float(
                getattr(config, "query_pre_attn_scalar", 0.0)
                or int(getattr(config, "head_dim", 0))
                or int(config.hidden_size) // int(config.num_attention_heads)
            ),
            embedding_scale=(
                float(int(config.hidden_size) ** 0.5)
                if model_type == "gemma3_text"
                else 1.0
            ),
            attn_logit_softcapping=float(
                getattr(config, "attn_logit_softcapping", 0.0) or 0.0
            ),
            final_logit_softcapping=float(
                getattr(config, "final_logit_softcapping", 0.0) or 0.0
            ),
            norm_weight_offset=1.0 if model_type == "gemma3_text" else 0.0,
        )


def _align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def write_engine(
    path: Path,
    config: ModelConfig,
    tensors: Mapping[str, np.ndarray],
    aliases: Mapping[str, str] | None = None,
) -> dict:
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
    try:
        model_type_id = MODEL_TYPES[config.model_type]
        activation_id = ACTIVATIONS[config.activation]
    except KeyError as exc:
        raise ValueError(
            f"unsupported model configuration value {exc.args[0]!r}"
        ) from exc
    metadata.write(
        CONFIG_V2.pack(
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_head_dim,
            config.max_position_embeddings,
            config.eos_token_id,
            model_type_id,
            activation_id,
            config.sliding_window,
            config.sliding_window_pattern,
            config.rope_theta,
            config.rope_local_theta or config.rope_theta,
            config.rms_norm_eps,
            config.query_pre_attn_scalar or config.attention_head_dim,
            config.embedding_scale,
            config.attn_logit_softcapping,
            config.final_logit_softcapping,
            config.norm_weight_offset,
            len(entries),
        )
    )
    for name, array, offset in entries:
        encoded = name.encode("utf-8")
        if len(encoded) > 65535 or array.ndim == 0 or array.ndim > 8:
            raise ValueError(f"invalid name or rank for tensor {name}")
        dtype_id = DTYPES.get(array.dtype)
        if dtype_id is None:
            raise TypeError(f"unsupported normalized dtype for {name}: {array.dtype}")
        metadata.write(
            TENSOR.pack(len(encoded), dtype_id, array.ndim, offset, array.nbytes)
        )
        metadata.write(struct.pack(f"<{array.ndim}I", *array.shape))
        metadata.write(encoded)

    metadata_bytes = metadata.getvalue()
    data_bytes = data.getvalue()
    data_start = _align(HEADER.size + len(metadata_bytes))
    header = HEADER.pack(
        MAGIC,
        VERSION,
        len(metadata_bytes),
        data_start,
        len(data_bytes),
        hashlib.sha256(metadata_bytes).digest(),
        hashlib.sha256(data_bytes).digest(),
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
        "model_type": config.model_type,
        "tensors": len(entries),
        "bytes": path.stat().st_size,
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
    }


def write_manifest(path: Path, result: dict, source_model: str) -> None:
    payload = {"source_model": source_model, **result}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
