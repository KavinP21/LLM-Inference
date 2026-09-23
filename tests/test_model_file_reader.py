from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest
from forge_llm.format import (
    ALIGNMENT,
    CONFIG_V1,
    HEADER,
    MAGIC,
    TENSOR,
    ModelConfig,
    write_engine,
)
from forge_llm.model_file import ModelFile


def _write_minimal(path: Path) -> None:
    config = ModelConfig(8, 4, 8, 1, 1, 1, 32, 2, 10_000.0, 1e-6)
    embedding = np.arange(32, dtype=np.float16).reshape(8, 4)
    write_engine(
        path,
        config,
        {
            "model.embed_tokens.weight": embedding,
            "lm_head.weight": embedding,
        },
        aliases={"lm_head.weight": "model.embed_tokens.weight"},
    )


def _write_v1_minimal(path: Path) -> None:
    name = b"model.embed_tokens.weight"
    tensor = np.arange(32, dtype=np.float16).reshape(8, 4)
    data = tensor.tobytes()
    metadata = bytearray(CONFIG_V1.pack(8, 4, 8, 1, 1, 1, 32, 2, 10_000.0, 1e-6, 1))
    metadata.extend(TENSOR.pack(len(name), 1, 2, 0, len(data)))
    metadata.extend(struct.pack("<2I", *tensor.shape))
    metadata.extend(name)
    data_start = (HEADER.size + len(metadata) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    header = HEADER.pack(
        MAGIC,
        1,
        len(metadata),
        data_start,
        len(data),
        hashlib.sha256(metadata).digest(),
        hashlib.sha256(data).digest(),
    )
    path.write_bytes(
        header + metadata + bytes(data_start - len(header) - len(metadata)) + data
    )


def test_python_reader_round_trips_aliases(tmp_path: Path) -> None:
    path = tmp_path / "model.engine"
    _write_minimal(path)
    with ModelFile(path) as model:
        embedding = model.tensor_numpy("model.embed_tokens.weight")
        head = model.tensor_numpy("lm_head.weight")
        assert model.config.hidden_size == 4
        assert (
            model.tensor_info("model.embed_tokens.weight").offset
            == model.tensor_info("lm_head.weight").offset
        )
        np.testing.assert_array_equal(embedding, head)
        assert not embedding.flags.writeable


def test_python_reader_rejects_corruption(tmp_path: Path) -> None:
    path = tmp_path / "model.engine"
    _write_minimal(path)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0xFF
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        ModelFile(path)


def test_python_reader_remains_backward_compatible_with_v1(tmp_path: Path) -> None:
    path = tmp_path / "model-v1.engine"
    _write_v1_minimal(path)
    with ModelFile(path) as model:
        assert model.version == 1
        assert model.config.model_type == "qwen2"
        assert model.config.activation == "silu"
        assert model.config.attention_head_dim == 4
        assert model.config.embedding_scale == 1.0
        np.testing.assert_array_equal(
            model.tensor_numpy("model.embed_tokens.weight"),
            np.arange(32, dtype=np.float16).reshape(8, 4),
        )


def test_v2_gemma_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "gemma.engine"
    config = ModelConfig(
        64,
        24,
        48,
        2,
        2,
        1,
        64,
        1,
        1_000_000.0,
        1e-6,
        model_type="gemma3_text",
        head_dim=8,
        sliding_window=4,
        sliding_window_pattern=2,
        activation="gelu_pytorch_tanh",
        rope_local_theta=10_000.0,
        query_pre_attn_scalar=8.0,
        embedding_scale=24**0.5,
        norm_weight_offset=1.0,
    )
    write_engine(
        path,
        config,
        {"model.embed_tokens.weight": np.ones((64, 24), dtype=np.float16)},
    )
    with ModelFile(path) as model:
        assert model.version == 2
        assert model.config.model_type == config.model_type
        assert model.config.head_dim == config.head_dim
        assert model.config.sliding_window == config.sliding_window
        assert model.config.sliding_window_pattern == config.sliding_window_pattern
        assert model.config.activation == config.activation
        assert model.config.rms_norm_eps == pytest.approx(config.rms_norm_eps)
        assert model.config.embedding_scale == pytest.approx(config.embedding_scale)
        assert model.config.is_sliding_layer(0)
        assert not model.config.is_sliding_layer(1)
