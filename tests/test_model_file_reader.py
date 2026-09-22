from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge_llm.format import ModelConfig, write_engine
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


def test_python_reader_round_trips_aliases(tmp_path: Path) -> None:
    path = tmp_path / "model.engine"
    _write_minimal(path)
    with ModelFile(path) as model:
        embedding = model.tensor_numpy("model.embed_tokens.weight")
        head = model.tensor_numpy("lm_head.weight")
        assert model.config.hidden_size == 4
        assert model.tensor_info("model.embed_tokens.weight").offset == model.tensor_info(
            "lm_head.weight"
        ).offset
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

