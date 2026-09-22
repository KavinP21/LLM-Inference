from pathlib import Path

import numpy as np

from forge_llm.format import HEADER, MAGIC, ModelConfig, write_engine


def test_deterministic_engine_file(tmp_path: Path) -> None:
    config = ModelConfig(32, 8, 16, 2, 2, 1, 64, 2, 10000.0, 1e-6)
    tensors = {
        "model.embed_tokens.weight": np.arange(256, dtype=np.float32).reshape(32, 8),
        "model.norm.weight": np.ones(8, dtype=np.float16),
    }
    first, second = tmp_path / "a.engine", tmp_path / "b.engine"
    one = write_engine(first, config, tensors)
    two = write_engine(second, config, tensors)
    assert first.read_bytes() == second.read_bytes()
    magic, version, _, data_start, data_bytes, _, digest = HEADER.unpack_from(first.read_bytes())
    assert magic == MAGIC and version == 1 and data_start % 256 == 0
    assert one["data_sha256"] == two["data_sha256"] == digest.hex()
    assert data_bytes > 0


def test_tied_weight_alias_does_not_duplicate_data(tmp_path: Path) -> None:
    config = ModelConfig(32, 8, 16, 2, 2, 1, 64, 2, 10000.0, 1e-6)
    embedding = np.arange(256, dtype=np.float16).reshape(32, 8)
    path = tmp_path / "tied.engine"
    write_engine(path, config,
                 {"model.embed_tokens.weight": embedding, "lm_head.weight": embedding},
                 aliases={"lm_head.weight": "model.embed_tokens.weight"})
    _, _, _, _, data_bytes, _, _ = HEADER.unpack_from(path.read_bytes())
    assert data_bytes == embedding.nbytes
