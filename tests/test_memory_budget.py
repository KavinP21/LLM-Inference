"""The host memory ceiling across model workers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from instinct.agents import cli
from instinct.agents.backends import memory

GB = memory.GB


def _model(folder: Path, size: int) -> str:
    folder.mkdir(parents=True)
    with open(folder / "model.safetensors", "wb") as stream:
        stream.truncate(size)  # Sparse: no real disk use.
    return str(folder)


def _config(tmp_path, workers, **extra):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"workers": workers, **extra}))
    return path


def test_every_local_worker_counts_lazy_ones_included(tmp_path):
    big = _model(tmp_path / "big", 16 * GB)
    small = _model(tmp_path / "small", 3 * GB)
    workers = [
        {"name": "a", "model": big, "tokenizer": big, "kv_cache_bytes": 26 * GB},
        {
            "name": "b",
            "model": big,
            "tokenizer": big,
            "kv_cache_bytes": 26 * GB,
            "lazy": True,
        },
        {"name": "c", "model": small, "tokenizer": small, "kv_cache_bytes": 9 * GB},
        {"type": "remote", "name": "far", "url": "https://x", "token_env": "T"},
    ]
    with pytest.raises(ValueError, match="over the 105 GB limit") as raised:
        cli.load_config(_config(tmp_path, workers))
    assert "a 45.3 GB" in str(raised.value) and "far" not in str(raised.value)
    workers[1]["kv_cache_bytes"] = 12 * GB
    loaded = cli.load_config(_config(tmp_path, workers))
    report = memory.budget(loaded)
    assert set(report["workers"]) == {"a", "b", "c"}
    assert report["total_bytes"] <= 105 * GB
    # The ceiling is configurable.
    with pytest.raises(ValueError, match="over the 80 GB limit"):
        cli.load_config(_config(tmp_path, workers, memory={"max_gb": 80}))
    with pytest.raises(ValueError, match="max_gb"):
        cli.load_config(_config(tmp_path, workers, memory={"max_gb": "lots"}))


def test_the_shipped_profile_fits_in_105_gb():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "examples" / "instinct.json").read_text())
    assert config["memory"] == {"max_gb": 105}
    # Sizes of the checkpoints it names, so this holds without them downloaded.
    sizes = {"gemma-4-26b-a4b": 15_642_000_000, "qwen3.5-4b": 3_062_000_000}
    total = memory.HARNESS_BYTES
    for worker in config["workers"]:
        weights = next(v for k, v in sizes.items() if k in worker["model"])
        total += (
            weights
            + worker["kv_cache_bytes"]
            + memory.ACTIVATION_BYTES
            + memory.PROCESS_BYTES
        )
    assert total <= 105 * GB, total / GB


def test_mlx_workers_get_their_share_as_a_memory_limit(tmp_path, monkeypatch):
    model = _model(tmp_path / "m", 2 * GB)
    seen = []

    class Process:
        @classmethod
        async def start(cls, definition, **_):
            seen.append(definition)
            return cls()

        async def close(self):
            pass

    monkeypatch.setattr(cli, "LocalProcessBackend", Process)
    import asyncio

    pool = asyncio.run(
        cli.create_pool(
            {"workers": [{"model": model, "tokenizer": model, "kv_cache_bytes": GB}]}
        )
    )
    asyncio.run(pool.close())
    limit = seen[0].engine_options["memory_limit_bytes"]
    assert limit == 2 * GB + GB + memory.ACTIVATION_BYTES
