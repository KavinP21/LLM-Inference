from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from forge_llm.calibrate_second_order import calibrate
from forge_llm.mlx_engine import MlxEngine
from forge_llm.quantization import quantize_model
from forge_llm.validate_int8 import validate
from test_quantization import tiny_artifact


@pytest.mark.skipif(sys.platform != "darwin", reason="MLX requires Apple Silicon")
@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_tiny_calibration_export_and_runtime_contract(tmp_path, monkeypatch, family):
    import transformers

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(
                input_ids=[1, 2, 3] if text == "calibrate" else [4, 5, 6]
            )

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    source, packed, stats = [
        tmp_path / n for n in ["source.engine", "packed.engine", "stats.npz"]
    ]
    tiny_artifact(source, family)
    report = calibrate(
        source,
        "offline-stub",
        ["calibrate"],
        ["regression"],
        stats,
        output_tokens=4,
        positions=(0, 1, 3),
        probe_count=2,
        candidate_pool=3,
    )
    assert report["cache_reclaimed"]
    assert report["policy"]["quantized_projection_fraction"] >= 0.25
    assert len(report["joint_selection"]) > 1
    quantize_model(source, packed, policy=report["policy"], calibration_stats=stats)
    quality = validate(
        source,
        packed,
        "offline-stub",
        ["regression"],
        4,
        [0, 1, 3],
        calibration_stats=stats,
        decode_mode="rowwise",
    )
    assert quality["derivation"]["method"] == "block_second_order_v1"
    assert quality["gates"]["cache_reclaimed"]
    assert quality["gates"]["kernel_exact_greedy"]
    with MlxEngine(
        packed, max_model_length=32, kv_cache_bytes=1 << 20, decode_mode="rowwise"
    ) as engine:
        assert len(engine.generate([4, 5, 6], 4)) == 4
