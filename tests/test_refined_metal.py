from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from forge_llm.cached_decode import replay_cached
from forge_llm.calibrate_cached import calibrate
from forge_llm.mlx_engine import MlxEngine
from forge_llm.quantization import quantize_model
from forge_llm.refined import ALGORITHM, DEFAULT_CONFIG
from forge_llm.validate_int8 import validate
from test_quantization import tiny_artifact

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_refined_calibration_export_native_composed_cached_identity(
    tmp_path, monkeypatch, family
):
    import transformers

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 2, 3] if text == "fit" else [4, 5, 6])

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
        ["fit"],
        ["check"],
        stats,
        output_tokens=4,
        weight_method="coordinate-refined",
        config={**DEFAULT_CONFIG, "block_size": 4},
        search_config={
            "probe_cases": 1,
            "candidate_pool": 2,
            "repair_rounds": 1,
            "removal_pool": 1,
            "full_trial_count": 1,
        },
    )
    quantize_model(source, packed, policy=report["policy"], calibration_stats=stats)
    quality = validate(
        source,
        packed,
        "offline-stub",
        ["check"],
        4,
        [0, 1, 2, 3],
        int8_mode="reconstruct",
        calibration_stats=stats,
        decode_mode="rowwise",
        cached_teacher_forcing=True,
    )
    assert quality["derivation"]["method"] == ALGORITHM
    assert (
        quality["gates"]["kernel_exact_greedy"] and quality["gates"]["cache_reclaimed"]
    )
    teacher = quality["cases"][0]["fp16_tokens"]
    values = []
    for mode in ("reconstruct", "dequantize"):
        with MlxEngine(
            packed,
            max_model_length=64,
            kv_cache_bytes=1 << 20,
            decode_mode="rowwise",
            int8_mode=mode,
        ) as engine:
            replay = replay_cached(engine, [4, 5, 6], teacher)
            assert replay["cache_reclaimed"]
            values.append(replay["logits"])
    assert (
        values[0].dtype == values[1].dtype
        and values[0].tobytes() == values[1].tobytes()
    )
