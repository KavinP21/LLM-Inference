from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_llm.cached_decode import replay_cached
from forge_llm.calibrate_cached import calibrate
from forge_llm.mlx_engine import MlxEngine
from forge_llm.quantization import quantize_model
from forge_llm.validate_int8 import validate
from test_quantization import tiny_artifact

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
@pytest.mark.parametrize("mode", ["fp16", "metal", "reconstruct"])
def test_cached_teacher_forcing_matches_scheduler_bitwise(
    tmp_path, monkeypatch, family, mode
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from validate_batch_numerics import fingerprint, observe_logits

    source = tmp_path / "source.engine"
    tiny_artifact(source, family)
    model = source
    if mode != "fp16":
        model = tmp_path / "int8.engine"
        quantize_model(source, model)
    with MlxEngine(
        model,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=7,
        decode_mode="rowwise",
        int8_mode="auto" if mode == "fp16" else mode,
    ) as engine:
        ids = [1, 2, 3] * 5
        rows, tokens = [], []
        with observe_logits(engine) as observed:
            request = engine.submit(ids, 8)
            while len(tokens) < 8:
                for event in engine.step():
                    tokens.append(event.token)
                    rows.append(observed[request]["sha256"])
        before = engine.stats()
        replay = replay_cached(engine, ids, tokens)
        assert [fingerprint(row)["sha256"] for row in replay["logits"]] == rows
        assert replay["greedy_tokens"] == tokens
        assert replay["kv_peak_materialized_blocks"] == 2 and replay["cache_reclaimed"]
        assert engine.stats() == before


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_cached_calibration_export_and_full_decode_quality(
    tmp_path, monkeypatch, family
):
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
        search_config={
            "probe_cases": 1,
            "candidate_pool": 3,
            "repair_rounds": 1,
            "removal_pool": 2,
            "full_trial_count": 2,
        },
    )
    assert report["search"]["summary"]["observations"] == 4
    assert (
        report["cache_reclaimed"]
        and report["policy"]["quantized_projection_fraction"] >= 0.25
    )
    quantize_model(source, packed, policy=report["policy"], calibration_stats=stats)
    quality = validate(
        source,
        packed,
        "offline-stub",
        ["regression"],
        4,
        [0, 1, 2, 3],
        int8_mode="reconstruct",
        calibration_stats=stats,
        decode_mode="rowwise",
        cached_teacher_forcing=True,
    )
    assert quality["configuration"]["teacher_forcing_path"] == "cached_paged_decode"
    assert quality["summary"]["bitwise_kernel_logit_rows"] == 4
    assert (
        quality["gates"]["cache_reclaimed"] and quality["gates"]["kernel_exact_greedy"]
    )
