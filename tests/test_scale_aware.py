from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from forge_llm.cached_policy import validate_search
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import seal_policy, validate_corpora, validate_policy
from forge_llm.quantization import dequantize_per_channel, quantize_model
from forge_llm.scale_aware import (
    ALGORITHM,
    DEFAULT_CONFIG,
    DEFAULT_SEARCH,
    POLICY_ALGORITHM,
    quantize_scale_aware,
    validate_config,
)
from forge_llm.second_order import (
    DEFAULT_CONFIG as BASE_CONFIG,
    CalibrationStats,
    block_moments,
    quantize_second_order,
    write_calibration_stats,
)
from forge_llm.validate_int8 import verify_derivation
from test_quantization import tiny_artifact
from test_second_order import fixture_stats


def test_scale_fit_improves_independent_fp16_objective_and_never_worsens_a_row():
    random = np.random.default_rng(711)
    inputs = random.normal(size=(193, 19))
    inputs[:, 0] = 0  # A large source coefficient on an unobserved feature.
    weights = random.normal(size=(31, 19)).astype(np.float16)
    weights[:, 0] = 20
    weights[0] = 0
    config = {**DEFAULT_CONFIG, "block_size": 4, "row_chunk_size": 7}
    moments = block_moments(inputs, 4)
    q, scales, metrics = quantize_scale_aware(weights, moments, config)
    old_q, old_scales, _ = quantize_second_order(
        weights, moments, {**BASE_CONFIG, "block_size": 4}
    )
    losses = []
    for packed, scale in [(old_q, old_scales), (q, scales)]:
        error = weights.astype(np.float64) - dequantize_per_channel(
            packed, scale
        ).astype(np.float64)
        losses.append(
            sum(
                np.square(inputs[:, s : s + 4] @ error[:, s : s + 4].T).mean(axis=0)
                for s in range(0, 19, 4)
            )
        )
    assert np.all(losses[1] <= losses[0] + 1e-12)
    assert losses[1].sum() < losses[0].sum()
    assert metrics["fixed_scale_block_objective"] == pytest.approx(losses[0].sum())
    assert metrics["calibrated_block_objective"] == pytest.approx(losses[1].sum())
    assert metrics["changed_scale_rows"] > 0
    assert scales[0] == 1 and np.all(q[0] == 0)
    assert q.dtype == np.int8 and scales.dtype == np.float32
    assert np.min(q) >= -127 and np.max(q) <= 127
    again_q, again_s, again_m = quantize_scale_aware(weights, moments, config)
    assert q.tobytes() == again_q.tobytes() and scales.tobytes() == again_s.tobytes()
    assert metrics == again_m


@pytest.mark.parametrize("dead", [False, True])
def test_identity_grid_and_dead_objectives_retain_old_recipe_bytes(dead):
    random = np.random.default_rng(99)
    weights = random.normal(size=(9, 13)).astype(np.float16)
    config = {**DEFAULT_CONFIG, "block_size": 4}
    if not dead:
        config["scale_factors"] = [1.0]
    inputs = np.zeros((20, 13)) if dead else random.normal(size=(20, 13))
    moments = block_moments(inputs, 4)
    q, scales, metrics = quantize_scale_aware(weights, moments, config)
    old_q, old_scales, _ = quantize_second_order(
        weights, moments, {**BASE_CONFIG, "block_size": 4}
    )
    assert q.tobytes() == old_q.tobytes() and scales.tobytes() == old_scales.tobytes()
    assert metrics["accepted_candidate_rows"] == 0
    assert metrics["baseline_fallback_rows"] == len(weights)


@pytest.mark.parametrize(
    "change",
    [
        {"scale_factors": []},
        {"scale_factors": [0.99]},
        {"scale_factors": [1, 1]},
        {"scale_factors": [1, 0.9, 0.95]},
        {"scale_factors": [True]},
        {"scale_factors": [1, float("nan")]},
        {"scale_factors": [1, 0.49]},
        {"scale_factors": [1, 1.01]},
        {"row_chunk_size": True},
        {"row_chunk_size": 0},
        {"row_chunk_size": 4097},
        {"damping": 0},
        {"unexpected": 1},
    ],
)
def test_invalid_scale_config_fails_closed(change):
    with pytest.raises(ValueError, match="configuration"):
        validate_config({**DEFAULT_CONFIG, **change})


def test_scale_fit_input_and_scale_override_guards():
    weight = np.ones((3, 5), dtype=np.float16)
    moments = block_moments(np.eye(5), 4)
    config = {**DEFAULT_CONFIG, "block_size": 4}
    with pytest.raises(ValueError, match="FP16"):
        quantize_scale_aware(weight.astype(np.float32), moments, config)
    with pytest.raises(ValueError):
        quantize_scale_aware(weight, moments.astype(np.float32), config)
    with pytest.raises(ValueError):
        quantize_scale_aware(np.full_like(weight, np.nan), moments, config)
    for scales in [
        np.ones(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float64),
        np.full(3, np.nan, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    ]:
        with pytest.raises(ValueError, match="scales"):
            quantize_second_order(
                weight, moments, {**BASE_CONFIG, "block_size": 4}, scales=scales
            )


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_scale_archive_export_exact_derivation_and_method_identity(tmp_path, family):
    source_path, stats_path = tmp_path / "source.engine", tmp_path / "stats.npz"
    tiny_artifact(source_path, family)
    with ModelFile(source_path) as source:
        metadata, arrays, _ = fixture_stats(source, tmp_path / "legacy.npz")
        metadata = {
            **metadata,
            "algorithm": ALGORITHM,
            "quantizer_config": {**DEFAULT_CONFIG, "block_size": 4},
        }
        checksum = write_calibration_stats(stats_path, metadata, arrays)
        recipe = CalibrationStats(stats_path, source, expected_sha=checksum)
        names = set(metadata["weight_to_moments"])
        retained = [min(names)]
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": POLICY_ALGORITHM,
                "source_data_sha256": source.data_sha256,
                "quantizer_config": metadata["quantizer_config"],
                "calibration_stats_sha256": checksum,
                "search_config": DEFAULT_SEARCH,
                "retained_fp16": retained,
                "quantized_weights": sorted(names - set(retained)),
                "calibration_passed": False,
            }
        )
        assert recipe.method == ALGORITHM
        assert validate_policy(policy, source.data_sha256, names) == retained
        output, repeat = tmp_path / "packed.engine", tmp_path / "repeat.engine"
        manifest = quantize_model(
            source_path, output, policy=policy, calibration_stats=stats_path
        )
        quantize_model(source_path, repeat, policy=policy, calibration_stats=stats_path)
        assert output.read_bytes() == repeat.read_bytes()
        assert manifest["quantization_method"] == ALGORITHM
        with ModelFile(output) as packed:
            assert verify_derivation(source, packed, stats_path)["method"] == ALGORITHM
            with pytest.raises(ValueError, match="does not derive"):
                verify_derivation(source, packed)
            assert (
                packed.tensor_numpy(retained[0]).tobytes()
                == source.tensor_numpy(retained[0]).tobytes()
            )
        with pytest.raises(FileExistsError):
            quantize_model(
                source_path, output, policy=policy, calibration_stats=stats_path
            )
        for change in [
            {"algorithm": "block_second_order_cached_repair_v1"},
            {"calibration_stats_sha256": "0" * 64},
            {"search_config": {}},
        ]:
            with pytest.raises(ValueError):
                quantize_model(
                    source_path,
                    tmp_path / "invalid.engine",
                    policy=seal_policy({**policy, **change}),
                    calibration_stats=stats_path,
                )
        for change in [
            {"algorithm": "block_second_order_joint_forward_v1"},
            {
                "quantizer_config": {
                    **metadata["quantizer_config"],
                    "scale_factors": [1, 1],
                }
            },
        ]:
            path = tmp_path / f"bad-{len(list(tmp_path.glob('bad-*')))}.npz"
            write_calibration_stats(path, {**metadata, **change}, arrays)
            with pytest.raises(ValueError):
                CalibrationStats(path, source)


def test_scale_corpora_are_fresh_and_search_is_bounded():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    calibration = json.loads((root / "scale-calibration-prompts.json").read_text())
    regression = json.loads((root / "scale-regression-prompts.json").read_text())
    assert len(calibration) == 32 and len(regression) == 25
    validate_corpora(calibration, regression)
    for name in [
        "quality-prompts.json",
        "calibration-prompts.json",
        "second-order-calibration-prompts.json",
        "second-order-regression-prompts.json",
        "cached-calibration-prompts.json",
        "cached-regression-prompts.json",
    ]:
        old = json.loads((root / name).read_text())
        validate_corpora(calibration, old)
        validate_corpora(regression, old)
    validate_search(DEFAULT_SEARCH)
    saved = copy.deepcopy(DEFAULT_CONFIG)
    validate_config(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG == saved
