from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import seal_policy, validate_corpora, validate_policy
from forge_llm.quantization import dequantize_per_channel, quantize_model
from forge_llm.refined import (
    ALGORITHM,
    DEFAULT_CONFIG,
    DEFAULT_SEARCH,
    POLICY_ALGORITHM,
    quantize_refined,
    validate_config,
)
from forge_llm.scale_aware import quantize_scale_aware
from forge_llm.second_order import (
    CalibrationStats,
    block_moments,
    write_calibration_stats,
)
from forge_llm.validate_int8 import verify_derivation
from test_quantization import tiny_artifact
from test_second_order import fixture_stats


def scale_config(config):
    return {k: v for k, v in config.items() if k != "refinement_sweeps"}


def independent_losses(inputs, weight, q, scales, block):
    error = weight.astype(np.float64) - dequantize_per_channel(q, scales).astype(
        np.float64
    )
    return sum(
        np.square(inputs[:, s : s + block] @ error[:, s : s + block].T).mean(axis=0)
        for s in range(0, weight.shape[1], block)
    )


def test_refinement_improves_correlated_objective_with_identical_scale_bytes():
    random = np.random.default_rng(707)
    inputs = random.normal(size=(129, 13))
    inputs[:, 1] = inputs[:, 0] * 0.9 + 0.1 * inputs[:, 1]
    weights = random.normal(size=(17, 13)).astype(np.float16)
    config = {**DEFAULT_CONFIG, "block_size": 4, "row_chunk_size": 5}
    moments = block_moments(inputs, 4)
    q, s, metrics = quantize_refined(weights, moments, config)
    old_q, old_s, _ = quantize_scale_aware(weights, moments, scale_config(config))
    old = independent_losses(inputs, weights, old_q, old_s, 4)
    actual = independent_losses(inputs, weights, q, s, 4)
    assert s.tobytes() == old_s.tobytes()
    assert np.all(actual <= old + 1e-14) and actual.sum() < old.sum()
    assert metrics["scale_aware_block_objective"] == pytest.approx(old.sum())
    assert metrics["calibrated_block_objective"] == pytest.approx(actual.sum())
    assert metrics["refinement_changed_elements"] == np.count_nonzero(q != old_q) > 0
    assert metrics["no_worse_scale_aware_row_objective"]
    assert q.dtype == np.int8 and np.min(q) >= -127 and np.max(q) <= 127
    repeat_q, repeat_s, repeat_m = quantize_refined(weights, moments, config)
    assert q.tobytes() == repeat_q.tobytes() and s.tobytes() == repeat_s.tobytes()
    assert metrics == repeat_m


@pytest.mark.parametrize("dead", [False, True])
def test_dead_and_diagonal_covariance_do_not_invent_integer_improvements(dead):
    random = np.random.default_rng(21)
    weights = random.normal(size=(9, 13)).astype(np.float16)
    weights[0] = 0
    inputs = np.zeros((13, 13)) if dead else np.eye(13)
    config = {**DEFAULT_CONFIG, "block_size": 4, "row_chunk_size": 2}
    moments = block_moments(inputs, 4)
    q, s, metrics = quantize_refined(weights, moments, config)
    old_q, old_s, _ = quantize_scale_aware(weights, moments, scale_config(config))
    assert q.tobytes() == old_q.tobytes() and s.tobytes() == old_s.tobytes()
    assert metrics["refinement_changed_elements"] == 0
    assert s[0] == 1 and np.all(q[0] == 0)


@pytest.mark.parametrize("sweeps", [0, 5, True, 1.5, float("nan")])
def test_invalid_refinement_bounds_are_rejected(sweeps):
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "refinement_sweeps": sweeps})


def test_refinement_rejects_missing_or_extra_configuration_and_bad_source():
    config = {**DEFAULT_CONFIG, "block_size": 4}
    for changed in (
        {k: v for k, v in config.items() if k != "refinement_sweeps"},
        {**config, "unexpected": 1},
    ):
        with pytest.raises(ValueError):
            validate_config(changed)
    weight = np.ones((3, 5), dtype=np.float16)
    moments = block_moments(np.eye(5), 4)
    for bad in (weight.astype(np.float32), np.full_like(weight, np.nan)):
        with pytest.raises(ValueError):
            quantize_refined(bad, moments, config)
    with pytest.raises(ValueError):
        quantize_refined(weight, moments.astype(np.float32), config)


def test_refinement_row_bound_on_rank_deficient_and_saturated_fixtures():
    for seed in range(5):
        random = np.random.default_rng(seed + 900)
        inputs = random.normal(size=(23, 11))
        inputs[:, 3] = inputs[:, 2]  # singular covariance, handled by base damping
        inputs[:, 9] = 0
        weights = random.normal(size=(7, 11)).astype(np.float16)
        weights[0, 0] = np.float16(65504)
        weights[1] = 0
        config = {**DEFAULT_CONFIG, "block_size": 4, "row_chunk_size": 3}
        moments = block_moments(inputs, 4)
        q, s, metrics = quantize_refined(weights, moments, config)
        base_q, base_s, _ = quantize_scale_aware(weights, moments, scale_config(config))
        assert s.tobytes() == base_s.tobytes()
        assert np.isfinite(dequantize_per_channel(q, s)).all()
        old = independent_losses(inputs, weights, base_q, base_s, 4)
        actual = independent_losses(inputs, weights, q, s, 4)
        assert np.all(actual <= old + 1e-9)
        assert metrics["no_worse_scale_aware_row_objective"]


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_refined_recipe_archive_export_and_derivation_are_method_bound(
    tmp_path, family
):
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
        assert recipe.algorithm == recipe.method == ALGORITHM
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
            assert (
                packed.tensor_numpy(retained[0]).tobytes()
                == source.tensor_numpy(retained[0]).tobytes()
            )
        with pytest.raises(FileExistsError):
            quantize_model(
                source_path, output, policy=policy, calibration_stats=stats_path
            )
        for change in (
            {"algorithm": "scale_aware_cached_repair_v1"},
            {"calibration_stats_sha256": "0" * 64},
            {"search_config": {}},
        ):
            with pytest.raises(ValueError):
                quantize_model(
                    source_path,
                    tmp_path / "invalid.engine",
                    policy=seal_policy({**policy, **change}),
                    calibration_stats=stats_path,
                )


def test_new_corpora_are_disjoint_from_all_frozen_previous_corpora():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    cal = json.loads((root / "refined-calibration-prompts.json").read_text())
    reg = json.loads((root / "refined-regression-prompts.json").read_text())
    assert len(cal) == 32 and len(reg) == 25
    validate_corpora(cal, reg)
    for name in [
        "quality-prompts.json",
        "calibration-prompts.json",
        "second-order-calibration-prompts.json",
        "second-order-regression-prompts.json",
        "cached-calibration-prompts.json",
        "cached-regression-prompts.json",
        "scale-calibration-prompts.json",
        "scale-regression-prompts.json",
    ]:
        old = json.loads((root / name).read_text())
        validate_corpora(cal, old)
        validate_corpora(reg, old)
