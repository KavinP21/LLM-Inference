from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from forge_llm.calibrate_second_order import (
    capture_inputs,
    joint_score,
    projection_groups,
)
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import seal_policy, validate_corpora, validate_policy
from forge_llm.quantization import (
    dequantize_per_channel,
    quantize_model,
    quantize_per_channel,
)
from forge_llm.second_order import (
    ALGORITHM,
    DEFAULT_CONFIG,
    CalibrationStats,
    block_moments,
    quantize_second_order,
    validate_config,
    write_calibration_stats,
)
from forge_llm.validate_int8 import verify_derivation
from test_quantization import tiny_artifact


def fixture_stats(source, path):
    groups = projection_groups(source.config.num_hidden_layers)
    mapping = {name: key for key, names in groups.items() for name in names}
    random = np.random.default_rng(31)
    arrays = {}
    for key, names in groups.items():
        width = source.tensor_info(names[0]).shape[1]
        values = random.normal(size=(40, width))
        values[:, 1:] += values[:, :1]
        arrays[key] = block_moments(values, 4)
    metadata = {
        "schema_version": 1,
        "algorithm": ALGORITHM,
        "source_data_sha256": source.data_sha256,
        "quantizer_config": {**DEFAULT_CONFIG, "block_size": 4},
        "weight_to_moments": mapping,
    }
    checksum = write_calibration_stats(path, metadata, arrays)
    return metadata, arrays, checksum


def test_correlated_error_compensation_improves_real_fp16_block_objective():
    random = np.random.default_rng(91)
    x = random.normal(size=(100, 19))
    x[:, 1:] = x[:, :1] + 0.05 * x[:, 1:]
    weight = random.normal(size=(31, 19)).astype(np.float16)
    config = {**DEFAULT_CONFIG, "block_size": 8}
    moments = block_moments(x, 8)
    q, scales, observations = quantize_second_order(weight, moments, config)
    rtn, rtn_scales = quantize_per_channel(weight)
    np.testing.assert_array_equal(scales, rtn_scales)
    assert observations["changed_quantized_elements"] > 0
    assert (
        observations["calibrated_block_objective"]
        < observations["rtn_block_objective"] * 0.5
    )
    losses = []
    for packed in [rtn, q]:
        error = weight.astype(np.float64) - dequantize_per_channel(
            packed, scales
        ).astype(np.float64)
        # Independent X @ error test per block, not the quantizer's quadratic code.
        loss = sum(
            np.square(x[:, i : i + 8] @ error[:, i : i + 8].T).sum() / len(x)
            for i in range(0, 19, 8)
        )
        losses.append(loss)
    assert losses == pytest.approx(
        [
            observations["rtn_block_objective"],
            observations["calibrated_block_objective"],
        ]
    )
    np.testing.assert_array_equal(q, quantize_second_order(weight, moments, config)[0])
    assert q.min() >= -127 and q.max() <= 127


def test_diagonal_and_dead_covariance_match_rtn():
    weight = np.random.default_rng(9).normal(size=(7, 9)).astype(np.float16)
    weight[0] = 0
    for x in [np.eye(9), np.zeros((9, 9))]:
        q, scales, _ = quantize_second_order(
            weight, block_moments(x, 4), {**DEFAULT_CONFIG, "block_size": 4}
        )
        expected_q, expected_scales = quantize_per_channel(weight)
        np.testing.assert_array_equal(q, expected_q)
        np.testing.assert_array_equal(scales, expected_scales)


@pytest.mark.parametrize(
    "change",
    [
        {"block_size": True},
        {"block_size": 0},
        {"block_size": 257},
        {"damping": 0},
        {"damping": float("nan")},
        {"activation_order": "yes"},
    ],
)
def test_invalid_configuration(change):
    with pytest.raises(ValueError, match="configuration"):
        validate_config({**DEFAULT_CONFIG, **change})


def test_invalid_moments_fail_closed():
    w = np.ones((2, 5), dtype=np.float16)
    valid = block_moments(np.eye(5), 4)
    variants = [
        valid.astype(np.float32),
        valid[:1],
        valid.copy(),
        valid.copy(),
        valid.copy(),
        valid.copy(),
    ]
    variants[2][0, 0, 1] = 1
    variants[3][0, 0, 0] = -1
    variants[4][-1, -1, -1] = 1
    variants[5][0, 0, 0] = np.nan
    for moments in variants:
        with pytest.raises(ValueError):
            quantize_second_order(w, moments, {**DEFAULT_CONFIG, "block_size": 4})


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_frozen_recipe_export_derivation_and_unchanged_rtn(tmp_path, family):
    source_path = tmp_path / "source.engine"
    tiny_artifact(source_path, family)
    stats_path = tmp_path / "stats.npz"
    with ModelFile(source_path) as source:
        metadata, _, checksum = fixture_stats(source, stats_path)
        recipe = CalibrationStats(stats_path, source, expected_sha=checksum)
        names = set(metadata["weight_to_moments"])
        retained = [min(names)]
        policy = seal_policy(
            {
                "schema_version": 1,
                "algorithm": ALGORITHM,
                "source_data_sha256": source.data_sha256,
                "quantizer_config": metadata["quantizer_config"],
                "calibration_stats_sha256": checksum,
                "retained_fp16": retained,
                "quantized_weights": sorted(names - set(retained)),
                "calibration_passed": False,
            }
        )
        assert validate_policy(policy, source.data_sha256, names) == retained
        with pytest.raises(ValueError, match="requires frozen"):
            quantize_model(source_path, tmp_path / "absent.engine", policy=policy)
        with pytest.raises(ValueError, match="checksum"):
            CalibrationStats(stats_path, source, expected_sha="0" * 64)
        with pytest.raises(ValueError, match="checksum"):
            validate_policy(
                seal_policy({**policy, "calibration_stats_sha256": "xyz"}),
                source.data_sha256,
                names,
            )
        output, second = tmp_path / "quantized.engine", tmp_path / "second.engine"
        result = quantize_model(
            source_path, output, policy=policy, calibration_stats=stats_path
        )
        quantize_model(source_path, second, policy=policy, calibration_stats=stats_path)
        assert output.read_bytes() == second.read_bytes()
        assert result["quantization_method"] == "block_second_order_v1"
        assert result["calibration_stats_sha256"] == recipe.sha256
        with ModelFile(output) as packed:
            proof = verify_derivation(source, packed, stats_path)
            assert proof["all_packed_values_and_scales_recomputed"]
            with pytest.raises(ValueError, match="does not derive"):
                verify_derivation(source, packed)
            np.testing.assert_array_equal(
                source.tensor_numpy(retained[0]), packed.tensor_numpy(retained[0])
            )
        rtn_path = tmp_path / "rtn.engine"
        quantize_model(source_path, rtn_path, retain_fp16=retained)
        with ModelFile(rtn_path) as packed:
            assert verify_derivation(source, packed)["method"] == "rtn_v1"
        with pytest.raises(ValueError, match="require a second-order"):
            quantize_model(
                source_path, tmp_path / "invalid.engine", calibration_stats=stats_path
            )


def test_archive_corruption_identity_padding_and_no_clobber(tmp_path):
    source_path = tmp_path / "source.engine"
    tiny_artifact(source_path)
    stats_path = tmp_path / "stats.npz"
    with ModelFile(source_path) as source:
        metadata, arrays, _ = fixture_stats(source, stats_path)
        with pytest.raises(FileExistsError):
            write_calibration_stats(stats_path, metadata, arrays)
        for index, mutation in enumerate(
            [
                {"source_data_sha256": "wrong"},
                {"weight_to_moments": {}},
                {"quantizer_config": {**DEFAULT_CONFIG, "block_size": 0}},
            ]
        ):
            bad = tmp_path / f"bad-{index}.npz"
            write_calibration_stats(bad, {**metadata, **mutation}, arrays)
            with pytest.raises(ValueError):
                CalibrationStats(bad, source)
        key = next(iter(arrays))
        for index, value in enumerate(
            [
                np.zeros((1,)),
                arrays[key].astype(np.float32),
                np.full_like(arrays[key], np.nan),
            ]
        ):
            bad = tmp_path / f"bad-array-{index}.npz"
            write_calibration_stats(bad, metadata, {**arrays, key: value})
            with pytest.raises(ValueError):
                CalibrationStats(bad, source)
        oversized = tmp_path / "oversized.npz"
        write_calibration_stats(
            oversized, metadata, {**arrays, key: np.zeros((1000, 1000))}
        )
        with pytest.raises(ValueError, match="oversized"):
            CalibrationStats(oversized, source)


def test_joint_objective_scores_combined_interactions_not_single_losses():
    assert joint_score(
        [{"argmax_equal": True, "relative_l2_error": 0.2}], 100
    ) < joint_score([{"argmax_equal": False, "relative_l2_error": 0.01}], 1000)
    assert joint_score(
        [{"argmax_equal": True, "relative_l2_error": 0.2}], 100
    ) > joint_score([{"argmax_equal": True, "relative_l2_error": 0.2}], 1000)
    with pytest.raises(ValueError):
        joint_score([], 0)


def test_activation_capture_restores_method_and_samples_shared_group_once():
    model = SimpleNamespace(
        mx=SimpleNamespace(array=np.array), _linear=lambda x, name: x
    )
    previous = model._linear
    observed = []
    x = np.arange(20).reshape(5, 4)
    with (
        pytest.raises(RuntimeError, match="unchunked"),
        capture_inputs(
            model,
            {"cov_0": ["q", "k"]},
            3,
            lambda key, rows: observed.append((key, rows)),
        ),
    ):
        np.testing.assert_array_equal(model._linear(x, "q"), x)
        model._linear(x, "k")
        model._linear(x, "q")
    assert model._linear is previous
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0][1], x[[0, 2, 4]])


def test_new_corpora_are_disjoint_and_frozen_sizes():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    calibration = json.loads(
        (root / "second-order-calibration-prompts.json").read_text()
    )
    old = json.loads((root / "quality-prompts.json").read_text())
    fresh = json.loads((root / "second-order-regression-prompts.json").read_text())
    assert len(calibration) == 32 and len(fresh) == 25 and len(old) == 25
    validate_corpora(calibration, old + fresh)
