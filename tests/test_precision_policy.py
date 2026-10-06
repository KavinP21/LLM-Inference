from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from forge_llm.calibrate_int8 import replaced_weights
from forge_llm.model_file import ModelFile
from forge_llm.precision_policy import (
    canonical_sha256,
    rank_sensitivities,
    retained_for_fraction,
    seal_policy,
    validate_corpora,
    validate_policy,
)
from forge_llm.quantization import quantize_model
from test_quantization import tiny_artifact


def policy(source_sha, names, retained):
    return seal_policy(
        {
            "schema_version": 1,
            "algorithm": "single_projection_logit_ablation_v1",
            "source_data_sha256": source_sha,
            "retained_fp16": sorted(retained),
            "quantized_weights": sorted(set(names) - set(retained)),
            "calibration_passed": False,
        }
    )


@pytest.mark.parametrize(
    "calibration,held",
    [
        (["  Hello   WORLD "], ["hello world"]),
        (["Ｈｅｌｌｏ"], ["hello"]),
        (["x", " X "], ["different"]),
        ([], ["y"]),
        ([""], ["y"]),
        ([3], ["y"]),
    ],
)
def test_corpus_leakage_is_rejected(calibration, held):
    with pytest.raises(ValueError):
        validate_corpora(calibration, held)


def test_frozen_corpora_are_disjoint():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    validate_corpora(
        json.loads((root / "benchmarks/calibration-prompts.json").read_text()),
        json.loads((root / "benchmarks/quality-prompts.json").read_text()),
    )


def test_sensitivity_ranking_is_deterministic_and_storage_aware():
    observations = [
        {
            "weight": "b",
            "fp16_bytes": 10,
            "top1_changes": 0,
            "mean_squared_relative_logit_error": 0.1,
        },
        {
            "weight": "a",
            "fp16_bytes": 100,
            "top1_changes": 1,
            "mean_squared_relative_logit_error": 0.01,
        },
        {
            "weight": "c",
            "fp16_bytes": 100,
            "top1_changes": 0,
            "mean_squared_relative_logit_error": 0.1,
        },
    ]
    ranked = rank_sensitivities(observations)
    assert [r["weight"] for r in ranked] == ["a", "b", "c"]
    assert rank_sensitivities(list(reversed(observations))) == ranked
    for fraction in [1, 0.75, 0.5, 0.25]:
        retained = set(retained_for_fraction(ranked, fraction))
        assert (
            sum(r["fp16_bytes"] for r in ranked if r["weight"] not in retained)
            >= 210 * fraction
        )
    assert retained_for_fraction(ranked, 1) == []
    with pytest.raises(ValueError):
        retained_for_fraction(ranked, 0)
    with pytest.raises(ValueError):
        rank_sensitivities(observations + observations)
    with pytest.raises(ValueError):
        rank_sensitivities(
            [{**observations[0], "mean_squared_relative_logit_error": float("nan")}]
        )


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_source_bound_mixed_precision_export(tmp_path, family):
    source, output = tmp_path / "source.engine", tmp_path / "mixed.engine"
    tiny_artifact(source, family)
    retained = ["model.layers.0.mlp.down_proj.weight"]
    with ModelFile(source) as file:
        names = {name for name in file.tensors if name.endswith("_proj.weight")}
        selected = policy(file.data_sha256, names, retained)
    result = quantize_model(source, output, policy=selected)
    assert result["quantized_matrices"] == 13
    assert 0 < result["quantized_projection_fraction"] < 1
    assert result["retained_fp16_projections"] == retained
    assert result["precision_policy_sha256"] == canonical_sha256(selected)
    with ModelFile(source) as original, ModelFile(output) as packed:
        assert packed.tensor_info(retained[0]).dtype == np.float16
        np.testing.assert_array_equal(
            packed.tensor_numpy(retained[0]), original.tensor_numpy(retained[0])
        )
        assert all(
            packed.tensor_info(name).dtype == np.int8 for name in packed.quantization
        )
    with pytest.raises(ValueError, match="different source"):
        validate_policy(selected, "wrong", names)
    corrupted = {**selected, "retained_fp16": []}
    with pytest.raises(ValueError, match="checksum"):
        validate_policy(corrupted, selected["source_data_sha256"], names)
    for change in [
        {"retained_fp16": retained * 2},
        {"retained_fp16": ["lm_head.weight"]},
        {"quantized_weights": []},
        {"schema_version": 2},
    ]:
        with pytest.raises(ValueError):
            validate_policy(
                seal_policy({**selected, **change}),
                selected["source_data_sha256"],
                names,
            )
    with pytest.raises(ValueError, match="either"):
        quantize_model(
            source, tmp_path / "bad.engine", policy=selected, retain_fp16=retained
        )
    for invalid in [retained * 2, ["lm_head.weight"], sorted(names)]:
        with pytest.raises(ValueError):
            quantize_model(source, tmp_path / "bad.engine", retain_fp16=invalid)
    assert not (tmp_path / "bad.engine").exists()
    with ModelFile(source) as file:
        smallest = min(names, key=lambda n: file.tensor_info(n).nbytes)
    below_floor = policy(selected["source_data_sha256"], names, names - {smallest})
    with pytest.raises(ValueError, match="25%"):
        quantize_model(source, tmp_path / "below-floor.engine", policy=below_floor)
    assert not (tmp_path / "below-floor.engine").exists()


def test_offline_perturbation_restores_weights_after_failure():
    original = object()
    model = SimpleNamespace(weights={"weight": original})
    with pytest.raises(RuntimeError), replaced_weights(model, {"weight": object()}):
        assert model.weights["weight"] is not original
        raise RuntimeError("simulated execution failure")
    assert model.weights["weight"] is original
