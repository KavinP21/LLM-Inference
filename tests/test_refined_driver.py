from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
from forge_llm.precision_policy import seal_policy
from forge_llm.refined import DEFAULT_CONFIG, DEFAULT_SEARCH, POLICY_ALGORITHM


def driver(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "refined_driver", root / "benchmarks/run_refined_checkpoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fixture_report():
    return {
        "configuration": {"positions": list(range(32))},
        "source_model": {"model_data_sha256": "a" * 64},
        "cases": [
            {
                "fp16_tokens": [1] * 32,
                "candidate_tokens": [1] * 32,
                "teacher_forced_tokens": [1] * 32,
                "exact_greedy_parity": True,
                "private_cache_reclaimed": True,
                "cached_teacher_forced_logits": [
                    {"generated_position": p, "argmax_equal": True, "cosine": 1.0}
                    for p in range(32)
                ],
            }
            for i in range(32)
        ],
        "local_objectives": [
            {
                "weight": n,
                "fp16_bytes": b,
                "rtn_block_objective": 2.0,
                "fixed_scale_block_objective": 1.0,
                "scale_aware_block_objective": 0.95,
                "calibrated_block_objective": 0.9,
                "no_worse_scale_aware_row_objective": True,
                "no_worse_fixed_scale_row_objective": True,
            }
            for n, b in [("q", 4), ("r", 12)]
        ],
        "summary": {
            "exact_greedy_cases": 32,
            "total_cases": 32,
            "minimum_logit_cosine": 1.0,
            "cached_top1_changes": 0,
            "calibration_passed": True,
        },
        "search": {
            "summary": {
                "top1_changes": 0,
                "observations": 1024,
                "minimum_logit_cosine": 1.0,
            }
        },
        "policy": seal_policy(
            {
                "schema_version": 1,
                "algorithm": POLICY_ALGORITHM,
                "source_data_sha256": "a" * 64,
                "calibration_stats_sha256": "b" * 64,
                "quantizer_config": DEFAULT_CONFIG,
                "search_config": DEFAULT_SEARCH,
                "retained_fp16": ["r"],
                "quantized_weights": ["q"],
                "quantized_projection_fraction": 0.25,
                "calibration_passed": True,
            }
        ),
        "cache_reclaimed": True,
    }


def test_registered_corpora_are_the_fresh_refined_set(monkeypatch):
    from forge_llm.precision_policy import validate_corpora

    mod = driver(monkeypatch)
    assert mod.CALIBRATION.name == "refined-calibration-prompts.json"
    assert mod.REGRESSION.name == "refined-regression-prompts.json"
    calibration = json.loads(mod.CALIBRATION.read_text())
    guards = [p for path in mod.GUARDS for p in json.loads(path.read_text())]
    assert len(calibration) == 32 and len(guards) == 237
    validate_corpora(calibration, guards)


def test_refinement_gate_cannot_certify_a_weaker_bound(monkeypatch):
    mod = driver(monkeypatch)
    for field, value in [
        ("no_worse_scale_aware_row_objective", 1),
        ("scale_aware_block_objective", 0.8),
    ]:
        report = fixture_report()
        report["local_objectives"][0][field] = value
        with pytest.raises(ValueError):
            mod.calibration_gates(report)


def test_calibration_quality_failure_is_a_saved_rejection_not_a_weaker_gate(
    monkeypatch,
):
    mod = driver(monkeypatch)
    report = fixture_report()
    assert mod.calibration_gates(report)["passed"]
    report["cases"][0]["candidate_tokens"][0] = 2
    report["cases"][0]["exact_greedy_parity"] = False
    report["summary"]["exact_greedy_cases"] = 31
    report["summary"]["calibration_passed"] = False
    report["policy"] = seal_policy({**report["policy"], "calibration_passed": False})
    gates = mod.calibration_gates(report)
    assert not gates["passed"] and not gates["exact_free_continuations"]
    assert gates["numerical"] and gates["minimum_projection_fraction"]


@pytest.mark.parametrize(
    "mutation",
    [
        "tokens",
        "cosine",
        "positions",
        "objective",
        "bool",
        "changes",
        "fraction",
        "partition",
    ],
)
def test_gate_readback_rejects_inconsistent_or_incomplete_observations(
    monkeypatch, mutation
):
    mod = driver(monkeypatch)
    report = fixture_report()
    if mutation == "tokens":
        report["cases"][0]["candidate_tokens"][0] = 2
    elif mutation == "cosine":
        report["cases"][0]["cached_teacher_forced_logits"][0]["cosine"] = float("nan")
    elif mutation == "positions":
        report["cases"][0]["cached_teacher_forced_logits"].pop()
    elif mutation == "objective":
        report["local_objectives"][0]["calibrated_block_objective"] = 2.0
    elif mutation == "bool":
        report["cache_reclaimed"] = "true"
    elif mutation == "changes":
        report["search"]["summary"]["top1_changes"] = 1
    elif mutation == "fraction":
        report["policy"] = seal_policy(
            {**report["policy"], "quantized_projection_fraction": 0.5}
        )
    else:
        report["policy"] = seal_policy({**report["policy"], "retained_fp16": []})
    with pytest.raises(ValueError):
        mod.calibration_gates(report)


def test_preregistration_rejects_runtime_file_and_prior_evidence_drift(
    tmp_path, monkeypatch
):
    mod = driver(monkeypatch)
    current, previous = tmp_path / "input", tmp_path / "old"
    current.write_text("fixed method")
    previous.write_text("preserved evidence")
    contract = {
        "environment": {"runtime_source_sha256": "current"},
        "files_sha256": {str(current): mod.file_sha256(current)},
        "prior_files_sha256": {str(previous): mod.file_sha256(previous)},
        "configuration": {
            "quantizer_config": DEFAULT_CONFIG,
            "search_config": DEFAULT_SEARCH,
        },
    }
    archive = tmp_path / "sources.zip"
    with zipfile.ZipFile(archive, "x") as bundle:
        bundle.writestr("input.py", b"fixed method")
    contract["source_archive"] = {
        "sha256": mod.file_sha256(archive),
        "files_sha256": {"input.py": hashlib.sha256(b"fixed method").hexdigest()},
    }
    mod.save(tmp_path / "contract.json", contract)
    monkeypatch.setattr(
        mod, "source_provenance", lambda: {"runtime_source_sha256": "current"}
    )
    assert mod.load_contract(tmp_path) == contract
    for path in (current, previous):
        original = path.read_text()
        path.write_text("drift")
        with pytest.raises(ValueError, match="file changed"):
            mod.load_contract(tmp_path)
        path.write_text(original)
    monkeypatch.setattr(
        mod, "source_provenance", lambda: {"runtime_source_sha256": "different"}
    )
    with pytest.raises(ValueError, match="runtime drift"):
        mod.load_contract(tmp_path)


def test_source_capture_contains_the_actual_dirty_runtime_and_detects_archive_drift(
    tmp_path, monkeypatch
):
    mod = driver(monkeypatch)
    captured = mod.capture_sources(tmp_path)
    assert "python/forge_llm/refined.py" in captured["files_sha256"]
    assert "benchmarks/run_refined_checkpoint.py" in captured["files_sha256"]
    mod.check_sources(tmp_path, captured)
    with pytest.raises(FileExistsError):
        mod.capture_sources(tmp_path)
    with zipfile.ZipFile(tmp_path / "sources.zip") as bundle:
        data = bundle.read("python/forge_llm/refined.py")
    assert (
        data
        == (
            Path(__file__).resolve().parents[1] / "python/forge_llm/refined.py"
        ).read_bytes()
    )
    wrong = copy.deepcopy(captured)
    wrong["files_sha256"]["python/forge_llm/refined.py"] = "0" * 64
    with pytest.raises(ValueError, match="member changed"):
        mod.check_sources(tmp_path, wrong)
    (tmp_path / "sources.zip").write_bytes(b"modified archive")
    with pytest.raises(ValueError, match="archive changed"):
        mod.check_sources(tmp_path, captured)


def test_partial_calibration_is_preserved_and_cannot_be_silently_rerun(
    tmp_path, monkeypatch
):
    mod = driver(monkeypatch)
    stats = tmp_path / "existing.npz"
    stats.write_bytes(b"partial archive")
    contract = {"families": {"qwen": {"stats": str(stats)}}}
    monkeypatch.setattr(mod, "load_contract", lambda _: contract)
    calls = []
    with pytest.raises(FileExistsError, match="preserved"):
        mod.calibrate_phase(lambda *a, **k: calls.append(a), tmp_path)
    assert calls == [] and stats.read_bytes() == b"partial archive"


def test_failed_calibration_stops_before_any_export_or_held_out_job(
    tmp_path, monkeypatch
):
    mod = driver(monkeypatch)
    contract = {
        "environment": {"runtime_source_sha256": "current"},
        "families": {
            family: {
                "stats": str(tmp_path / f"{family}.npz"),
                "source": "source",
                "tokenizer": "offline",
            }
            for family in ("qwen", "gemma")
        },
    }
    mod.save(tmp_path / "contract.json", contract)
    monkeypatch.setattr(mod, "load_contract", lambda _: contract)
    report = fixture_report()
    report["cases"][0]["candidate_tokens"][0] = 2
    report["cases"][0]["exact_greedy_parity"] = False
    report["summary"].update(exact_greedy_cases=31, calibration_passed=False)
    report["policy"] = seal_policy({**report["policy"], "calibration_passed": False})
    report["environment"] = contract["environment"]
    commands = []

    def run(command, output, *args):
        commands.append(command)
        mod.save(output, report)
        stats_path = Path(command[command.index("--stats") + 1])
        stats_path.write_bytes(b"frozen stats")
        return copy.deepcopy(report)

    monkeypatch.setattr(
        mod,
        "checked_report",
        lambda _, command, output, *args: run(command, output, *args),
    )
    mod.calibrate_phase(None, tmp_path)
    checkpoint = json.loads((tmp_path / "calibration-checkpoint.json").read_text())
    assert len(commands) == 2 and all("--weight-method" in c for c in commands)
    assert checkpoint["state"] == "rejected_at_calibration"
    assert not checkpoint["held_out_inference_executed"]
    assert not checkpoint["official_packed_exports_created"]
    assert not checkpoint["performance_matrix_executed"]
    assert not checkpoint["strict_checkpoint_passed"]
