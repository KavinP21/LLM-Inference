from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def auditor(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "refined_final_audit", root / "benchmarks/audit_refined_checkpoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_objective_readback_rejects_missing_rows_and_changed_evidence(monkeypatch):
    mod = auditor(monkeypatch)
    identity = {
        "shape": [8, 4],
        "changed_scale_rows": 5,
        "changed_quantized_elements": 15,
        "baseline_objective": 2.0,
        "refined_objective": 1.0,
        "calibration_report_sha256": "a" * 64,
        "stats_sha256": "b" * 64,
    }
    observation = {
        **identity,
        "family": "qwen",
        "weight": "layer.weight",
        "rows": 8,
        "maximum_row_loss_delta": 0.0,
        "positive_row_deltas_within_tolerance": 0,
        "row_bound_verified": True,
        "packed_sha256": "c" * 64,
        "scales_sha256": "d" * 64,
    }
    report = {"observations": [observation], "rows": 8, "matrices": 1, "passed": True}
    expected = {("qwen", "layer.weight"): identity}
    summary = mod.objective_contract(report, expected)
    assert summary["qwen"]["rows"] == 8
    assert summary["qwen"]["refined_objective"] == 1
    mutations = {
        "shape": [7, 4],
        "rows": 7,
        "changed_scale_rows": 4,
        "changed_quantized_elements": 14,
        "baseline_objective": float("nan"),
        "refined_objective": 3.0,
        "maximum_row_loss_delta": 1.0,
        "calibration_report_sha256": "e" * 64,
        "stats_sha256": "f" * 64,
        "row_bound_verified": 1,
        "packed_sha256": "not-a-checksum",
        "positive_row_deltas_within_tolerance": 9,
    }
    for field, value in mutations.items():
        changed = copy.deepcopy(report)
        changed["observations"][0][field] = value
        with pytest.raises(ValueError):
            mod.objective_contract(changed, expected)
    for mutation in ("duplicate", "missing", "total", "passed"):
        changed = copy.deepcopy(report)
        if mutation == "duplicate":
            changed["observations"].append(observation)
        elif mutation == "missing":
            changed["observations"] = []
        elif mutation == "total":
            changed["rows"] = 7
        else:
            changed["passed"] = False
        with pytest.raises(ValueError):
            mod.objective_contract(changed, expected)


def test_final_regression_requires_complete_clean_junit(tmp_path, monkeypatch):
    mod = auditor(monkeypatch)
    path = tmp_path / "tests.xml"
    path.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>'
    )
    assert mod.regression_contract(path, 2)["tests"] == 2
    with pytest.raises(ValueError):
        mod.regression_contract(path, 3)
    for field in ("failures", "errors", "skipped"):
        path.write_text(f'<testsuites><testsuite tests="2" {field}="1"/></testsuites>')
        with pytest.raises(ValueError):
            mod.regression_contract(path, 2)
