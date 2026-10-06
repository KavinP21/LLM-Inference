from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


def auditor(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "cached_final_audit", root / "benchmarks/audit_cached_checkpoint.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_byte_readback_recomputes_all_positions_and_rejects_tampering(monkeypatch):
    mod = auditor(monkeypatch)
    identity = {"dtype": "<f2", "shape": [100], "sha256": "a" * 64}
    report = {
        "observations": [
            {
                "case_index": i,
                "rows": [
                    {"position": p, "native": identity, "composed": identity}
                    for p in range(32)
                ],
                "byte_equal_rows": 32,
                "private_cache_reclaimed": True,
            }
            for i in range(25)
        ],
        "total_rows": 800,
        "byte_equal_rows": 800,
        "passed": True,
    }
    assert mod.byte_contract(report) == 800
    for mutation in ("bytes", "dtype", "shape", "position", "cleanup", "count"):
        changed = copy.deepcopy(report)
        case = changed["observations"][0]
        if mutation in ("bytes", "dtype", "shape"):
            case["rows"][0]["composed"] = dict(identity)
            field, value = {
                "bytes": ("sha256", "b" * 64),
                "dtype": ("dtype", "<f4"),
                "shape": ("shape", [101]),
            }[mutation]
            case["rows"][0]["composed"][field] = value
        elif mutation == "position":
            case["rows"][0]["position"] = 1
        elif mutation == "cleanup":
            case["private_cache_reclaimed"] = False
        else:
            changed["byte_equal_rows"] = 799
        with pytest.raises(ValueError):
            mod.byte_contract(changed)


def test_final_test_readback_requires_complete_clean_suite(tmp_path, monkeypatch):
    mod = auditor(monkeypatch)
    path = tmp_path / "tests.xml"
    path.write_text(
        '<testsuites><testsuite tests="2" failures="0" errors="0" skipped="0"/></testsuites>'
    )
    assert mod.test_contract(path, 2)["tests"] == 2
    with pytest.raises(ValueError):
        mod.test_contract(path, 3)
    path.write_text('<testsuites><testsuite tests="2" skipped="1"/></testsuites>')
    with pytest.raises(ValueError):
        mod.test_contract(path, 2)
