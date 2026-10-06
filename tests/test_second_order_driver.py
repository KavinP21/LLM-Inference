from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "second_order_driver", root / "benchmarks/run_second_order_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_saved_failed_gate_is_not_an_execution_error(tmp_path, monkeypatch):
    driver = module(monkeypatch)
    path = tmp_path / "failed.json"
    driver.save(path, {"gates": {"passed": False}})
    report = driver.checked_report(
        lambda *a, **k: SimpleNamespace(returncode=1), [], path, ("gates", "passed")
    )
    assert not report["gates"]["passed"]
    for code in [0, 2, -6]:
        with pytest.raises(RuntimeError):
            driver.checked_report(
                lambda *a, code=code, **k: SimpleNamespace(returncode=code),
                [],
                path,
                ("gates", "passed"),
            )
    with pytest.raises(RuntimeError):
        driver.checked_report(
            lambda *a, **k: SimpleNamespace(returncode=1),
            [],
            tmp_path / "absent.json",
            ("gates", "passed"),
        )
    with pytest.raises(FileExistsError):
        driver.save(path, {})


def test_freeze_rejects_runtime_driver_and_corpus_drift(tmp_path, monkeypatch):
    driver = module(monkeypatch)
    corpus = tmp_path / "corpus.json"
    driver.save(corpus, ["held out"])
    frozen = {
        "environment": {"runtime_source_sha256": "original"},
        "driver_sha256": driver.file_sha256(Path(driver.__file__)),
        "corpora_sha256": {str(corpus): driver.file_sha256(corpus)},
        "artifacts": {},
    }
    driver.save(tmp_path / "frozen.json", frozen)
    monkeypatch.setattr(
        driver, "source_provenance", lambda: {"runtime_source_sha256": "new"}
    )
    with pytest.raises(RuntimeError, match="runtime changed"):
        driver.load_frozen(tmp_path)
    monkeypatch.setattr(
        driver, "source_provenance", lambda: {"runtime_source_sha256": "original"}
    )
    assert driver.load_frozen(tmp_path) == frozen
    monkeypatch.setattr(
        driver,
        "file_sha256",
        lambda p: "new" if Path(p) == corpus else frozen["driver_sha256"],
    )
    with pytest.raises(RuntimeError, match="corpus changed"):
        driver.load_frozen(tmp_path)


def test_incomplete_evidence_cannot_be_audited(tmp_path, monkeypatch):
    driver = module(monkeypatch)
    monkeypatch.setattr(driver, "load_frozen", lambda p: {})
    (tmp_path / "regression").mkdir()
    driver.save(
        tmp_path / "regression/checkpoint.json", {"strict_checkpoint_passed": False}
    )
    with pytest.raises(ValueError, match="incomplete evidence"):
        driver.audit(tmp_path)


def test_kernel_diagnostic_selects_common_prefix_and_rejects_bad_lengths(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    from trace_quantized_divergence import divergences

    report = {
        "cases": [
            {"int8_tokens": [1, 2, 3], "dequantize_tokens": [1, 2, 4]},
            {"int8_tokens": [3, 2, 1], "dequantize_tokens": [4, 2, 1]},
            {"int8_tokens": [1, 2, 3], "dequantize_tokens": [1, 2, 3]},
        ]
    }
    assert divergences(report) == [(0, 2), (1, 0)]
    with pytest.raises(ValueError, match="malformed"):
        divergences({"cases": [{"int8_tokens": [], "dequantize_tokens": [1]}]})


def test_whole_file_snapshot_binds_metadata_not_only_data_identity(
    tmp_path, monkeypatch
):
    driver = module(monkeypatch)
    import check_second_order_integrity as integrity

    paths = {
        name: tmp_path / filename
        for name, filename in {
            "source": "source.engine",
            "model": "model.engine",
            "rtn_control": "rtn.engine",
            "stats": "stats.npz",
            "calibration_report": "calibration.json",
        }.items()
    }
    for name, path in paths.items():
        driver.save(path, {"payload": name})
        if name in {"source", "model", "rtn_control"}:
            driver.save(path.with_suffix(path.suffix + ".json"), {})
    driver.save(
        tmp_path / "frozen.json",
        {
            "environment": {"runtime_source_sha256": "same"},
            "artifacts": {"test": {n: str(p) for n, p in paths.items()}},
        },
    )
    monkeypatch.setattr(
        integrity, "source_provenance", lambda: {"runtime_source_sha256": "same"}
    )
    before = integrity.snapshot(tmp_path)
    # Deliberately mutate a synthetic file in this test fixture only.
    with paths["model"].open("a") as handle:
        handle.write("metadata modification")
    after = integrity.snapshot(tmp_path)
    assert before["files"] != after["files"]
    assert (
        before["files"][str(paths["model"])]["sha256"]
        != after["files"][str(paths["model"])]["sha256"]
    )
    monkeypatch.setattr(
        integrity, "source_provenance", lambda: {"runtime_source_sha256": "changed"}
    )
    with pytest.raises(ValueError, match="runtime drift"):
        integrity.snapshot(tmp_path)
