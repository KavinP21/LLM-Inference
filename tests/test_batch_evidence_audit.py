from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("failure", ["checksum", "escape", "stale"])
def test_evidence_auditor_fails_closed(tmp_path, monkeypatch, failure):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import audit_batch_checkpoint

    root = tmp_path / "results"
    for folder in ("regression", "resources-and-matrix"):
        directory = root / folder
        directory.mkdir(parents=True)
        content = json.dumps(
            {"environment": {"runtime_source_sha256": "original"}}
        ).encode()
        (directory / "evidence.json").write_bytes(content)
        record = {
            "path": "evidence.json",
            "sha256": hashlib.sha256(content).hexdigest(),
            "runtime_source_sha256": "original",
            "gate": None,
            "passed": True,
        }
        if failure == "checksum":
            record["sha256"] = "corrupted"
        elif failure == "escape":
            record["path"] = "../../outside.json"
        (directory / "checkpoint.json").write_text(
            json.dumps(
                {
                    "all_requested_gates_passed": True,
                    "runtime_source_sha256": "original",
                    "records": [record],
                }
            )
        )
    monkeypatch.setattr(
        audit_batch_checkpoint,
        "source_provenance",
        lambda: {"runtime_source_sha256": "changed"},
    )
    with pytest.raises(
        ValueError,
        match={
            "checksum": "checksum mismatch",
            "escape": "escapes",
            "stale": "current runtime changed",
        }[failure],
    ):
        audit_batch_checkpoint.audit(root)
