from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from forge_llm.runtime import SequenceState


def load_driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import check_int8_batch

    return check_int8_batch


def test_batch_driver_checks_identity_before_loading_gpu(tmp_path, monkeypatch):
    driver = load_driver(monkeypatch)
    report = tmp_path / "quality.json"
    report.write_text(
        json.dumps({"quantized_model": {"model_data_sha256": "expected"}})
    )
    monkeypatch.setattr(
        driver, "model_provenance", lambda _: {"model_data_sha256": "different"}
    )
    with pytest.raises(ValueError, match="different artifact"):
        driver.check(tmp_path / "model.engine", report, "metal")


def test_batch_driver_arrival_cancellation_and_reference_selection(
    tmp_path, monkeypatch
):
    driver = load_driver(monkeypatch)
    cases = [
        {"input_ids": [i], "int8_tokens": [i] * 32, "fp16_tokens": [i] * 32}
        for i in range(1, 7)
    ]
    report = tmp_path / "quality.json"
    report.write_text(
        json.dumps(
            {
                "quantized_model": {"model_data_sha256": "same"},
                "source_model": {"model_data_sha256": "same"},
                "configuration": {"int8_mode": "metal"},
                "environment": {"runtime_source_sha256": "runtime"},
                "cases": cases,
            }
        )
    )
    monkeypatch.setattr(
        driver, "model_provenance", lambda _: {"model_data_sha256": "same"}
    )
    monkeypatch.setattr(
        driver, "source_provenance", lambda: {"runtime_source_sha256": "runtime"}
    )

    class Engine:
        def __init__(self, *_, **__):
            self.requests = {}
            self.scheduler = SimpleNamespace(request=lambda r: self.requests[r])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def submit(self, ids, count, **_):
            r = len(self.requests) + 1
            self.requests[r] = SimpleNamespace(
                token=ids[0], count=count, output=[], state=SequenceState.RUNNING
            )
            return r

        def step(self):
            events = []
            for r, request in self.requests.items():
                if request.state is SequenceState.RUNNING:
                    request.output.append(request.token)
                    if len(request.output) == request.count:
                        request.state = SequenceState.COMPLETED
                    events.append(SimpleNamespace(request_id=r))
            return events

        def cancel(self, r):
            self.requests[r].state = SequenceState.CANCELLED

        def stats(self):
            live = sum(r.state is SequenceState.RUNNING for r in self.requests.values())
            return {
                "kv_cache": {"allocated_blocks": live, "reserved_blocks": live},
                "kv_device_bytes": live * 64,
            }

        def build_info(self):
            return {"backend": "fake-host-only-test"}

    monkeypatch.setattr(driver, "MlxEngine", Engine)
    for fp16_source in [False, True]:
        result = driver.check(
            tmp_path / "model.engine", report, "metal", fp16_source=fp16_source
        )
        assert result["summary"]["passed"]
        assert result["summary"]["exact_cases"] == 6
        assert result["cancellation"]["blocks_before"] == 1
        assert result["cancellation"]["reclaimed"]
    with pytest.raises(ValueError, match="execution mode"):
        driver.check(tmp_path / "model.engine", report, "reconstruct")
