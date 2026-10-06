from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from forge_llm.runtime import SequenceState


def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    import validate_batch_numerics

    return validate_batch_numerics


def test_fingerprint_and_identity_guard_before_gpu(tmp_path, monkeypatch):
    module = driver(monkeypatch)
    a = module.fingerprint(np.array([1, 2, 3], dtype=np.float16))
    assert a == module.fingerprint(np.array([1, 2, 3], dtype=np.float32))
    assert a["argmax"] == 2
    for invalid in [np.array([np.nan, 1]), np.ones((2, 3))]:
        with pytest.raises(ValueError, match="logits"):
            module.fingerprint(invalid)
    report = tmp_path / "historical.json"
    report.write_text(json.dumps({"source_model": {"model_data_sha256": "original"}}))
    monkeypatch.setattr(
        module, "model_provenance", lambda _: {"model_data_sha256": "other"}
    )
    with pytest.raises(ValueError, match="different artifact"):
        module.validate(tmp_path / "model.engine", report, "auto")


@pytest.mark.parametrize("staggered", [False, True])
def test_capacity_aware_arrivals_never_drop_cases_or_leak(monkeypatch, staggered):
    module = driver(monkeypatch)
    cases = [{"input_ids": [i]} for i in range(6)]
    references = [
        {"tokens": [i] * 32, "logits": [{"sha256": str(i)}] * 32} for i in range(6)
    ]

    class Engine:
        max_num_sequences = 2
        model = SimpleNamespace(decode_mode="rowwise")

        def __init__(self):
            self.requests = {}
            self.scheduler = SimpleNamespace(request=lambda r: self.requests[r])
            self.current = {}

        def submit(self, ids, budget):
            assert (
                sum(r.state is SequenceState.RUNNING for r in self.requests.values())
                < self.max_num_sequences
            )
            request_id = len(self.requests) + 1
            self.requests[request_id] = SimpleNamespace(
                token=ids[0], budget=budget, output=[], state=SequenceState.RUNNING
            )
            return request_id

        def step(self):
            events = []
            self.current.clear()
            for request_id, r in self.requests.items():
                if r.state is SequenceState.RUNNING:
                    r.output.append(r.token)
                    self.current[request_id] = {"sha256": str(r.token)}
                    if len(r.output) == r.budget:
                        r.state = SequenceState.COMPLETED
                    events.append(SimpleNamespace(request_id=request_id))
            return events

        def stats(self):
            assert all(
                r.state is SequenceState.COMPLETED for r in self.requests.values()
            )
            return {
                "kv_cache": {"allocated_blocks": 0, "reserved_blocks": 0},
                "kv_device_bytes": 0,
            }

    @contextmanager
    def observe(engine):
        yield engine.current

    monkeypatch.setattr(module, "observe_logits", observe)
    result = module.replay(Engine(), cases, references, staggered=staggered)
    assert result["summary"]["exact_token_cases"] == 6
    assert result["summary"]["events_exact"]
    assert result["summary"]["cache_reclaimed"]
    assert (
        result["summary"]["exact_logit_rows"] == result["summary"]["total_logit_rows"]
    )
