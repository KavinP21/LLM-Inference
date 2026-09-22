from types import SimpleNamespace

from forge_llm.benchmark import model_provenance, percentile, run_once
from forge_llm.format import ModelConfig, write_engine

import numpy as np


class DummyEngine:
    def __init__(self) -> None:
        self.next_id = 1
        self.remaining: dict[int, int] = {}

    def submit(self, prompt: list[int], output_length: int, eos: list[int]) -> int:
        del prompt, eos
        request_id = self.next_id
        self.next_id += 1
        self.remaining[request_id] = output_length
        return request_id

    def step(self) -> list[SimpleNamespace]:
        events = []
        for request_id in list(self.remaining):
            self.remaining[request_id] -= 1
            finished = self.remaining[request_id] == 0
            events.append(SimpleNamespace(request_id=request_id, token=42, finished=finished))
            if finished:
                del self.remaining[request_id]
        return events

    def stats(self) -> dict:
        active = len(self.remaining)
        return {"kv_cache": {"allocated_blocks": active, "reserved_blocks": 0,
                             "internal_fragmentation_tokens": active * 15,
                             "occupancy": active / 4}}


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([0.0, 10.0], 0.95) == 9.5


def test_run_once_collects_fixed_length_requests() -> None:
    metrics, wall_ms, peak = run_once(DummyEngine(), [[1, 2], [3]], 3, [])
    assert len(metrics) == 2
    assert [item.output_tokens for item in metrics] == [3, 3]
    assert all(item.tpot_ms is not None and item.tpot_ms >= 0 for item in metrics)
    assert wall_ms >= 0
    assert peak["allocated_blocks"] == 2
    assert peak["internal_fragmentation_tokens"] == 30


def test_model_provenance_reads_container_digest(tmp_path) -> None:
    path = tmp_path / "model.engine"
    config = ModelConfig(4, 2, 4, 1, 1, 1, 8, 1, 10000.0, 1e-6)
    result = write_engine(path, config, {"weight": np.ones((2, 2), dtype=np.float16)})
    provenance = model_provenance(path)
    assert provenance["model_format_version"] == 1
    assert provenance["model_data_sha256"] == result["data_sha256"]
    assert provenance["model_file_bytes"] == path.stat().st_size
