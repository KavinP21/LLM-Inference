from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def driver(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "prefix_driver", repo / "benchmarks/run_prefix_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean():
    return {
        "kv_cache": {"allocated_blocks": 0, "reserved_blocks": 0},
        "kv_device_bytes": 0,
        "scheduler": {"running": 0, "waiting": 0, "completed": 0},
    }


def item():
    return {
        "provenance": {"identity": "source"},
        "vocab_size": 32,
        "text_input_ids": [[i % 32] for i in range(25)],
    }


def continuation(module, count=32, prefill=1):
    row = module.fingerprint(np.arange(32, dtype=np.float32))
    return {
        "tokens": [31] * count,
        "logits": [copy.deepcopy(row) for _ in range(count)],
        "work": {"prefill_tokens": prefill, "decode_rows": count - 1},
    }


def quality(module):
    model = item()
    prompts = [
        [(17 + i * 104729 + index * 113) % 32 for i in range(length)]
        for index, length in enumerate(module.CONFIG["prompt_lengths"])
    ] + model["text_input_ids"]
    observations = []
    for index, prompt in enumerate(prompts):
        observations.append(
            {
                "case_index": index,
                "prompt": prompt,
                "branch_prompt": prompt[:-1] + [(prompt[-1] + 1) % 32],
                **{
                    mode: continuation(
                        module, prefill=0 if mode == "warm" else len(prompt)
                    )
                    for mode in (
                        "reference",
                        "cold",
                        "warm",
                        "branch_reference",
                        "branch",
                    )
                },
            }
        )
    return {
        "model": model["provenance"],
        "chunk_size": 127,
        "observations": observations,
        "cleanup": True,
        "cleanup_stats": {"baseline": clean(), "cached": clean()},
        "passed": True,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "coverage",
        "workload",
        "branch",
        "token",
        "hash",
        "dtype",
        "shape",
        "bool",
        "warm_work",
        "decode_work",
        "uncached_work",
        "cleanup",
        "model",
        "flag",
    ],
)
def test_quality_rejects_incomplete_or_tampered_evidence(monkeypatch, mutation):
    module = driver(monkeypatch)
    report = quality(module)
    assert module.quality_readback(report, item(), 127)
    observation = report["observations"][0]
    row = observation["warm"]["logits"][0]
    if mutation == "coverage":
        report["observations"].pop()
    elif mutation == "workload":
        observation["prompt"][0] = 0
    elif mutation == "branch":
        observation["branch_prompt"][0] = 0
    elif mutation == "token":
        observation["warm"]["tokens"][0] = 1
    elif mutation in {"hash", "dtype", "shape", "bool"}:
        row[{"hash": "sha256", "bool": "argmax"}.get(mutation, mutation)] = {
            "hash": "g" * 64,
            "dtype": "<f2",
            "shape": [1],
            "bool": True,
        }[mutation]
    elif mutation == "warm_work":
        observation["warm"]["work"]["prefill_tokens"] = 1
    elif mutation == "decode_work":
        observation["cold"]["work"]["decode_rows"] = 30
    elif mutation == "uncached_work":
        observation["reference"]["work"]["prefill_tokens"] = 0
    elif mutation == "cleanup":
        report["cleanup_stats"]["cached"]["kv_device_bytes"] = 1
    elif mutation == "model":
        report["model"] = {}
    else:
        report["passed"] = 1
    with pytest.raises(ValueError):
        module.quality_readback(report, item(), 127)


def test_complete_negative_quality_is_a_valid_gate_not_an_exception(monkeypatch):
    module = driver(monkeypatch)
    report = quality(module)
    report["observations"][0]["warm"]["logits"][0]["sha256"] = "0" * 64
    report["passed"] = False
    assert not module.quality_readback(report, item(), 127)


@pytest.mark.parametrize("mode", ["disabled", "cold", "primed"])
def test_raw_benchmark_latency_throughput_and_cache_work(monkeypatch, mode):
    module = driver(monkeypatch)
    report = benchmark_fixture(module, mode)
    assert module.benchmark_readback(
        report, item(), mode, 128, 1, {"runtime_source_sha256": "runtime"}
    )["generated_tokens_per_second"] == pytest.approx(320)


def benchmark_fixture(module, mode):
    report = {
        "model": item()["provenance"],
        "environment": {"runtime_source_sha256": "runtime"},
        "configuration": {
            "mode": mode,
            "prompt_length": 128,
            "concurrency": 1,
            "output_tokens": 32,
            "warmups": 5,
            "repetitions": 3,
            "decode_mode": "rowwise",
            "prefill_chunk_size": 512,
            "workload": [(17 + i * 104729) % 32 for i in range(128)],
            "prefix_cache_bytes": 0 if mode == "disabled" else 64 << 20,
            "priming_outside_timing": mode == "primed",
        },
        "requests": [
            {
                "request_id": trial + 1,
                "trial": trial,
                "prompt_tokens": 128,
                "output_tokens": 32,
                "ttft_ms": 10.0,
                "e2e_ms": 72.0,
                "tpot_ms": 2.0,
            }
            for trial in range(3)
        ],
        "trials": [
            {
                "trial": trial,
                "seconds": 0.1,
                "wall_ms": 100.0,
                "before": {**clean(), "prefix_cache": {"hits": 0, "reused_tokens": 0}},
                "after": {
                    **clean(),
                    "scheduler": {"waiting": 0, "running": 0, "completed": 1},
                    "prefix_cache": {
                        "hits": int(mode == "primed"),
                        "reused_tokens": 128 if mode == "primed" else 0,
                    },
                },
            }
            for trial in range(3)
        ],
        "cleared_stats": clean(),
    }
    report["aggregate"] = {
        "generated_tokens_per_second": 96 / sum(t["seconds"] for t in report["trials"]),
        **{
            name: {
                f"p{int(q * 100)}": module.percentile(
                    [r[name] for r in report["requests"]], q
                )
                for q in (0.5, 0.95, 0.99)
            }
            for name in ("ttft_ms", "tpot_ms", "e2e_ms")
        },
    }
    return report


@pytest.mark.parametrize(
    "mutation",
    [
        "unit",
        "nan",
        "tpot",
        "aggregate",
        "work",
        "count",
        "cleanup",
        "runtime",
        "priming",
        "length",
    ],
)
def test_raw_benchmark_rejects_forged_or_inconsistent_metrics(monkeypatch, mutation):
    module = driver(monkeypatch)
    report = benchmark_fixture(module, "primed")
    if mutation == "unit":
        report["trials"][0]["seconds"] = 100.0
    elif mutation == "nan":
        report["requests"][0]["e2e_ms"] = float("nan")
    elif mutation == "tpot":
        report["requests"][0]["tpot_ms"] = 3.0
    elif mutation == "aggregate":
        report["aggregate"]["generated_tokens_per_second"] *= 2
    elif mutation == "work":
        report["trials"][0]["after"]["prefix_cache"]["reused_tokens"] = 127
    elif mutation == "count":
        report["requests"].pop()
    elif mutation == "cleanup":
        report["cleared_stats"]["kv_cache"]["reserved_blocks"] = 1
    elif mutation == "runtime":
        report["environment"]["runtime_source_sha256"] = "different"
    elif mutation == "priming":
        report["configuration"]["priming_outside_timing"] = False
    else:
        report["configuration"]["workload"].pop()
    with pytest.raises(ValueError):
        module.benchmark_readback(
            report, item(), "primed", 128, 1, {"runtime_source_sha256": "runtime"}
        )


def test_32k_readback_requires_full_execution_and_parity(monkeypatch):
    module = driver(monkeypatch)
    report = {
        "model": item()["provenance"],
        "prompt_tokens": 32766,
        **{
            mode: continuation(module, count=2, prefill=0 if mode == "warm" else 32766)
            for mode in ("reference", "cold", "warm")
        },
        "cleanup_stats": {"baseline": clean(), "cached": clean()},
        "cleanup": True,
        "passed": True,
    }
    assert module.long_readback(report, item())
    report["warm"]["work"]["decode_rows"] = 0
    with pytest.raises(ValueError):
        module.long_readback(report, item())


@pytest.mark.parametrize(
    "mutation", ["count", "namespace", "cow", "pressure", "work", "cleanup"]
)
def test_stress_readback_requires_shared_execution_and_cleanup(monkeypatch, mutation):
    module = driver(monkeypatch)
    root = [(i * 104729 + 37) % 32 for i in range(513)]
    report = {
        "model": item()["provenance"],
        "prompts": [
            root if i % 2 == 0 else root[:-1] + [(i + 2) % 32] for i in range(8)
        ],
        "references": [continuation(module, prefill=513) for _ in range(8)],
        "results": [continuation(module) for _ in range(8)],
        "work": {"prefill_tokens": 20, "decode_rows": 8 * 31},
        "namespace_misses_before": 1,
        "namespace_misses_after": 2,
        "isolated": True,
        "cow_copies": 8,
        "pressure_stats": {"prefix_cache": {"evictions": 3}},
        "cleanup_stats": {k: clean() for k in ("baseline", "cached", "pressure")},
        "cleanup": True,
        "passed": True,
    }
    assert module.stress_readback(report, item())
    if mutation == "count":
        report["results"].pop()
    elif mutation == "namespace":
        report["namespace_misses_after"] = 1
    elif mutation == "cow":
        report["cow_copies"] = 0
    elif mutation == "pressure":
        report["pressure_stats"]["prefix_cache"]["evictions"] = 0
    elif mutation == "work":
        report["work"]["decode_rows"] -= 1
    else:
        report["cleanup_stats"]["pressure"]["scheduler"]["waiting"] = 1
    with pytest.raises(ValueError):
        module.stress_readback(report, item())


def test_failed_validation_forbids_any_timing(monkeypatch, tmp_path):
    module = driver(monkeypatch)
    monkeypatch.setattr(module, "validation_readback", lambda root: False)
    with pytest.raises(ValueError, match="forbids timing"):
        module.benchmark(lambda *a, **k: pytest.fail("spawned timing"), tmp_path)
    monkeypatch.setattr(module, "contract", lambda root: {})
    (tmp_path / "matrix").mkdir()
    with pytest.raises(ValueError, match="timing after rejected"):
        module.audit(tmp_path)


def test_final_index_rejects_missing_or_unexpected_files(monkeypatch, tmp_path):
    module = driver(monkeypatch)
    module.save(tmp_path / "verification.json", {"milestone_passed": True})
    module.save(tmp_path / "evidence-index.json", {"files_sha256": {}})
    monkeypatch.setattr(module, "audit", lambda root: {"milestone_passed": True})
    with pytest.raises(ValueError, match="final evidence"):
        module.final_audit(tmp_path, 1)


def test_exclusive_artifact_saves_never_overwrite(tmp_path, monkeypatch):
    module = driver(monkeypatch)
    path = tmp_path / "checkpoint.json"
    module.save(path, {"passed": False})
    with pytest.raises(FileExistsError):
        module.save(path, {"passed": True})
    assert json.loads(path.read_text()) == {"passed": False}
