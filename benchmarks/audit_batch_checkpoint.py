"""Portable hash/gate audit of saved batching evidence; no device initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from forge_llm.benchmark import source_provenance


def audit(root):
    sources, records, reports, checkpoints = set(), [], {}, []
    folders = ["regression", "resources-and-matrix"]
    if (root / "extended-stage-replay" / "checkpoint.json").exists():
        folders.append("extended-stage-replay")
    for folder in folders:
        checkpoint = root / folder / "checkpoint.json"
        index = json.loads(checkpoint.read_text())
        checkpoints.append(
            {
                "path": str(checkpoint.relative_to(root)),
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
        if not index["all_requested_gates_passed"]:
            raise ValueError("requested checkpoint gate failed")
        for item in index["records"]:
            path = checkpoint.parent / item["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("evidence path escapes result directory")
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"evidence checksum mismatch: {path}")
            report = json.loads(content)
            runtime = report["environment"]["runtime_source_sha256"]
            if (
                runtime != index["runtime_source_sha256"]
                or runtime != item["runtime_source_sha256"]
            ):
                raise ValueError("runtime sources are inconsistent")
            sources.add(runtime)
            name = str(path.relative_to(root))
            records.append(
                {
                    "path": name,
                    "sha256": digest,
                    "purpose": item["gate"] or "stage_or_performance",
                    "passed": item["passed"],
                }
            )
            reports[name] = report
    current = source_provenance()["runtime_source_sha256"]
    if sources != {current}:
        raise ValueError("current runtime changed since evidence collection")
    batches = [r for name, r in reports.items() if name.endswith("-batch.json")]
    boundaries = [r for name, r in reports.items() if name.endswith("-boundary.json")]
    contexts = [r for name, r in reports.items() if name.endswith("-32k.json")]
    matrix = [r for name, r in reports.items() if "/matrix/" in name]
    stages = [r for name, r in reports.items() if "/gemma-stages-" in name]
    for r in stages:
        if r["configuration"]["decode_mode"] == "rowwise" and any(
            item["first_different_stage"] is not None
            or item["independent_argmax"] != item["batched_argmax"]
            for item in r["replays"]
        ):
            raise ValueError("rowwise stage replay diverged")
    if tuple(map(len, (batches, boundaries, contexts, matrix))) != (6, 6, 6, 32):
        raise ValueError("incomplete checkpoint matrix")
    model_ids = {r["model"]["model_data_sha256"] for r in batches}
    expected_shapes = {
        (model, p, c, mode)
        for model in model_ids
        for p in (128, 1024)
        for c in (1, 8)
        for mode in ("batched", "rowwise")
    }
    measured_shapes = {
        (
            r["model"]["model_data_sha256"],
            r["configuration"]["prompt_token_lengths"][0],
            r["configuration"]["concurrency"],
            r["configuration"]["decode_mode"],
        )
        for r in matrix
    }
    if (
        len(model_ids) != 4
        or len(measured_shapes) != 32
        or measured_shapes != expected_shapes
    ):
        raise ValueError("benchmark A/B coverage mismatch")
    for r in batches:
        if not r["gates"]["checkpoint_passed"]:
            raise ValueError("batch numerical gate failed")
    for r in boundaries:
        if not r["gates"]["checkpoint_passed"]:
            raise ValueError("boundary gate failed")
    for r in contexts:
        if not r["all_reclaimed"]:
            raise ValueError("context cache leaked")
        for o in r["observations"]:
            if (
                o["context_tokens"],
                o["prompt_tokens"],
                o["output_tokens"],
                o["decode_iterations"],
                o["prefill_iterations"],
                o["peak_pool_blocks"],
            ) != (32768, 32766, 2, 1, 64, 2048):
                raise ValueError("context workload differs from the declared gate")
    for r in matrix:
        config, stats = r["configuration"], r["engine_stats"]
        if (
            config["warmups"],
            config["repetitions"],
            config["output_length"],
            config["warmup_output_length"],
        ) != (5, 3, 32, 32):
            raise ValueError("benchmark protocol mismatch")
        if (
            stats["decode_mode"] != config["decode_mode"]
            or not all(
                stats["kv_cache"][k] == 0
                for k in ("allocated_blocks", "reserved_blocks")
            )
            or stats["kv_device_bytes"] != 0
        ):
            raise ValueError("benchmark policy mismatch or cache leak")
        if len(r["requests"]) != config["concurrency"] * 3 or any(
            item["output_tokens"] != 32 for item in r["requests"]
        ):
            raise ValueError("benchmark did not execute every requested token")
    workloads = []
    for name, report in reports.items():
        if "/matrix/" not in name:
            continue
        path = root / name
        stem = path.stem.replace("-batched-", "-").replace("-rowwise-", "-")
        workload = path.with_name(stem + "-workload.json")
        digest = hashlib.sha256(workload.read_bytes()).hexdigest()
        if digest != report["configuration"]["workload_sha256"]:
            raise ValueError("benchmark workload checksum mismatch")
        relative = str(workload.relative_to(root))
        if relative not in {item["path"] for item in workloads}:
            workloads.append({"path": relative, "sha256": digest})
    quality = {}
    for family in ("qwen", "gemma"):
        fp16 = reports[f"regression/{family}-fp16-batch.json"]["references"]
        quality[family] = {
            mode: sum(
                a["tokens"] == b["tokens"]
                for a, b in zip(
                    fp16,
                    reports[f"regression/{family}-mixed-{mode}-batch.json"][
                        "references"
                    ],
                    strict=True,
                )
            )
            for mode in ("metal", "reconstruct")
        }
    return {
        "schema_version": 1,
        "runtime_source_sha256": current,
        "batch_consistency_checkpoint_passed": True,
        "strict_int8_quality_passed": all(
            n == 25 for modes in quality.values() for n in modes.values()
        ),
        "fresh_cross_artifact_exact_cases_out_of_25": quality,
        "exact_rowwise_arrival_logit_rows": sum(
            w["summary"]["exact_logit_rows"]
            for r in batches
            for w in r["modes"]["rowwise"]["workloads"]
        ),
        "exact_rowwise_boundary_logit_rows": sum(
            r["batched"]["summary"]["exact_logit_rows"] for r in boundaries
        ),
        "published_raw_records": records,
        "checkpoint_files": checkpoints,
        "workload_files": workloads,
        "counts": {
            "batch": 6,
            "boundary": 6,
            "context": 6,
            "performance": 32,
            "stage": len(stages),
        },
        "excluded_diagnostics": ["gemma-stage-trace.json", "gemma-fp16-batching.json"],
        "limitations": "Same-artifact consistency only. No cross-artifact INT8 pass, Transformers corpus certification, CUDA validation, GPU counter attribution or production tail-latency certification.",
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.result_dir)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "published_raw_records"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
