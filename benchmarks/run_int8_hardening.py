"""Regenerate frozen-policy evidence without treating failed quality gates as success."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpu_guard import exclusive_gpu_workflow


def _run(run) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-benchmarks", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse evidence directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)
    gates, batch_checks = {}, {}
    # Serialize GPU work. A saved, explicitly failed quality gate is expected
    # evidence; an execution error, missing report or malformed output is not.
    for mode in ["reconstruct", "metal"]:
        output = args.output_dir / f"quality-{mode}.json"
        command = [
            sys.executable,
            "-m",
            "forge_llm.validate_int8",
            "--source",
            str(args.source),
            "--model",
            str(args.model),
            "--tokenizer",
            args.tokenizer,
            "--int8-mode",
            mode,
            "--output",
            str(output),
        ]
        print(f"Held-out quality: {mode}", flush=True)
        result = run(command, check=False)
        if result.returncode not in {0, 1} or not output.is_file():
            raise RuntimeError(f"quality execution failed: {result.returncode}")
        report = json.loads(output.read_text())
        passed = report["gates"]["strict_checkpoint_passed"]
        if passed != (result.returncode == 0):
            raise RuntimeError("quality gate and exit status disagree")
        gates[mode] = report["gates"]
        print(f"32K resource gate: {mode}", flush=True)
        run(
            [
                sys.executable,
                "-m",
                "forge_llm.long_context",
                "--model",
                str(args.model),
                "--context-lengths",
                "32768",
                "--output-tokens",
                "2",
                "--kv-cache-mib",
                "1024",
                "--int8-mode",
                mode,
                "--output",
                str(args.output_dir / f"32k-{mode}.json"),
            ],
            check=True,
        )
        batch_path = args.output_dir / f"batch-{mode}.json"
        print(f"Dynamic-arrival batch regression: {mode}", flush=True)
        batch_result = run(
            [
                sys.executable,
                "benchmarks/check_int8_batch.py",
                "--model",
                str(args.model),
                "--quality",
                str(output),
                "--mode",
                mode,
                "--output",
                str(batch_path),
            ],
            check=False,
        )
        if batch_result.returncode not in {0, 1} or not batch_path.is_file():
            raise RuntimeError("batch regression execution failed")
        batch_checks[mode] = json.loads(batch_path.read_text())["summary"]
        if batch_checks[mode]["passed"] != (batch_result.returncode == 0):
            raise RuntimeError("batch gate and exit status disagree")
    if not args.skip_benchmarks:
        run(
            [
                sys.executable,
                "benchmarks/run_int8_matrix.py",
                "--source",
                str(args.source),
                "--model",
                str(args.model),
                "--output-dir",
                str(args.output_dir / "matrix"),
                "--include-reconstruct",
                "--warmups",
                "5",
                "--repetitions",
                "3",
            ],
            check=True,
        )
    checkpoint = {
        "quality_gates": gates,
        "batch_checks": batch_checks,
        "strict_checkpoint_passed": all(
            gate["strict_checkpoint_passed"] for gate in gates.values()
        )
        and all(check["passed"] for check in batch_checks.values()),
        "benchmark_executed": not args.skip_benchmarks,
        "scope": "one frozen policy and model family; not permission to advance W4A16",
    }
    with (args.output_dir / "checkpoint.json").open("x") as handle:
        json.dump(checkpoint, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(checkpoint, indent=2))
    if not checkpoint["strict_checkpoint_passed"]:
        raise SystemExit(1)


def main() -> None:
    with exclusive_gpu_workflow() as run:
        _run(run)


if __name__ == "__main__":
    main()
