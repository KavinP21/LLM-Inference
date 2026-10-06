"""Serialized reproducible numerical-consistency checkpoint and controlled A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from gpu_guard import exclusive_gpu_workflow


def artifacts(historical_dir):
    for family, stem in (("qwen", "qwen2.5-0.5b"), ("gemma", "gemma-3-1b-it")):
        for precision, suffix, mode in (
            ("fp16", "", "auto"),
            ("mixed-metal", "-int8-mixed", "metal"),
            ("mixed-reconstruct", "-int8-mixed", "reconstruct"),
        ):
            yield (
                f"{family}-{precision}",
                Path("models") / f"{stem}{suffix}.engine",
                mode,
                historical_dir
                / family
                / f"quality-{'reconstruct' if mode == 'auto' else mode}.json",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--historical-dir",
        type=Path,
        default=Path("benchmarks/results/int8-hardening-m3-max-2026-10-01"),
    )
    parser.add_argument(
        "--checks",
        choices=["batch", "boundary", "context", "matrix", "trace"],
        nargs="+",
        default=["batch", "boundary", "context", "matrix", "trace"],
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--trace-steps", type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions <= 0:
        parser.error("invalid benchmark repetitions")
    if not 1 <= args.trace_steps <= 31:
        parser.error("trace-steps must be between 1 and 31")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []

    def execute(run, command, output, gate=None):
        if output.exists():
            raise FileExistsError(output)
        print("Running " + output.name, flush=True)
        result = run(command, check=False)
        if result.returncode not in (0, 1) or not output.exists():
            raise RuntimeError(
                f"execution failed, not a saved correctness gate: {output}"
            )
        report = json.loads(output.read_text())
        passed = (
            report["gates"]["checkpoint_passed"]
            if gate in {"batch", "boundary"}
            else report["all_reclaimed"]
            if gate == "context"
            else True
        )
        # The context CLI saves resource status but does not encode it in its
        # exit code. The batch CLI has an explicit failed-gate exit contract.
        expected_returncode = 0 if gate not in {"batch", "boundary"} or passed else 1
        if result.returncode != expected_returncode:
            raise RuntimeError("exit status contradicts saved gate")
        records.append(
            {
                "path": str(output.relative_to(args.output_dir)),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "gate": gate,
                "passed": passed,
                "runtime_source_sha256": report["environment"]["runtime_source_sha256"],
            }
        )

    with exclusive_gpu_workflow() as run:
        for label, model, mode, quality in artifacts(args.historical_dir):
            if "batch" in args.checks:
                output = args.output_dir / f"{label}-batch.json"
                execute(
                    run,
                    [
                        sys.executable,
                        "benchmarks/validate_batch_numerics.py",
                        "--model",
                        str(model),
                        "--quality",
                        str(quality),
                        "--int8-mode",
                        mode,
                        "--output",
                        str(output),
                    ],
                    output,
                    "batch",
                )
            if "boundary" in args.checks:
                output = args.output_dir / f"{label}-boundary.json"
                execute(
                    run,
                    [
                        sys.executable,
                        "benchmarks/validate_batch_boundaries.py",
                        "--model",
                        str(model),
                        "--int8-mode",
                        mode,
                        "--output",
                        str(output),
                    ],
                    output,
                    "boundary",
                )
            if "context" in args.checks:
                output = args.output_dir / f"{label}-32k.json"
                execute(
                    run,
                    [
                        sys.executable,
                        "-m",
                        "forge_llm.long_context",
                        "--model",
                        str(model),
                        "--int8-mode",
                        mode,
                        "--decode-mode",
                        "rowwise",
                        "--context-lengths",
                        "32768",
                        "--output-tokens",
                        "2",
                        "--kv-cache-mib",
                        "1024",
                        "--output",
                        str(output),
                    ],
                    output,
                    "context",
                )
        if "trace" in args.checks:
            for mode in ("batched", "rowwise"):
                output = args.output_dir / f"gemma-stages-{mode}.json"
                execute(
                    run,
                    [
                        sys.executable,
                        "benchmarks/trace_batch_numerics.py",
                        "--model",
                        "models/gemma-3-1b-it.engine",
                        "--quality",
                        str(args.historical_dir / "gemma/quality-reconstruct.json"),
                        "--decode-mode",
                        mode,
                        "--steps",
                        str(args.trace_steps),
                        "--output",
                        str(output),
                    ],
                    output,
                )
        if "matrix" in args.checks:
            matrix = args.output_dir / "matrix"
            matrix.mkdir()
            # Same weights/workload in each pair; rotate execution order and
            # never combine correctness host-copy instrumentation with timings.
            for label, model, int8_mode, _ in artifacts(args.historical_dir):
                if int8_mode == "reconstruct":
                    continue  # Direct/mixed plus FP16; no all-mode performance claim.
                for index, (length, width) in enumerate(
                    (p, c) for p in (128, 1024) for c in (1, 8)
                ):
                    workload = matrix / f"{label}-p{length}-c{width}-workload.json"
                    with workload.open("x") as handle:
                        json.dump([[42] * length for _ in range(width)], handle)
                    modes = (
                        ("batched", "rowwise")
                        if index % 2 == 0
                        else ("rowwise", "batched")
                    )
                    for mode in modes:
                        output = matrix / f"{label}-{mode}-p{length}-c{width}.json"
                        execute(
                            run,
                            [
                                sys.executable,
                                "-m",
                                "forge_llm.benchmark",
                                "--model",
                                str(model),
                                "--backend",
                                "mlx",
                                "--int8-mode",
                                int8_mode,
                                "--decode-mode",
                                mode,
                                "--prompts",
                                str(workload),
                                "--max-sequences",
                                str(width),
                                "--max-model-length",
                                "2048",
                                "--kv-cache-mib",
                                "1024",
                                "--output-length",
                                "32",
                                "--warmups",
                                str(args.warmups),
                                "--repetitions",
                                str(args.repetitions),
                                "--output",
                                str(output),
                            ],
                            output,
                        )
    hashes = {r["runtime_source_sha256"] for r in records}
    if len(hashes) != 1:
        raise RuntimeError("runtime source changed during evidence collection")
    result = {
        "schema_version": 1,
        "checks": args.checks,
        "records": records,
        "all_requested_gates_passed": all(r["passed"] for r in records),
        "runtime_source_sha256": hashes.pop(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation": "Batch consistency is independent of cross-artifact INT8 quality. No CUDA claims or GPU-counter evidence.",
    }
    with (args.output_dir / "checkpoint.json").open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not result["all_requested_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
