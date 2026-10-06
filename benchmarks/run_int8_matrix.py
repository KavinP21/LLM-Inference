"""Isolated-process FP16 / naive INT8 / fused INT8 comparison on one Metal GPU."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--include-reconstruct",
        action="store_true",
        help="also measure one-pass reconstruction plus native GEMM at all row counts",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="capture the first fused trial of each configuration",
    )
    args = parser.parse_args()
    if (
        min(
            *args.prompt_lengths,
            *args.concurrencies,
            args.output_length,
            args.repetitions,
        )
        <= 0
        or args.warmups < 0
    ):
        parser.error("workload dimensions must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = [
        ("fp16", args.source, "auto"),
        ("int8-dequantize", args.model, "dequantize"),
        ("int8-metal", args.model, "metal"),
    ]
    if args.include_reconstruct:
        modes.append(("int8-reconstruct", args.model, "reconstruct"))
    for index, (length, concurrency) in enumerate(
        (p, c) for p in args.prompt_lengths for c in args.concurrencies
    ):
        workload = args.output_dir / f"workload-p{length}-c{concurrency}.json"
        with workload.open("x") as handle:
            handle.write(json.dumps([[42] * length for _ in range(concurrency)]) + "\n")
        # Rotate implementation order; never run competing GPU benchmarks concurrently.
        order = modes[index % len(modes) :] + modes[: index % len(modes)]
        for mode, model, int8_mode in order:
            output = args.output_dir / f"{mode}-p{length}-c{concurrency}.json"
            if output.exists():
                raise FileExistsError(
                    f"refusing to overwrite benchmark evidence: {output}"
                )
            command = [
                sys.executable,
                "-m",
                "forge_llm.benchmark",
                "--model",
                str(model),
                "--backend",
                "mlx",
                "--prompts",
                str(workload),
                "--output",
                str(output),
                "--max-model-length",
                "2048",
                "--max-sequences",
                str(concurrency),
                "--kv-cache-mib",
                "1024",
                "--int8-mode",
                int8_mode,
                "--output-length",
                str(args.output_length),
                "--warmups",
                str(args.warmups),
                "--repetitions",
                str(args.repetitions),
            ]
            if args.capture and mode == "int8-metal":
                command += [
                    "--capture",
                    str(
                        Path("profiles")
                        / args.output_dir.name
                        / f"p{length}-c{concurrency}.gputrace"
                    ),
                ]
            print(
                f"Running {mode}, prompt={length}, concurrency={concurrency}",
                flush=True,
            )
            run(command, check=True)


def main() -> None:
    with exclusive_gpu_workflow() as run:
        _run(run)


if __name__ == "__main__":
    main()
