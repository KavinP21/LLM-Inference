"""Time cold, shared-prefix and follow-up prompts on the native chat runner.

PYTHONPATH=python python benchmarks/bench_runner.py models/gemma-4-26b-a4b-mlx
"""

from __future__ import annotations

import argparse
import json
import statistics

from instinct.chat_runner import ChatRunner

SYSTEM = (
    "You are a coding agent working in a Python repository. Read files before "
    "editing them, keep changes small, and run the tests you touch. "
) * 40
TASKS = [
    "Write a Python function that parses an ISO-8601 duration like P3DT4H5M into seconds.",
    "Explain what a paged KV cache is and why inference servers use one.",
    "Summarize the tradeoffs of speculative decoding for mixture-of-experts models.",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model")
    parser.add_argument("--draft", help="requires the historical MLX-VLM benchmark")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.draft:
        parser.error(
            "For draft measurements, use this script at commit 50e4739 "
            "with the mlx and vlm extras. The native runner has no drafter."
        )
    if min(args.tokens, args.repetitions) <= 0:
        parser.error("--tokens and --repetitions must be positive")

    rows: dict[str, list[float]] = {}

    def record(name: str, value: float) -> None:
        rows.setdefault(name, []).append(value)

    with ChatRunner(args.model) as runner:
        for repetition in range(args.repetitions):
            runner.clear()
            # Separate trials while allowing cache reuse within each trial.
            namespace = f"bench-{repetition}"
            for index, task in enumerate(TASKS):
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task},
                ]
                first = runner.generate(
                    messages, max_tokens=args.tokens, namespace=namespace
                )
                record(
                    "cold ttft ms" if index == 0 else "snapshot ttft ms",
                    first.metrics["ttft_ms"],
                )
                record("decode tok/s", first.metrics["decode_tps"])
                follow = runner.generate(
                    messages
                    + [
                        {"role": "assistant", "content": first.text},
                        {"role": "user", "content": "Now make it shorter."},
                    ],
                    max_tokens=64,
                    namespace=namespace,
                )
                record("session ttft ms", follow.metrics["ttft_ms"])
        weights_gib = round(runner.weight_bytes / 2**30, 2)

    summary = {
        name: round(statistics.median(values), 1) for name, values in rows.items()
    }
    print(
        json.dumps(
            {**summary, "model": args.model, "weights GiB": weights_gib}, indent=2
        )
    )


if __name__ == "__main__":
    main()
