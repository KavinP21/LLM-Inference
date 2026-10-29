"""Time the MLX-VLM runner on an agent-shaped conversation.

Reports time to first token for a cold prompt, a follow-up turn (session
reuse) and a new conversation sharing the system prompt (snapshot reuse),
plus decode speed with and without the draft model.

    PYTHONPATH=python python benchmarks/bench_runner.py models/gemma-4-26b-a4b-mlx \\
        --draft models/gemma-4-26b-a4b-assistant
"""

from __future__ import annotations

import argparse
import json
import statistics

from instinct.vlm_runner import VlmRunner

SYSTEM = (
    "You are a coding agent working in a Python repository. Read files before "
    "editing them, keep changes small, and run the tests you touch. "
) * 40
# Drafting only runs on agent turns, which always carry tools.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }
]
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
    parser.add_argument("--draft")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()

    runner = VlmRunner(args.model, draft_path=args.draft)
    rows: dict[str, list[float]] = {}

    def record(name: str, value: float) -> None:
        rows.setdefault(name, []).append(value)

    for repetition in range(args.repetitions):
        runner.clear()
        for index, task in enumerate(TASKS):
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": task},
            ]
            first = runner.generate(messages, max_tokens=args.tokens)
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
            )
            record("session ttft ms", follow.metrics["ttft_ms"])
        if runner.draft_model is not None:
            # Force drafting on for a separate measurement of its effect.
            runner._draft_cooldown, runner._draft_ewma = 0, 3.0
            drafted = runner.generate(
                [{"role": "user", "content": TASKS[0]}],
                max_tokens=args.tokens,
                tools=TOOLS,
            )
            record("decode tok/s with draft", drafted.metrics["decode_tps"])
            runner._draft_cooldown = 10**9
            plain = runner.generate(
                [{"role": "user", "content": TASKS[0] + " "}],
                max_tokens=args.tokens,
                tools=TOOLS,
            )
            record("decode tok/s without draft", plain.metrics["decode_tps"])
            runner._draft_cooldown = 0

    summary = {
        name: round(statistics.median(values), 1) for name, values in rows.items()
    }
    summary["model"] = args.model
    summary["weights GiB"] = round(runner.weight_bytes / 2**30, 2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
