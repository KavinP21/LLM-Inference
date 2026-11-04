"""One long prompt, then a follow-up turn on top of it.

The first turn shows what a long context costs to prefill. The second shows
what a chat pays once the prefix is cached: only the new message is
processed. Decode speed is measured at that depth.

    PYTHONPATH=python python benchmarks/long_context.py \\
        models/gemma-4-26b-a4b-mlx --tokens 131072 --output long.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from instinct.chat_runner import ChatRunner


def filler(tokenizer, tokens: int) -> str:
    """Text of about ``tokens`` tokens: numbered log lines a question can
    point into."""
    rng = random.Random(7)
    lines, count, n = [], 0, 0
    while count < tokens:
        n += 1
        line = (
            f"[{n:06d}] worker {rng.randint(1, 64)} finished batch "
            f"{rng.randint(1000, 9999)} in {rng.randint(10, 999)} ms with status "
            f"{rng.choice(['ok', 'ok', 'ok', 'retry', 'slow'])}.\n"
        )
        lines.append(line)
        if n % 200 == 0:
            count = len(tokenizer.encode("".join(lines), add_special_tokens=False))
    text = "".join(lines)
    ids = tokenizer.encode(text, add_special_tokens=False)[:tokens]
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--kv-cache-gb", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ChatRunner(
        args.model,
        max_num_sequences=2,
        max_context_tokens=args.tokens + 8192,
        kv_cache_bytes=args.kv_cache_gb << 30,
    ) as runner:
        log = filler(runner.tokenizer, args.tokens)
        convo = [
            {
                "role": "user",
                "content": log + "\nHow many lines above have status retry? Estimate.",
            }
        ]
        first = runner.generate(convo, max_tokens=128)
        convo += [
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": "Which worker number appears most? One line."},
        ]
        second = runner.generate(convo, max_tokens=128)
    report = {
        "model": args.model,
        "context_tokens": first.input_tokens,
        "first_turn": first.metrics,
        "follow_up": second.metrics,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
