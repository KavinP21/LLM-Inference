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

FIRST_QUESTION = "How many lines above have status retry? Estimate."
FOLLOW_UP = "Which worker number appears most? One line."
REPLY_TOKENS = 128


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


def log_token_budget(runner: ChatRunner, requested: int) -> int:
    framing = runner.render(
        [
            {"role": "user", "content": FIRST_QUESTION},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": FOLLOW_UP},
        ],
        tools=None,
        thinking=False,
    )
    framing_tokens = len(runner.tokenizer.encode(framing, add_special_tokens=False))
    # Leave room for both replies and for tokenization at the text boundaries.
    available = runner.max_context_tokens - framing_tokens - 2 * REPLY_TOKENS - 32
    if available < 1:
        raise ValueError("model context is too small for the two-turn benchmark")
    return min(requested, available)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, default=131072, help="target log length")
    parser.add_argument("--kv-cache-gb", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens <= 0:
        parser.error("--tokens must be positive")
    with ChatRunner(
        args.model,
        max_num_sequences=2,
        max_context_tokens=args.tokens + 8192,
        kv_cache_bytes=args.kv_cache_gb << 30,
    ) as runner:
        log_tokens = log_token_budget(runner, args.tokens)
        log = filler(runner.tokenizer, log_tokens)
        convo = [
            {
                "role": "user",
                "content": f"{log}\n{FIRST_QUESTION}",
            }
        ]
        first = runner.generate(convo, max_tokens=REPLY_TOKENS)
        convo += [
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": FOLLOW_UP},
        ]
        second = runner.generate(convo, max_tokens=REPLY_TOKENS)
    report = {
        "model": args.model,
        "requested_log_tokens": args.tokens,
        "log_token_budget": log_tokens,
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
