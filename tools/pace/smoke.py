"""Real models on this GPU: right answers, cache reuse and first speeds.

For each model, through ChatRunner as the server uses it:
- a question with a known answer, then a follow-up that should reuse the cache
- a 4,000-token prompt, for prefill speed

Exits nonzero if an answer is wrong or the follow-up wasn't served from the
cache.

    python tools/pace/smoke.py --output smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from instinct.chat_runner import ChatRunner

MODELS = ["models/qwen3.5-4b-mlx", "models/gemma-4-26b-a4b-mlx"]
NOTES = (
    "The deploy script copies the build to the staging host, runs the database "
    "migrations, warms the cache and then restarts the API servers one at a time. "
)


def run(path: str) -> dict:
    with ChatRunner(
        path, max_num_sequences=4, max_context_tokens=32768, kv_cache_bytes=8 << 30
    ) as runner:
        convo = [{"role": "user", "content": "What is 12 times 12? Number only."}]
        first = runner.generate(convo, max_tokens=24)
        convo += [
            {"role": "assistant", "content": first.text},
            {"role": "user", "content": "Add one. Number only."},
        ]
        second = runner.generate(convo, max_tokens=24)
        long = runner.generate(
            [{"role": "user", "content": NOTES * 90 + "\n\nSummarize in one line."}],
            max_tokens=48,
        )
        stats = runner.engine.stats()
    checks = {
        "first_answer": "144" in first.text,
        "second_answer": "145" in second.text,
        "follow_up_cached": second.cached_tokens >= first.input_tokens,
    }
    return {
        "model": path,
        "kernels": stats["kernels"],
        "answers": [first.text, second.text],
        "checks": checks,
        "follow_up": second.metrics,
        "long_prompt": long.metrics,
        "ok": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="*", default=MODELS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []
    for path in args.models:
        result = run(path)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not all(r["ok"] for r in results):
        sys.exit("a model gave a wrong answer or missed the cache; see the report")


if __name__ == "__main__":
    main()
