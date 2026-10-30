"""Several users on one model at once, as the app and its API serve them.

Three users decode together, each with different sampling (greedy, nucleus,
top-k with min-p). Partway through, a fourth arrives with a long prompt, so
the others' token gaps show what chunked prefill costs them. At the end
every K/V page must be back in its pool.

    PYTHONPATH=python python benchmarks/multi_user.py models/gemma-4-26b-a4b-mlx
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from instinct.mlx_engine import MlxEngine
from instinct.sampling import SamplingParams

SETTINGS = [
    SamplingParams(),
    SamplingParams(temperature=0.7, top_p=0.95, seed=1),
    SamplingParams(temperature=1.0, top_k=64, min_p=0.05, seed=2),
]


def prompt(length: int, seed: int) -> list[int]:
    # Token ids from the common range of both vocabularies.
    return [(seed * 7919 + i * 104729) % 20000 + 1000 for i in range(length)]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run(model: Path, *, users: int, tokens: int, late_prompt: int, **options) -> dict:
    engine = MlxEngine(
        model,
        max_num_sequences=users + 1,
        kv_cache_bytes=16 << 30,
        max_model_length=65536,
        **options,
    )
    engine.generate(prompt(64, 99), 8)  # Compile kernels before timing.
    lengths = [256 + 512 * i for i in range(users)]
    ids = [
        engine.submit(prompt(n, i), tokens, sampling=SETTINGS[i % len(SETTINGS)])
        for i, n in enumerate(lengths)
    ]
    stamps: dict[int, list[float]] = {i: [] for i in ids}
    late = None
    late_started = late_first = None
    started = time.perf_counter()
    while engine.has_work():
        events = engine.step()
        now = time.perf_counter()
        for event in events:
            if event.request_id in stamps:
                stamps[event.request_id].append(now)
            elif event.request_id == late and late_first is None:
                late_first = now
        # The late user arrives once everyone has a quarter of their reply.
        if late is None and all(len(s) >= tokens // 4 for s in stamps.values()):
            late = engine.submit(prompt(late_prompt, 77), 16)
            late_started = now
    elapsed = time.perf_counter() - started

    per_user = []
    gaps_during_prefill = []
    for index, request_id in enumerate(ids):
        times = stamps[request_id]
        gaps = [b - a for a, b in zip(times, times[1:])]
        decode = len(times) - 1
        per_user.append(
            {
                "sampling": SETTINGS[index % len(SETTINGS)].__dict__,
                "prompt_tokens": lengths[index],
                "tokens": len(times),
                "tok_per_s": round(decode / (times[-1] - times[0]), 1),
                "gap_p50_ms": round(statistics.median(gaps) * 1000, 1),
                "gap_p99_ms": round(percentile(gaps, 0.99) * 1000, 1),
            }
        )
        gaps_during_prefill += [
            (b - a) * 1000
            for a, b in zip(times, times[1:])
            if late_started <= b and a <= (late_first or b)
        ]
    engine.clear_prefix_cache()
    pools = engine.stats()["kv_pools"]
    leaked = {
        name: p["used_pages"] + p["reserved_pages"]
        for name, p in pools.items()
        if p["used_pages"] or p["reserved_pages"]
    }
    result = {
        "model": str(model),
        "users": per_user,
        "aggregate_tok_per_s": round(
            sum(u["tokens"] - 1 for u in per_user) / elapsed, 1
        ),
        "late_prompt_tokens": late_prompt,
        "late_time_to_first_token_s": round(late_first - late_started, 2)
        if late_first
        else None,
        "gap_while_late_prefills_ms": {
            "p50": round(statistics.median(gaps_during_prefill), 1)
            if gaps_during_prefill
            else None,
            "max": round(max(gaps_during_prefill), 1) if gaps_during_prefill else None,
        },
        "leaked_pages": leaked,
    }
    engine.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("--users", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--late-prompt", type=int, default=8192)
    parser.add_argument("--prefill-chunk", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.model,
        users=args.users,
        tokens=args.tokens,
        late_prompt=args.late_prompt,
        prefill_chunk_size=args.prefill_chunk,
    )
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
