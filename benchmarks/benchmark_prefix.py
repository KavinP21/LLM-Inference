"""Isolated FP16 cold/primed prefix-cache A/B; raw request timings, no counters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_llm.benchmark import (
    model_provenance,
    percentile,
    run_once,
    source_provenance,
)
from forge_llm.mlx_engine import MlxEngine
from gpu_guard import exclusive_gpu_workflow
from run_second_order_checkpoint import save


def measure(model, mode, length, concurrency, warmups=5, repetitions=3):
    from dataclasses import asdict

    if (
        mode not in {"disabled", "cold", "primed"}
        or min(length, concurrency, repetitions) <= 0
        or warmups < 0
    ):
        raise ValueError("invalid controlled prefix workload")
    options = dict(
        max_model_length=2048,
        max_num_sequences=concurrency,
        kv_cache_bytes=512 << 20,
        prefill_chunk_size=512,
        decode_mode="rowwise",
        prefix_cache_bytes=0 if mode == "disabled" else 64 << 20,
    )
    requests, trials = [], []
    with MlxEngine(model, **options) as engine:
        mx = engine.model.mx
        prompt = [
            (17 + i * 104729) % engine.model.config.vocab_size for i in range(length)
        ]
        for trial in range(-warmups, repetitions):
            engine.clear_prefix_cache()
            if mode == "primed":
                # Cache preparation is outside the measured interval, disclosed
                # below; it is work, not an amortization-free speedup.
                engine.generate(prompt, 1)
            baseline_memory = int(mx.get_active_memory())
            mx.reset_peak_memory()
            before = engine.stats()
            metrics, wall_ms, peak = run_once(engine, [prompt] * concurrency, 32, [])
            if trial >= 0:
                requests.extend({**asdict(m), "trial": trial} for m in metrics)
                trials.append(
                    {
                        "trial": trial,
                        "seconds": wall_ms / 1000,
                        "wall_ms": wall_ms,
                        "peak_cache": peak,
                        "baseline_metal_bytes": baseline_memory,
                        "peak_metal_bytes": int(mx.get_peak_memory()),
                        "before": before,
                        "after": engine.stats(),
                    }
                )
        retained = engine.stats()
        engine.clear_prefix_cache()
        cleared = engine.stats()
        build = engine.build_info()
    elapsed = sum(t["seconds"] for t in trials)
    return {
        "schema_version": 1,
        "purpose": __doc__,
        "environment": {"native_build": build, **source_provenance()},
        "model": model_provenance(model),
        "configuration": {
            "mode": mode,
            "prompt_length": length,
            "concurrency": concurrency,
            "output_tokens": 32,
            "warmups": warmups,
            "repetitions": repetitions,
            "decode_mode": "rowwise",
            "prefill_chunk_size": 512,
            "workload": prompt,
            "prefix_cache_bytes": options["prefix_cache_bytes"],
            "priming_outside_timing": mode == "primed",
        },
        "requests": requests,
        "trials": trials,
        "aggregate": {
            "generated_tokens_per_second": concurrency * repetitions * 32 / elapsed,
            **{
                name: {
                    f"p{int(q * 100)}": percentile([r[name] for r in requests], q)
                    for q in (0.5, 0.95, 0.99)
                }
                for name in ("ttft_ms", "tpot_ms", "e2e_ms")
            },
        },
        "retained_cache_stats": retained,
        "cleared_stats": cleared,
        "scope": "Same deterministic full prompts reused, 5 warmups/3 trials, raw queue-inclusive request times. Priming excluded, cold insertion included. Rowwise contract only; no semantic quality, universal speedup, GPU utilization/power or hardware-counter claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("disabled", "cold", "primed"), required=True)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with exclusive_gpu_workflow():
        report = measure(args.model, args.mode, args.prompt_length, args.concurrency)
    save(args.output, report)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
