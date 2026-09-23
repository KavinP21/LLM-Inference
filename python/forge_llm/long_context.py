"""Reproducible MLX long-context execution and cache-integrity validation."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .benchmark import model_provenance
from .mlx_engine import MlxEngine
from .runtime import SequenceState


@dataclass(frozen=True)
class ContextObservation:
    context_tokens: int
    prompt_tokens: int
    output_tokens: int
    prefill_iterations: int
    decode_iterations: int
    first_token: int
    ttft_ms: float
    e2e_ms: float
    peak_pool_blocks: int
    peak_materialized_blocks: int
    peak_kv_bytes: int
    baseline_metal_bytes: int
    peak_metal_bytes: int
    peak_incremental_metal_bytes: int
    blocks_reclaimed: bool


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() or None


def _prompt(length: int, vocab_size: int) -> list[int]:
    # A deterministic full-vocabulary walk avoids tokenizer/network dependencies.
    return [int((17 + index * 104_729) % vocab_size) for index in range(length)]


def validate_contexts(
    model_path: Path,
    context_lengths: list[int],
    *,
    output_tokens: int,
    prefill_chunk_size: int,
    kv_cache_bytes: int,
) -> dict[str, object]:
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if not context_lengths or min(context_lengths) <= output_tokens:
        raise ValueError("every context length must exceed output_tokens")
    maximum = max(context_lengths)
    engine = MlxEngine(
        model_path,
        max_num_sequences=1,
        max_model_length=maximum,
        kv_cache_bytes=kv_cache_bytes,
        prefill_chunk_size=prefill_chunk_size,
    )
    mx = engine.model.mx
    observations: list[ContextObservation] = []
    try:
        for context_tokens in context_lengths:
            prompt_tokens = context_tokens - output_tokens
            engine.kv_store.reset_peak()
            engine.cache_pool.reset_peak()
            mx.clear_cache()
            baseline_metal_bytes = int(mx.get_active_memory())
            mx.reset_peak_memory()
            request_id = engine.submit(
                _prompt(prompt_tokens, engine.model.config.vocab_size),
                max_new_tokens=output_tokens,
                eos_token_ids=[],
            )
            started = time.perf_counter_ns()
            iterations = 0
            events = []
            first_event_ns = None
            while (
                engine.scheduler.request(request_id).state
                is not SequenceState.COMPLETED
            ):
                step_events = engine.step()
                iterations += 1
                if step_events:
                    if any(event.request_id != request_id for event in step_events):
                        raise RuntimeError(
                            "unexpected event during single-request validation"
                        )
                    if first_event_ns is None:
                        first_event_ns = time.perf_counter_ns()
                    events.extend(step_events)
            finished_ns = time.perf_counter_ns()
            if (
                len(events) != output_tokens
                or first_event_ns is None
                or not events[-1].finished
            ):
                raise RuntimeError(
                    "long-context request did not produce the expected output"
                )
            stats = engine.stats()
            prefill_iterations = math.ceil(prompt_tokens / prefill_chunk_size)
            decode_iterations = output_tokens - 1
            expected_iterations = prefill_iterations + decode_iterations
            if iterations != expected_iterations:
                raise RuntimeError("execution did not honor the expected step count")
            reclaimed = (
                stats["kv_cache"]["allocated_blocks"] == 0
                and stats["kv_cache"]["reserved_blocks"] == 0
                and stats["kv_materialized_blocks"] == 0
            )
            if not reclaimed:
                raise RuntimeError("K/V pages leaked after long-context completion")
            peak_metal_bytes = int(mx.get_peak_memory())
            observations.append(
                ContextObservation(
                    context_tokens=context_tokens,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    prefill_iterations=prefill_iterations,
                    decode_iterations=decode_iterations,
                    first_token=events[0].token,
                    ttft_ms=(first_event_ns - started) / 1e6,
                    e2e_ms=(finished_ns - started) / 1e6,
                    peak_pool_blocks=int(stats["kv_cache"]["peak_allocated_blocks"]),
                    peak_materialized_blocks=int(stats["kv_peak_materialized_blocks"]),
                    peak_kv_bytes=int(stats["kv_peak_device_bytes"]),
                    baseline_metal_bytes=baseline_metal_bytes,
                    peak_metal_bytes=peak_metal_bytes,
                    peak_incremental_metal_bytes=max(
                        0, peak_metal_bytes - baseline_metal_bytes
                    ),
                    blocks_reclaimed=reclaimed,
                )
            )
            print(json.dumps(asdict(observations[-1]), sort_keys=True), flush=True)
    finally:
        engine.close()

    return {
        "schema_version": 1,
        "purpose": "execution-and-resource-integrity; not a throughput benchmark",
        "model": model_provenance(model_path),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "native_build": engine.build_info(),
            "git_commit": _git_commit(),
        },
        "configuration": {
            "context_lengths": context_lengths,
            "output_tokens": output_tokens,
            "prefill_chunk_size": prefill_chunk_size,
            "kv_cache_bytes": kv_cache_bytes,
            "block_tokens": engine.block_tokens,
        },
        "observations": [asdict(item) for item in observations],
        "all_reclaimed": all(item.blocks_reclaimed for item in observations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate bounded-workspace MLX execution at long context lengths"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=2,
        help="use at least two to exercise paged decode attention after prefill",
    )
    parser.add_argument("--kv-cache-mib", type=int, default=512)
    args = parser.parse_args()
    result = validate_contexts(
        args.model,
        args.context_lengths,
        output_tokens=args.output_tokens,
        prefill_chunk_size=args.prefill_chunk_size,
        kv_cache_bytes=args.kv_cache_mib << 20,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
