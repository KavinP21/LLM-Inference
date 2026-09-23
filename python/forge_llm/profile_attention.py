"""Controlled MLX attention A/B measurements for optimization decisions."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np

from .backends.mlx import MlxQwenModel, mlx_build_info
from .benchmark import model_provenance


@dataclass(frozen=True)
class AttentionMeasurement:
    implementation: str
    key_tokens: int
    query_tokens: int
    median_ms: float
    minimum_ms: float
    peak_incremental_metal_bytes: int
    cosine_vs_materialized: float
    max_abs_error_vs_materialized: float


def _measure(
    mx: object, operation: Callable[[], object], *, warmups: int, repetitions: int
) -> tuple[object, list[float], int]:
    output = None
    for _ in range(warmups):
        output = operation()
        mx.eval(output)
    mx.clear_cache()
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        durations.append((time.perf_counter_ns() - started) / 1e6)
    if output is None:
        raise RuntimeError("attention profiler produced no output")
    peak_incremental = max(0, int(mx.get_peak_memory()) - baseline)
    return output, durations, peak_incremental


def profile_attention(
    model_path: Path,
    key_lengths: list[int],
    *,
    query_tokens: int,
    tile_size: int,
    warmups: int,
    repetitions: int,
    capture_path: Path | None = None,
) -> dict[str, object]:
    if not key_lengths or min(key_lengths) <= 0:
        raise ValueError("attention key lengths must be positive")
    if min(query_tokens, tile_size, repetitions) <= 0 or warmups < 0:
        raise ValueError("invalid attention profiling dimensions")
    model = MlxQwenModel(
        str(model_path),
        max_model_length=max(key_lengths),
        attention_tile_size=tile_size,
    )
    mx = model.mx
    mx.random.seed(17)
    measurements: list[AttentionMeasurement] = []
    try:
        for key_tokens in key_lengths:
            current_queries = min(query_tokens, key_tokens)
            query_start = key_tokens - current_queries
            query = mx.random.normal(
                (
                    current_queries,
                    model.config.num_attention_heads,
                    model.head_dim,
                )
            ).astype(mx.float16)
            key = mx.random.normal(
                (
                    key_tokens,
                    model.config.num_key_value_heads,
                    model.head_dim,
                )
            ).astype(mx.float16)
            value = mx.random.normal(key.shape).astype(mx.float16)
            mx.eval(query, key, value)

            operations = {
                "materialized": partial(
                    model._attention, query, key, value, query_start
                ),
                "online_tiled": partial(
                    model._tiled_attention, query, key, value, query_start
                ),
                "mlx_fused_gqa": partial(model._fast_attention, query, key, value),
            }
            reference = None
            for name, operation in operations.items():
                output, durations, peak_incremental = _measure(
                    mx,
                    operation,
                    warmups=warmups,
                    repetitions=repetitions,
                )
                actual = np.asarray(output, dtype=np.float32)
                if reference is None:
                    reference = actual
                numerator = float(np.dot(reference.ravel(), actual.ravel()))
                denominator = float(
                    np.linalg.norm(reference.ravel()) * np.linalg.norm(actual.ravel())
                )
                measurements.append(
                    AttentionMeasurement(
                        implementation=name,
                        key_tokens=key_tokens,
                        query_tokens=current_queries,
                        median_ms=statistics.median(durations),
                        minimum_ms=min(durations),
                        peak_incremental_metal_bytes=peak_incremental,
                        cosine_vs_materialized=numerator / denominator,
                        max_abs_error_vs_materialized=float(
                            np.max(np.abs(reference - actual))
                        ),
                    )
                )
            print(
                json.dumps(
                    {
                        "key_tokens": key_tokens,
                        "measurements": [
                            asdict(item)
                            for item in measurements
                            if item.key_tokens == key_tokens
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if capture_path is not None:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            key_tokens = max(key_lengths)
            current_queries = min(query_tokens, key_tokens)
            query_start = key_tokens - current_queries
            query = mx.random.normal(
                (
                    current_queries,
                    model.config.num_attention_heads,
                    model.head_dim,
                )
            ).astype(mx.float16)
            key = mx.random.normal(
                (
                    key_tokens,
                    model.config.num_key_value_heads,
                    model.head_dim,
                )
            ).astype(mx.float16)
            value = mx.random.normal(key.shape).astype(mx.float16)
            mx.eval(query, key, value)
            try:
                mx.metal.start_capture(str(capture_path))
            except RuntimeError as exc:
                raise RuntimeError(
                    "Metal capture failed; relaunch with MTL_CAPTURE_ENABLED=1"
                ) from exc
            try:
                mx.eval(model._fast_attention(query, key, value))
            finally:
                mx.metal.stop_capture()
    finally:
        model.close()

    return {
        "schema_version": 1,
        "purpose": "controlled attention A/B; not an end-to-end throughput claim",
        "model": model_provenance(model_path),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "native_build": mlx_build_info(),
        },
        "configuration": {
            "key_lengths": key_lengths,
            "query_tokens": query_tokens,
            "tile_size": tile_size,
            "warmups": warmups,
            "repetitions": repetitions,
            "capture_path": str(capture_path) if capture_path is not None else None,
        },
        "measurements": [asdict(item) for item in measurements],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile MLX attention implementations"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-lengths", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    result = profile_attention(
        args.model,
        args.key_lengths,
        query_tokens=args.query_tokens,
        tile_size=args.tile_size,
        warmups=args.warmups,
        repetitions=args.repetitions,
        capture_path=args.capture,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
