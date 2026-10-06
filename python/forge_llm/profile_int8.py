"""Controlled linear A/B: resident FP16 vs ephemeral dequantization vs fused W8A16."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .backends.factory import create_mlx_model
from .backends.mlx import mlx_build_info
from .benchmark import model_provenance, source_provenance


def profile(
    model_path: Path,
    rows: list[int],
    warmups: int,
    repetitions: int,
    capture_path: Path | None = None,
) -> dict:
    if not rows or min(rows) <= 0 or warmups < 0 or repetitions <= 0:
        raise ValueError("invalid INT8 profiling dimensions")
    model = create_mlx_model(str(model_path), int8_mode="metal")
    mx = model.mx
    if not model.file.quantization:
        model.close()
        raise ValueError("profiling requires a quantized model artifact")
    measurements = []
    mx.random.seed(17)
    try:
        names = [
            name
            for name in model.file.quantization
            if name.startswith("model.layers.0.")
        ]
        for name in names:
            weight = model.weights[name]
            scale = model.weights[model.file.quantization[name].scale_name]

            def reconstruct(weight=weight, scale=scale):
                return (weight.astype(mx.float32) * scale[:, None]).astype(mx.float16)

            dense = reconstruct()
            mx.eval(dense)
            for count in rows:
                x = mx.random.normal((count, weight.shape[1])).astype(mx.float16)
                mx.eval(x)
                operations = {
                    "resident_fp16": lambda x=x, dense=dense: x @ dense.T,
                    "int8_dequantize": lambda x=x, reconstruct=reconstruct: (
                        x @ reconstruct().T
                    ),
                    "int8_fused_reconstruction": lambda x=x, weight=weight, scale=scale: (
                        x @ model.int8.reconstruct(weight, scale).T
                    ),
                }
                if count <= model.int8.max_rows:
                    operations["int8_metal"] = lambda x=x, weight=weight, scale=scale: (
                        model.int8.linear(x, weight, scale)
                    )
                expected = (
                    np.asarray(x, dtype=np.float32)
                    @ np.asarray(dense, dtype=np.float32).T
                )
                for mode, operation in operations.items():
                    for _ in range(warmups):
                        mx.eval(operation())
                    mx.clear_cache()
                    baseline = mx.get_active_memory()
                    mx.reset_peak_memory()
                    durations = []
                    for _ in range(repetitions):
                        started = time.perf_counter_ns()
                        output = operation()
                        mx.eval(output)
                        durations.append((time.perf_counter_ns() - started) / 1e6)
                    peak = int(mx.get_peak_memory())
                    actual = np.asarray(output, dtype=np.float32)
                    error = float(np.max(np.abs(actual - expected)))
                    relative = float(
                        np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                    )
                    measurements.append(
                        {
                            "weight": name,
                            "shape": list(weight.shape),
                            "rows": count,
                            "implementation": mode,
                            "durations_ms": durations,
                            "median_ms": statistics.median(durations),
                            "incremental_peak_active_bytes": max(0, peak - baseline),
                            "max_abs_error_vs_fp32_reconstruction": error,
                            "relative_l2_error_vs_fp32_reconstruction": relative,
                            "dense_weight_bytes": dense.nbytes,
                            "int8_weight_and_scale_bytes": weight.nbytes + scale.nbytes,
                        }
                    )
                    del output
                print(
                    json.dumps(
                        {
                            "weight": name,
                            "rows": count,
                            "median_ms": {
                                item["implementation"]: item["median_ms"]
                                for item in measurements
                                if item["weight"] == name and item["rows"] == count
                            },
                        }
                    ),
                    flush=True,
                )
        if capture_path:
            name = next(name for name in names if name.endswith("mlp.down_proj.weight"))
            weight = model.weights[name]
            scale = model.weights[model.file.quantization[name].scale_name]
            dense = (weight.astype(mx.float32) * scale[:, None]).astype(mx.float16)
            x = mx.random.normal((8, weight.shape[1])).astype(mx.float16)
            mx.eval(x, dense)
            # Compile/prime all three paths before the capture.
            mx.eval(model.int8.linear(x, weight, scale))
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            mx.metal.start_capture(str(capture_path))
            try:
                mx.eval(x @ dense.T)
                mx.eval(
                    x
                    @ (weight.astype(mx.float32) * scale[:, None]).astype(mx.float16).T
                )
                mx.eval(x @ model.int8.reconstruct(weight, scale).T)
                mx.eval(model.int8.linear(x, weight, scale))
            finally:
                mx.metal.stop_capture()
        return {
            "schema_version": 1,
            "purpose": "synchronized wall-time linear A/B, not GPU-only time",
            "environment": {
                "platform": platform.platform(),
                "device": mx.device_info(),
                "native_build": mlx_build_info(),
                **source_provenance(),
            },
            "model": model_provenance(model_path),
            "seed": 17,
            "warmups": warmups,
            "repetitions": repetitions,
            "capture": str(capture_path) if capture_path else None,
            "capture_shape": {"weight": name, "rows": 8} if capture_path else None,
            "measurements": measurements,
        }
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 128])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    result = profile(
        args.model, args.rows, args.warmups, args.repetitions, args.capture
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
