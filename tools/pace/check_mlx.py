"""Check each MLX op the engine relies on, on this machine's GPU.

Run before benchmarking on a new device. Each check compares against a float32
reference built from basic ops. The report lists versions, the device, every
check, and which implementation the engine will pick for each fused op (see
``instinct/backends/portable.py``). The exit status is nonzero if an op the
engine can't work without fails.

    python tools/pace/check_mlx.py --output mlx-ops.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mlx.core as mx
from instinct.backends import portable
from instinct.backends.base import gpu_backend
from instinct.paged_kv import quantize

F32 = mx.float32
results: dict[str, dict] = {}
# Ops with no plain fallback: the engine can't run without them.
REQUIRED = {
    "quantized_matmul",
    "gather_qmm",
    "indexing",
    "sampling",
    "int8_kv",
    "matmul_softmax",
}


def check(name: str):
    def run(fn):
        started = time.perf_counter()
        try:
            detail = fn()
            results[name] = {
                "ok": True,
                "ms": round((time.perf_counter() - started) * 1e3, 1),
            }
            if detail:
                results[name]["detail"] = detail
        except Exception as exc:  # noqa: BLE001 - the report is the point.
            results[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "trace": traceback.format_exc(limit=3)[-800:],
            }
        return fn

    return run


def close(a, b, tolerance):
    a, b = a.astype(F32), b.astype(F32)
    mx.eval(a, b)
    error = float(mx.abs(a - b).max())
    scale = float(mx.abs(b).max()) + 1e-6
    assert error <= tolerance * scale, f"max error {error:.3g} vs scale {scale:.3g}"
    return round(error / scale, 6)


@check("matmul_softmax")
def _():
    a = mx.random.normal((64, 128)).astype(mx.bfloat16)
    b = mx.random.normal((128, 32)).astype(mx.bfloat16)
    out = mx.softmax(a @ b, axis=-1)
    return close(out, mx.softmax(a.astype(F32) @ b.astype(F32), axis=-1), 2e-2)


@check("quantized_matmul")
def _():
    errors = {}
    for bits in (4, 8):  # Both checkpoints: affine, group 64; Gemma has some 8-bit.
        w = mx.random.normal((768, 512)).astype(mx.bfloat16)
        packed, scales, biases = mx.quantize(w, group_size=64, bits=bits)
        dense = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
        for rows in (1, 7, 300):
            x = mx.random.normal((rows, 512)).astype(mx.bfloat16)
            out = mx.quantized_matmul(
                x, packed, scales, biases, transpose=True, group_size=64, bits=bits
            )
            errors[f"{bits}bit_{rows}rows"] = close(
                out, x.astype(F32) @ dense.T.astype(F32), 3e-2
            )
    return errors


@check("gather_qmm")
def _():
    experts, hidden, inner, top = 16, 256, 128, 4
    w = mx.random.normal((experts, inner, hidden)).astype(mx.bfloat16)
    packed, scales, biases = mx.quantize(w, group_size=64, bits=4)
    dense = mx.dequantize(packed, scales, biases, group_size=64, bits=4).astype(F32)
    errors = {}
    for tokens in (1, 3, 40):  # Decode batches and a prefill chunk.
        x = mx.random.normal((tokens, 1, 1, hidden)).astype(mx.bfloat16)
        indices = mx.random.randint(0, experts, (tokens, top)).astype(mx.uint32)
        # Unsorted, as decode calls it.
        out = mx.gather_qmm(
            x,
            packed,
            scales,
            biases,
            rhs_indices=indices,
            transpose=True,
            group_size=64,
            bits=4,
        )
        expected = x.astype(F32) @ dense[indices].swapaxes(-1, -2)
        errors[f"{tokens}_tokens"] = close(out, expected, 3e-2)
        # Sorted by expert, as prefill calls it.
        flat = indices.flatten()
        order = mx.argsort(flat)
        rows = x.flatten(0, -3)[order // top]
        out = mx.gather_qmm(
            rows,
            packed,
            scales,
            biases,
            rhs_indices=flat[order],
            transpose=True,
            group_size=64,
            bits=4,
            sorted_indices=True,
        )
        expected = rows.astype(F32) @ dense[flat[order]].swapaxes(-1, -2)
        errors[f"{tokens}_tokens_sorted"] = close(out, expected, 3e-2)
    return errors


@check("indexing")
def _():
    arena = mx.zeros((64, 2, 16), dtype=mx.bfloat16)
    rows = mx.array([3, 9, 40], mx.int32)
    values = mx.random.normal((3, 2, 16)).astype(mx.bfloat16)
    arena[rows] = values
    close(arena[rows], values, 0)
    table = mx.array([[5, 2, 7], [1, 0, 3]], mx.int32)
    picked = mx.take_along_axis(table, mx.array([[2, 0], [1, 1]]), axis=1)
    assert picked.tolist() == [[7, 5], [0, 0]]
    return None


@check("sampling")
def _():
    logits = mx.random.normal((4, 1000))
    top = mx.argpartition(-logits, kth=19, axis=-1)[:, :20]
    assert top.shape == (4, 20)
    draws = mx.random.categorical(logits, axis=-1, key=mx.random.key(1))
    assert draws.shape == (4,)
    return None


@check("int8_kv")
def _():
    x = mx.random.normal((32, 2, 128)).astype(mx.bfloat16)
    q, scale = quantize(mx, x)
    restored = q.astype(F32) * scale[..., None].astype(F32)
    return close(restored, x, 2e-2)


def _fail():
    raise AssertionError("fused op disagrees with the plain version")


# Checked at import, like the ops above.
for name, probe in portable._PROBES.items():
    check(f"fused_{name}")(lambda probe=probe: bool(probe(mx)) or _fail())


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = mx.device_info()
    report = {
        "python": platform.python_version(),
        "packages": {
            name: _version(name)
            for name in (
                "mlx",
                "mlx-cuda-12",
                "mlx-cuda-13",
                "mlx-metal",
                "transformers",
            )
        },
        "gpu_backend": gpu_backend(mx),
        "device": {k: v for k, v in device.items() if isinstance(v, (str, int, float))},
        "checks": results,
        "engine_picks": portable.report(mx),
    }
    missing = sorted(n for n in REQUIRED if not results.get(n, {}).get("ok"))
    report["ok"] = not missing and report["gpu_backend"] is not None
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if report["gpu_backend"] is None:
        raise SystemExit("MLX can't see a GPU (is this a GPU job, run with --nv?)")
    if missing:
        raise SystemExit(f"required ops failed: {', '.join(missing)}")


if __name__ == "__main__":
    main()
