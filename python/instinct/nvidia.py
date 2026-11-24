"""Running Instinct on an NVIDIA GPU: what fits on the card, and a check.

``python -m instinct.nvidia check`` reports the GPU, MLX's CUDA backend and
whether each custom CUDA kernel passes its self-test at the shapes Gemma 4
and Qwen3.5 use. It loads no model, so it takes seconds.

``python -m instinct.nvidia profile --output PATH`` writes a worker
configuration sized to the card's memory:

- **Under 6 GB**: refused; Qwen3.5 4B needs 2.4 GB for weights alone.
- **6 to 20 GB** (an RTX 3070 Ti's 8 GB, for example): Qwen3.5 4B for the
  main agent and the sub-agents. Gemma 4 26B's experts alone are 12.9 GB.
- **20 to 40 GB**: Gemma 4 26B for both, or Qwen if Gemma just misses.
- **40 GB or more**: Gemma 4 for the main agent and Qwen3.5 for the
  sub-agents, as on a Mac.

Each worker gets an explicit MLX memory limit (weights, K/V, recurrent state
slots and activations), and the config's ``memory.gpu_gb`` holds the total
to the card minus room for the desktop and CUDA's own context, so a config
that couldn't fit is refused at startup instead of failing mid-reply.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .agents.backends import memory

GB = 10**9
GIB = 1 << 30
# The desktop (Windows draws it with the same GPU under WSL), CUDA's context
# and cuBLAS workspaces, none of which MLX counts.
RESERVED_BYTES = int(1.5 * GB)
QWEN = "models/qwen3.5-4b-mlx"
GEMMA = "models/gemma-4-26b-a4b-mlx"
# Weights the engine loads, for before the models are downloaded. Qwen's
# vision tower isn't loaded.
QWEN_BYTES = int(2.4 * GB)
GEMMA_BYTES = int(15.6 * GB)
QWEN_ACTIVATION_BYTES = 1 * GB
GEMMA_ACTIVATION_BYTES = memory.ACTIVATION_BYTES
# One recurrent-state slot of Qwen3.5-4B: 24 GatedDeltaNet layers, each a
# 32 x 128 x 128 float32 memory plus a small convolution window.
QWEN_SLOT_BYTES = 52_000_000
# K/V bytes per token of context in full-attention layers. Qwen: 8 layers,
# 4 heads of 256, K and V at 8 bits plus fp16 scales. Gemma: 5 layers, 2
# heads of 512, at 16 bits. Gemma's sliding layers take up to half the
# arena, the rest is for full-context tokens.
QWEN_TOKEN_BYTES = 8 * 4 * (2 * 256 + 2 * 2)
GEMMA_TOKEN_BYTES = 5 * 2 * 2 * 512 * 2
LENGTHS = (262144, 131072, 65536, 32768, 16384, 8192)


def gpu_info() -> dict[str, Any] | None:
    """The first NVIDIA GPU, from ``nvidia-smi``, or None."""
    smi = shutil.which("nvidia-smi") or (
        "/usr/lib/wsl/lib/nvidia-smi"
        if os.access("/usr/lib/wsl/lib/nvidia-smi", os.X_OK)
        else None
    )
    if smi is None:
        return None
    try:
        output = subprocess.run(
            [
                smi,
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_smi(output)


def parse_smi(output: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, mib, capability, driver = parts
        try:
            total = int(float(mib)) * (1 << 20)
        except ValueError:
            continue
        return {
            "name": name,
            "memory_bytes": total,
            "compute_capability": capability,
            "driver": driver,
        }
    return None


def host_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def in_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def _weights(root: Path, model: str, fallback: int) -> int:
    return memory.weight_bytes(root / model) or fallback


def _text_weights(root: Path, model: str, fallback: int) -> int:
    """Bytes of a checkpoint's tensors outside its vision tower."""
    total = 0
    for file in sorted((root / model).glob("*.safetensors")):
        try:
            with file.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(min(size, 100_000_000)))
        except (OSError, ValueError, struct.error):
            return fallback
        for name, entry in header.items():
            if name != "__metadata__" and not name.startswith("vision_tower."):
                begin, end = entry["data_offsets"]
                total += end - begin
    return total or fallback


def _length(tokens: int) -> int:
    for length in LENGTHS:
        if tokens >= length:
            return length
    raise ValueError("too little memory left for the K/V cache")


def _qwen(root: Path, group: str, budget: int, *, pending: int, entries: int, cap: int):
    weights = _text_weights(root, QWEN, QWEN_BYTES)
    states = (pending + entries) * QWEN_SLOT_BYTES
    fixed = weights + states + QWEN_ACTIVATION_BYTES
    kv = min(cap, budget - fixed)
    if kv < int(0.75 * GB):
        raise ValueError(
            f"Qwen3.5 4B needs about {(fixed + 0.75 * GB) / GB:.1f} GB of GPU memory"
        )
    length = _length(kv // QWEN_TOKEN_BYTES)
    return {
        "type": "process",
        "name": "qwen",
        "group": group,
        "model": str(root / QWEN),
        "tokenizer": str(root / QWEN),
        "backend": "mlx",
        "max_model_length": length,
        "kv_cache_bytes": int(kv),
        "engine_options": {
            "kv_bits": 8,
            "prefix_cache_entries": entries,
            "memory_limit_bytes": int(fixed + kv),
        },
        "request_timeout": 600,
        "native_tool_calls": True,
        "max_pending": pending,
    }


def _gemma(root: Path, budget: int, *, pending: int, cap: int):
    weights = _weights(root, GEMMA, GEMMA_BYTES)
    fixed = weights + GEMMA_ACTIVATION_BYTES
    kv = min(cap, budget - fixed)
    if kv < 2 * GB:
        raise ValueError(
            f"Gemma 4 26B needs about {(fixed + 2 * GB) / GB:.1f} GB of GPU memory"
        )
    length = _length(kv // 2 // GEMMA_TOKEN_BYTES)
    return {
        "type": "process",
        "name": "gemma-4",
        "group": "driver",
        "model": str(root / GEMMA),
        "tokenizer": str(root / GEMMA),
        "backend": "mlx",
        "max_model_length": length,
        "kv_cache_bytes": int(kv),
        "engine_options": {"memory_limit_bytes": int(fixed + kv)},
        "request_timeout": 600,
        "native_tool_calls": True,
        "max_pending": pending,
    }


def profile(
    gpu_bytes: int, *, root: str | Path = ".", host_bytes: int | None = None
) -> dict[str, Any]:
    """A worker configuration for an NVIDIA card with ``gpu_bytes`` of memory."""
    root = Path(root)
    budget = gpu_bytes - RESERVED_BYTES
    if gpu_bytes < 6 * GB:
        raise ValueError(
            f"this GPU has {gpu_bytes / GB:.1f} GB; Instinct needs at least 6 GB"
        )
    gemma = None
    if 20 * GB <= gpu_bytes < 40 * GB:
        try:
            gemma = _gemma(root, budget, pending=4, cap=24 * GIB)
        except ValueError:
            pass  # Just short of Gemma: run Qwen instead.
    if gpu_bytes < 40 * GB and gemma is None:
        pending = 2 if gpu_bytes < 7.5 * GB else 4 if gpu_bytes < 12 * GB else 8
        workers = [
            _qwen(
                root,
                "driver",
                budget,
                pending=pending,
                entries=2 * pending,
                cap=8 * GIB,
            )
        ]
        subagent = "driver"
    elif gemma is not None:
        workers = [gemma]
        subagent = "driver"
    else:
        qwen_fixed = (
            _text_weights(root, QWEN, QWEN_BYTES)
            + 32 * QWEN_SLOT_BYTES
            + QWEN_ACTIVATION_BYTES
        )
        gemma_fixed = _weights(root, GEMMA, GEMMA_BYTES) + GEMMA_ACTIVATION_BYTES
        spare = budget - qwen_fixed - gemma_fixed
        gemma_kv = min(24 * GIB, int(0.7 * spare))
        gemma = _gemma(root, gemma_fixed + gemma_kv, pending=16, cap=24 * GIB)
        qwen = _qwen(
            root,
            "subagent",
            budget - gemma["engine_options"]["memory_limit_bytes"],
            pending=16,
            entries=16,
            cap=8 * GIB,
        )
        qwen.update({"lazy": True, "idle_seconds": 600})
        workers = [gemma, qwen]
        subagent = "subagent"
    driver = workers[0]
    context = int(driver["max_model_length"] * 0.95)
    concurrent = min(8, driver["max_pending"])
    host_gb = max(4, int((host_bytes or 16 * GB) / GB))
    return {
        "workers": workers,
        "runtime": {
            "driver_group": "driver",
            "subagent_group": subagent,
            "max_agents": 6,
            "max_depth": 1,
            "max_concurrent_generations": concurrent,
            "max_output_tokens": min(8192, driver["max_model_length"] // 8),
            "max_steps_per_agent": 200,
            "max_total_tokens": 50_000_000,
            "max_context_tokens": context,
            "deadline_seconds": 14400,
            "generation_timeout_seconds": 600,
            "tool_timeout_seconds": 390,
            "completion_timeout_seconds": 390,
            "max_tool_result_chars": 8000,
        },
        "memory": {
            "max_gb": host_gb,
            "gpu_gb": math.ceil(budget / GB * 100) / 100,
        },
    }


def models_for(gpu_bytes: int) -> list[str]:
    """``prepare_models`` names of the checkpoints a profile would load."""
    names = {QWEN: "qwen3.5-4b", GEMMA: "gemma-4-26b-a4b"}
    chosen = profile(gpu_bytes, root=Path("."))
    return [names[str(Path(w["model"]))] for w in chosen["workers"]]


# --- check --------------------------------------------------------------------

# The attention and recurrence shapes the two models run, as
# (label, head dim, query heads, kv heads, window, 8-bit K/V).
ATTENTION_SHAPES = (
    ("Qwen3.5 attention, 8-bit K/V", 256, 16, 4, 0, True),
    ("Qwen3.5 attention, 16-bit K/V", 256, 16, 4, 0, False),
    ("Gemma 4 sliding attention", 256, 16, 8, 1024, False),
    ("Gemma 4 full attention", 512, 16, 2, 0, False),
)


def kernel_report(mx: Any) -> list[dict[str, Any]]:
    """Each custom kernel's self-test at the models' shapes."""
    from .backends import cuda_kernels
    from .mlx_engine import PAGE_SIZE

    results = []
    for label, dim, q_heads, kv_heads, window, quantized in ATTENTION_SHAPES:
        started = time.perf_counter()
        try:
            ok = cuda_kernels._check_decode(
                mx,
                mx.bfloat16,
                mx.int8 if quantized else mx.bfloat16,
                head_dim=dim,
                q_heads=q_heads,
                kv_heads=kv_heads,
                page_size=PAGE_SIZE,
                window=window,
                softcap=0.0,
                quantized=quantized,
            )
            error = None if ok else "disagreed with the portable version"
        except Exception as exc:  # noqa: BLE001 - reported, not raised.
            ok, error = False, f"{type(exc).__name__}: {exc}"[:400]
        results.append(_result(label, ok, error, started))
    started = time.perf_counter()
    try:
        ok = cuda_kernels._check_gated_delta(
            mx,
            (mx.bfloat16, mx.bfloat16, mx.float32),
            k_dim=128,
            v_dim=128,
            k_heads=16,
            v_heads=32,
        )
        error = None if ok else "disagreed with the portable version"
    except Exception as exc:  # noqa: BLE001
        ok, error = False, f"{type(exc).__name__}: {exc}"[:400]
    results.append(_result("Qwen3.5 gated delta rule", ok, error, started))
    return results


def _result(label: str, ok: bool, error: str | None, started: float):
    return {
        "kernel": label,
        "passed": ok,
        "error": error,
        "seconds": round(time.perf_counter() - started, 2),
    }


def check() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "wsl": in_wsl(),
        "python": platform.python_version(),
        "gpu": gpu_info(),
        "host_memory_gb": round((host_memory_bytes() or 0) / GB, 1),
    }
    try:
        from importlib.metadata import version

        import mlx.core as mx

        from .backends import portable
        from .backends.base import gpu_backend

        report["mlx"] = version("mlx")
        report["mlx_gpu_backend"] = gpu_backend(mx) or "none"
        report["kernels"] = portable.report(mx)
        if portable.custom_kernels(mx) == "cuda":
            report["cuda_kernels"] = kernel_report(mx)
    except Exception as exc:  # noqa: BLE001 - the report says what's missing.
        report["mlx_error"] = f"{type(exc).__name__}: {exc}"[:400]
    if report["gpu"]:
        try:
            chosen = profile(report["gpu"]["memory_bytes"])
            report["profile"] = [w["name"] for w in chosen["workers"]]
        except ValueError as exc:
            report["profile_error"] = str(exc)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m instinct.nvidia")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="report the GPU, MLX and the CUDA kernels")
    commands.add_parser(
        "models", help="print the prepare_models names this GPU's profile uses"
    )
    made = commands.add_parser("profile", help="write a config sized to this GPU")
    made.add_argument("--output", type=Path, required=True)
    made.add_argument(
        "--gpu-gb",
        type=float,
        help="size for this much GPU memory instead of asking nvidia-smi",
    )
    made.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    if args.command == "check":
        report = check()
        print(json.dumps(report, indent=2))
        failed = [k for k in report.get("cuda_kernels", []) if not k["passed"]]
        return 1 if failed or "mlx_error" in report else 0
    if args.command == "models":
        info = gpu_info()
        gpu_bytes = info["memory_bytes"] if info else 8 * GIB
        try:
            print(" ".join(models_for(gpu_bytes)))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        return 0
    if args.gpu_gb is not None:
        gpu_bytes = int(args.gpu_gb * GB)
    else:
        info = gpu_info()
        if info is None:
            print("No NVIDIA GPU found (nvidia-smi didn't answer).", file=sys.stderr)
            return 1
        gpu_bytes = info["memory_bytes"]
    try:
        config = profile(gpu_bytes, root=args.root, host_bytes=host_memory_bytes())
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    memory.check(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    models = ", ".join(f"{w['name']} ({w['group']})" for w in config["workers"])
    print(f"Wrote {args.output}: {models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
