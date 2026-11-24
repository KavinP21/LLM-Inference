"""A ceiling on the host memory the model workers can use together.

Every local worker holds its weights, a K/V arena of ``kv_cache_bytes`` and
the activations of one forward pass, plus the Python process around them. A
configuration whose workers couldn't all be loaded at once without passing
the ceiling is refused at startup; lazy workers count, since they can all be
running at the same moment.

Each MLX worker is also given that much as its MLX memory limit, and a small
cache of freed buffers. By default MLX keeps freed buffers up to the whole
memory limit, so several processes could otherwise sit on gigabytes each
that nothing uses.

The ceiling comes from the config's ``memory`` section, ``{"max_gb": 105}``
by default, in decimal gigabytes.

On an NVIDIA GPU the weights, K/V and activations live in the card's memory
instead, so a config for one also gives ``"gpu_gb"``: the workers' MLX limits
must fit in that together, and ``max_gb`` covers only their processes and
the app. A worker whose ``engine_options`` set ``memory_limit_bytes`` counts
that instead of the estimate (``instinct.nvidia`` sizes them for the card).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

GB = 10**9
DEFAULT_MAX_GB = 105
# One forward pass's activations: MoE expert intermediates and the bounded
# prefill score tiles (512 MB), with room to spare.
ACTIVATION_BYTES = int(2.5 * GB)
# The Python process itself: interpreter, tokenizer, HTTP server.
PROCESS_BYTES = int(0.8 * GB)
# The app server, its journal and the guest gateway.
HARNESS_BYTES = 1 * GB
# Freed MLX buffers each worker may keep for reuse.
CACHE_LIMIT_BYTES = 1 << 30


def weight_bytes(model: str | Path) -> int:
    path = Path(model)
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0  # Loading it fails with its own error.
    return sum(f.stat().st_size for f in path.glob("*.safetensors"))


def mlx_limit(worker: dict[str, Any]) -> int:
    """The MLX memory limit for one worker: weights, K/V and activations,
    unless its engine options name one."""
    explicit = (worker.get("engine_options") or {}).get("memory_limit_bytes")
    if explicit is not None:
        if type(explicit) is not int or explicit <= 0:
            raise ValueError("memory_limit_bytes must be a positive integer")
        return explicit
    return (
        weight_bytes(worker["model"])
        + int(worker.get("kv_cache_bytes", 0))
        + ACTIVATION_BYTES
    )


def budget(config: dict[str, Any]) -> dict[str, Any]:
    """Each local worker's share and the total, against the ceiling."""
    settings = config.get("memory", {})
    if not isinstance(settings, dict) or settings.keys() - {"max_gb", "gpu_gb"}:
        raise ValueError(
            'memory takes {"max_gb": <decimal gigabytes>} and, for an NVIDIA '
            'GPU, "gpu_gb"'
        )
    limit_gb = _gigabytes(settings.get("max_gb", DEFAULT_MAX_GB), "max_gb", 4)
    gpu_gb = settings.get("gpu_gb")
    if gpu_gb is not None:
        gpu_gb = _gigabytes(gpu_gb, "gpu_gb", 2)
    workers, gpu = {}, {}
    for index, worker in enumerate(config["workers"]):
        if worker.get("type", "process") not in {"process", "local"}:
            continue  # Remote workers use another machine's memory.
        name = worker.get("name", f"worker-{index}")
        if gpu_gb is None:
            workers[name] = mlx_limit(worker) + PROCESS_BYTES
        else:
            workers[name] = PROCESS_BYTES
            gpu[name] = mlx_limit(worker)
    total = HARNESS_BYTES + sum(workers.values())
    report = {
        "limit_bytes": int(limit_gb * GB),
        "total_bytes": total,
        "workers": workers,
    }
    if gpu_gb is not None:
        report["gpu"] = {
            "limit_bytes": int(gpu_gb * GB),
            "total_bytes": sum(gpu.values()),
            "workers": gpu,
        }
    return report


def _gigabytes(value: Any, name: str, least: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"memory.{name} must be a number")
    if not least <= value <= 4096:
        raise ValueError(f"memory.{name} must be between {least:g} and 4096")
    return value


def check(config: dict[str, Any]) -> dict[str, Any]:
    report = budget(config)
    if report["total_bytes"] > report["limit_bytes"]:
        parts = ", ".join(
            f"{name} {size / GB:.1f} GB" for name, size in report["workers"].items()
        )
        raise ValueError(
            f"the model workers could use {report['total_bytes'] / GB:.1f} GB "
            f"together ({parts}, plus {HARNESS_BYTES / GB:.0f} GB for the app), "
            f"over the {report['limit_bytes'] / GB:.0f} GB limit in memory.max_gb; "
            "lower kv_cache_bytes or drop a worker"
        )
    gpu = report.get("gpu")
    if gpu and gpu["total_bytes"] > gpu["limit_bytes"]:
        parts = ", ".join(
            f"{name} {size / GB:.1f} GB" for name, size in gpu["workers"].items()
        )
        raise ValueError(
            f"the model workers could use {gpu['total_bytes'] / GB:.1f} GB of GPU "
            f"memory together ({parts}), over the {gpu['limit_bytes'] / GB:.1f} GB "
            "in memory.gpu_gb; lower kv_cache_bytes or drop a worker"
        )
    return report
