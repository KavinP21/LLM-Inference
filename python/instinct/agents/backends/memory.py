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
    """The MLX memory limit for one worker: weights, K/V and activations."""
    return (
        weight_bytes(worker["model"])
        + int(worker.get("kv_cache_bytes", 0))
        + ACTIVATION_BYTES
    )


def budget(config: dict[str, Any]) -> dict[str, Any]:
    """Each local worker's share and the total, against the ceiling."""
    settings = config.get("memory", {})
    if not isinstance(settings, dict) or settings.keys() - {"max_gb"}:
        raise ValueError('memory takes only {"max_gb": <decimal gigabytes>}')
    limit_gb = settings.get("max_gb", DEFAULT_MAX_GB)
    if isinstance(limit_gb, bool) or not isinstance(limit_gb, (int, float)):
        raise ValueError("memory.max_gb must be a number")
    if not 4 <= limit_gb <= 4096:
        raise ValueError("memory.max_gb must be between 4 and 4096")
    workers = {}
    for index, worker in enumerate(config["workers"]):
        if worker.get("type", "process") not in {"process", "local"}:
            continue  # Remote workers use another machine's memory.
        name = worker.get("name", f"worker-{index}")
        workers[name] = mlx_limit(worker) + PROCESS_BYTES
    total = HARNESS_BYTES + sum(workers.values())
    return {
        "limit_bytes": int(limit_gb * GB),
        "total_bytes": total,
        "workers": workers,
    }


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
    return report
