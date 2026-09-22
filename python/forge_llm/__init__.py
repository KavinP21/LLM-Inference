"""Python convenience layer for Forge LLM's CUDA and MLX backends."""

from __future__ import annotations

import platform
from pathlib import Path

try:
    from _forge import Engine as CudaEngine
    from _forge import SequenceState as CudaSequenceState
    from _forge import build_info as cuda_build_info
except ImportError:  # CUDA extension is intentionally absent on Apple development hosts.
    CudaEngine = None
    CudaSequenceState = None
    cuda_build_info = None

from .backends.mlx import mlx_build_info
from .mlx_engine import MlxEngine, SequenceState as MlxSequenceState


def create_engine(model_path: str | Path, *, backend: str = "auto", **kwargs: object):
    """Construct a backend explicitly or choose the native available backend."""
    selected = backend.lower()
    if selected == "auto":
        selected = "cuda" if CudaEngine is not None else "mlx"
    if selected == "cuda":
        if CudaEngine is None:
            raise RuntimeError("the CUDA extension is not installed")
        return CudaEngine(model_path, **kwargs)
    if selected == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("the MLX backend requires Apple Silicon")
        return MlxEngine(model_path, **kwargs)
    raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'cuda', or 'mlx'")


if CudaEngine is not None:
    Engine = CudaEngine
    SequenceState = CudaSequenceState
    build_info = cuda_build_info
else:
    Engine = MlxEngine
    SequenceState = MlxSequenceState
    build_info = mlx_build_info

__all__ = [
    "CudaEngine",
    "Engine",
    "MlxEngine",
    "SequenceState",
    "build_info",
    "create_engine",
]
