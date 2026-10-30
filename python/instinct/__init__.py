"""Python convenience layer for Instinct's CUDA and MLX backends."""

from __future__ import annotations

import platform
from pathlib import Path

try:
    from _instinct import Engine as CudaEngine
    from _instinct import SequenceState as CudaSequenceState
    from _instinct import build_info as cuda_build_info
except (
    ImportError
):  # CUDA extension is intentionally absent on Apple development hosts.
    CudaEngine = None
    CudaSequenceState = None
    cuda_build_info = None

from .backends.base import mlx_build_info
from .mlx_engine import MlxEngine
from .runtime import SequenceState as MlxSequenceState

DEFAULT_MODEL_PATH = "models/gemma-4-26b-a4b-mlx"


def create_engine(
    model_path: str | Path | None = None, *, backend: str = "auto", **kwargs: object
):
    """Construct a backend explicitly or choose the native available backend."""
    selected = backend.lower()
    if selected == "auto":
        selected = "cuda" if CudaEngine is not None else "mlx"
    if selected == "cuda":
        if CudaEngine is None:
            raise RuntimeError("the CUDA extension is not installed")
        # The default checkpoint is an MLX directory the CUDA engine can't read.
        if model_path is None:
            raise ValueError("the CUDA backend needs an exported .engine model path")
        return CudaEngine(model_path, **kwargs)
    if selected == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("the MLX backend requires Apple Silicon")
        return MlxEngine(
            DEFAULT_MODEL_PATH if model_path is None else model_path, **kwargs
        )
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
    "DEFAULT_MODEL_PATH",
    "Engine",
    "MlxEngine",
    "SequenceState",
    "build_info",
    "create_engine",
]
