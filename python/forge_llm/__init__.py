"""Python convenience layer for Forge LLM's CUDA and MLX backends."""

from __future__ import annotations

import platform
from pathlib import Path

try:
    from _forge import Engine as CudaEngine
    from _forge import SequenceState as CudaSequenceState
    from _forge import build_info as cuda_build_info
except (
    ImportError
):  # CUDA extension is intentionally absent on Apple development hosts.
    CudaEngine = None
    CudaSequenceState = None
    cuda_build_info = None

from .backends.mlx import mlx_build_info
from .mlx_engine import MlxEngine
from .mlx_engine import SequenceState as MlxSequenceState
from .speculative import SpeculativeEngine

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
        path = Path(DEFAULT_MODEL_PATH if model_path is None else model_path)
        if path.is_dir():
            # A checkpoint directory (Gemma 4, Qwen3.5, ...) runs through mlx-vlm.
            from .vlm_runner import VlmRunner

            options = dict(kwargs)
            if "max_model_length" in options:
                options["max_context_tokens"] = options.pop("max_model_length")
            return VlmRunner(path, **options)
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
    "DEFAULT_MODEL_PATH",
    "Engine",
    "MlxEngine",
    "SequenceState",
    "SpeculativeEngine",
    "build_info",
    "create_engine",
]
