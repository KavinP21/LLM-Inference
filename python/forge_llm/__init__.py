"""Python convenience layer for Forge LLM's CUDA and MLX backends."""

from __future__ import annotations

import json
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

DEFAULT_MODEL_PATH = "models/qwen3.5-4b-mlx"


def create_engine(
    model_path: str | Path | None = None, *, backend: str = "auto", **kwargs: object
):
    """Construct a backend explicitly or choose the native available backend."""
    selected = backend.lower()
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if selected == "auto":
        selected = "cuda" if CudaEngine is not None else "mlx"
    if selected == "cuda":
        if CudaEngine is None:
            raise RuntimeError("the CUDA extension is not installed")
        return CudaEngine(model_path, **kwargs)
    if selected == "mlx":
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("the MLX backend requires Apple Silicon")
        path = Path(model_path)
        if path.is_dir():
            with (path / "config.json").open("rb") as stream:
                config_data = stream.read((1 << 20) + 1)
            if len(config_data) > 1 << 20:
                raise ValueError("checkpoint configuration exceeds 1 MiB")
            config = json.loads(config_data)
            if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
                raise ValueError(
                    "directory model loading requires a Qwen3.5 checkpoint"
                )
            if kwargs.pop("max_num_sequences", 1) != 1:
                raise ValueError(
                    "Qwen3.5 uses one active request per replica; use a worker pool for concurrency"
                )
            from .qwen35_engine import Qwen35Engine

            return Qwen35Engine(path, **kwargs)
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
