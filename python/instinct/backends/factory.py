"""Model-family dispatch for MLX engine artifacts."""

from __future__ import annotations

from typing import Any

from ..model_file import ModelFile
from .base import MlxDecoderModel
from .gemma3 import MlxGemma3Model
from .mlx import MlxQwenModel

_ADAPTERS = {
    "qwen2": MlxQwenModel,
    "gemma3_text": MlxGemma3Model,
}


def create_mlx_model(model_path: str, **kwargs: Any) -> MlxDecoderModel:
    """Inspect one artifact and instantiate its registered family adapter."""
    model_file = ModelFile(model_path)
    try:
        adapter = _ADAPTERS.get(model_file.config.model_type)
        if adapter is None:
            raise ValueError(
                f"no MLX adapter for model type {model_file.config.model_type!r}"
            )
        return adapter(model_path, _model_file=model_file, **kwargs)
    except Exception:
        model_file.close()
        raise
