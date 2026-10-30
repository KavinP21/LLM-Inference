"""Pick the model adapter for a checkpoint directory or ``.engine`` file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dense import DenseModel


def _adapters() -> dict[str, Any]:
    from .gemma4 import Gemma4Model

    return {"gemma4": Gemma4Model}


def load_model(path: str | Path) -> Any:
    location = Path(path)
    if not location.is_dir():
        return DenseModel(location)
    model_type = json.loads((location / "config.json").read_text()).get("model_type")
    adapter = _adapters().get(model_type)
    if adapter is None:
        raise ValueError(f"no MLX adapter for model type {model_type!r}")
    return adapter(location)
