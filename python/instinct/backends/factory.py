"""Pick the model adapter for a checkpoint directory or ``.engine`` file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dense import DenseModel


def load_model(path: str | Path) -> Any:
    location = Path(path)
    if location.is_dir():
        model_type = json.loads((location / "config.json").read_text()).get(
            "model_type"
        )
        raise ValueError(f"no MLX adapter for model type {model_type!r}")
    return DenseModel(location)
