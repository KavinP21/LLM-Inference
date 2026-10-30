"""Weights and checkpoint loading shared by the model adapters.

An adapter exposes ``mx``, ``model_type``, ``dtype``, ``vocab_size``,
``max_position_embeddings``, ``kv_groups`` (and ``state_groups`` if it has
recurrent layers), ``forward(batch)`` returning float32 logits (the last row
only for a prefill chunk) and ``close()``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


def mlx() -> Any:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "the MLX engine needs Apple silicon and the 'mlx' extra: pip install 'instinct[mlx]'"
        ) from exc
    return mx


def mlx_build_info() -> dict[str, object]:
    from importlib.metadata import version

    mx = mlx()
    device = mx.device_info()
    return {
        "backend": "mlx",
        "mlx_version": version("mlx"),
        "device": device.get("device_name", "Apple GPU"),
        "architecture": device.get("architecture", "unknown"),
    }


class Weights:
    """Named tensors; linear layers may be MLX affine-quantized.

    A quantized layer ``name`` has ``name.weight`` (packed uint32),
    ``name.scales`` and ``name.biases``, with its group size and bit width
    from the checkpoint config.
    """

    def __init__(
        self, mx: Any, tensors: dict[str, Any], quantization: dict[str, tuple[int, int]]
    ) -> None:
        self.mx = mx
        self.tensors = tensors
        self.quantization = quantization

    def __getitem__(self, name: str) -> Any:
        return self.tensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def linear(self, name: str, x: Any) -> Any:
        mx = self.mx
        weight = self.tensors[name + ".weight"]
        spec = self.quantization.get(name)
        if spec is None:
            out = x @ weight.T
        else:
            group_size, bits = spec
            out = mx.quantized_matmul(
                x,
                weight,
                self.tensors[name + ".scales"],
                self.tensors[name + ".biases"],
                transpose=True,
                group_size=group_size,
                bits=bits,
            )
        bias = self.tensors.get(name + ".bias")
        return out if bias is None else out + bias

    def embed(self, name: str, tokens: Any) -> Any:
        weight = self.tensors[name + ".weight"]
        spec = self.quantization.get(name)
        if spec is None:
            return weight[tokens]
        group_size, bits = spec
        return self.mx.dequantize(
            weight[tokens],
            self.tensors[name + ".scales"][tokens],
            self.tensors[name + ".biases"][tokens],
            group_size=group_size,
            bits=bits,
        )

    def nbytes(self) -> int:
        return sum(array.nbytes for array in self.tensors.values())

    def clear(self) -> None:
        self.tensors.clear()


def load_checkpoint(
    path: str | Path, keep: Callable[[str], bool] = lambda name: True
) -> tuple[dict[str, Any], Weights]:
    """Load an MLX checkpoint directory (``config.json`` plus safetensors).

    Weight names lose a leading ``language_model.``; ``keep`` filters them.
    """
    mx = mlx()
    directory = Path(path)
    config = json.loads((directory / "config.json").read_text())
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {directory}")
    tensors: dict[str, Any] = {}
    for file in files:
        for name, array in mx.load(str(file)).items():
            name = name.removeprefix("language_model.")
            if keep(name):
                tensors[name] = array
    settings = config.get("quantization") or {}
    default = (settings.get("group_size", 64), settings.get("bits", 4))
    quantization = {}
    for name in tensors:
        if name.endswith(".scales"):
            layer = name.removesuffix(".scales")
            override = settings.get("language_model." + layer) or settings.get(layer)
            quantization[layer] = (
                (override["group_size"], override["bits"])
                if isinstance(override, dict)
                else default
            )
    mx.eval(*tensors.values())
    return config, Weights(mx, tensors, quantization)


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """The language model's settings, merged over the top-level ones."""
    return {**config, **(config.get("text_config") or {})}
