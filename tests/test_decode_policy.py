"""Portable dispatch checks: no MLX import or device initialization."""

from types import SimpleNamespace

import numpy as np
import pytest
from forge_llm.backends.mlx import MlxQwenModel
from forge_llm.mlx_engine import MlxEngine


def test_invalid_decode_mode_rejected_before_device_initialization():
    with pytest.raises(ValueError, match="decode_mode"):
        MlxEngine("does-not-exist.engine", decode_mode="deterministic_magic")


@pytest.mark.parametrize("mode", ["batched", "rowwise"])
def test_decode_dispatch_does_not_change_prefill(mode):
    model = object.__new__(MlxQwenModel)
    model.decode_mode = mode
    calls = []

    def linear(x, name, bias, *, rowwise=False):
        calls.append((name, bias, rowwise))
        return x

    model._linear = linear
    values = np.zeros((4, 8), dtype=np.float16)
    model._decode_linear(values, "w", "b")
    model._linear(values, "w", "b")
    assert calls == [("w", "b", mode == "rowwise"), ("w", "b", False)]
    with pytest.raises(ValueError, match="rank two"):
        model._decode_linear(values.reshape(2, 2, 8), "w")


@pytest.mark.parametrize("mode", ["metal", "reconstruct", "dequantize"])
def test_rowwise_quantized_dispatch_reuses_reconstruction_and_splits_direct_kernel(
    mode,
):
    model = object.__new__(MlxQwenModel)
    model.decode_mode, model.int8_mode = "rowwise", mode
    model.mx = SimpleNamespace(
        concatenate=np.concatenate, float16=np.float16, float32=np.float32
    )
    w = np.ones((5, 8), dtype=np.int8)
    model.weights = {
        "w": w,
        "scale": np.ones(5, dtype=np.float32),
        "bias": np.ones(5, dtype=np.float16),
    }
    model.file = SimpleNamespace(
        quantization={"w": SimpleNamespace(scale_name="scale")}
    )
    calls = []

    def direct(x, weight, scale):
        calls.append(("direct", x.shape[0]))
        return x @ (weight * scale[:, None]).astype(np.float16).T

    def reconstruct(weight, scale):
        calls.append(("reconstruct",))
        return (weight * scale[:, None]).astype(np.float16)

    model.int8 = (
        None
        if mode == "dequantize"
        else SimpleNamespace(max_rows=16, linear=direct, reconstruct=reconstruct)
    )
    # More than 16 rows must still select the same direct kernel as an independent request.
    x = np.arange(17 * 8).reshape(17, 8).astype(np.float16)
    np.testing.assert_array_equal(
        model._decode_linear(x, "w", "bias"), x @ w.astype(np.float16).T + 1
    )
    if mode == "metal":
        assert calls == [("direct", 1)] * 17
    elif mode == "reconstruct":
        assert calls == [("reconstruct",)]
    else:
        assert not calls
