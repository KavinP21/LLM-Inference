"""Portable dispatch checks: no MLX import or device initialization."""

import numpy as np
import pytest
from instinct.backends.mlx import MlxQwenModel
from instinct.mlx_engine import MlxEngine


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
