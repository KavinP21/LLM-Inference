from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from forge_llm.refined import DEFAULT_CONFIG, quantize_refined
from forge_llm.second_order import block_moments


def test_independent_readback_matches_source_activation_projection_error(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "refined_readback", root / "benchmarks/verify_refined_objectives.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    random = np.random.default_rng(991)
    inputs = random.normal(size=(101, 13))
    weight = random.normal(size=(17, 13)).astype(np.float16)
    moments = block_moments(inputs, 4)
    q, scales, metrics = quantize_refined(
        weight, moments, {**DEFAULT_CONFIG, "block_size": 4}
    )
    loss = mod.independent_loss(weight, q, scales, moments, 4)
    dense = (q.astype(np.float32) * scales[:, None]).astype(np.float16)
    error = weight.astype(np.float64) - dense.astype(np.float64)
    expected = sum(
        np.square(inputs[:, s : s + 4] @ error[:, s : s + 4].T).mean(axis=0)
        for s in range(0, 13, 4)
    )
    np.testing.assert_allclose(loss, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        loss.sum(), metrics["calibrated_block_objective"], rtol=1e-12
    )
