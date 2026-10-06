from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from forge_llm.backends.int8_kernels import Int8LinearKernels
from forge_llm.format import write_engine
from forge_llm.mlx_engine import MlxEngine
from forge_llm.model_file import ModelFile
from forge_llm.quantization import (
    dequantize_per_channel,
    quantize_model,
    quantize_per_channel,
)
from test_quantization import tiny_artifact


@pytest.mark.parametrize(
    "rows,n,k", [(1, 7, 13), (2, 33, 127), (8, 128, 256), (16, 129, 513), (1, 896, 896)]
)
def test_int8_kernel_matches_independent_fp32_reference(rows, n, k):
    rng = np.random.default_rng(8)
    x = rng.normal(0, 0.3, (rows, k)).astype(np.float16)
    weights = rng.normal(0, 0.12, (n, k)).astype(np.float16)
    packed, scales = quantize_per_channel(weights)
    expected = (
        x.astype(np.float32)
        @ dequantize_per_channel(packed, scales).astype(np.float32).T
    )
    kernel = Int8LinearKernels(mx)
    # Strided input, tails and repeat launches check MLX's contiguous contract.
    inputs = mx.array(x.T).T, mx.array(packed), mx.array(scales)
    np.testing.assert_array_equal(
        np.asarray(kernel.reconstruct(inputs[1], inputs[2])),
        dequantize_per_channel(packed, scales),
    )
    for _ in range(3):
        actual = np.asarray(kernel.linear(*inputs), dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_quantized_model_matches_fp16_reconstruction_and_reclaims_cache(
    tmp_path: Path, family
):
    source, packed, reconstructed = [
        tmp_path / name for name in ["fp16.engine", "int8.engine", "dequant.engine"]
    ]
    tiny_artifact(source, family)
    quantize_model(source, packed)
    with ModelFile(packed) as file:
        tensors = {
            name: file.tensor_numpy(name).copy()
            for name in file.tensors
            if not name.endswith(".int8_scale")
        }
        for name, spec in file.quantization.items():
            tensors[name] = dequantize_per_channel(
                tensors[name], file.tensor_numpy(spec.scale_name)
            )
        write_engine(
            reconstructed,
            file.config,
            tensors,
            {"lm_head.weight": "model.embed_tokens.weight"},
        )
    options = {
        "max_model_length": 64,
        "kv_cache_bytes": 1 << 20,
        "prefill_chunk_size": 4,
    }
    prompt = [2, 7, 11, 5, 19, 23, 3, 29, 31, 13, 9, 8, 12, 3, 7, 1, 5]
    with MlxEngine(reconstructed, **options) as engine:
        expected_logits = np.array(engine.debug_prefill_logits(prompt))
        expected = engine.generate(prompt, 8)
    for mode in ["dequantize", "metal", "auto", "reconstruct"]:
        with MlxEngine(packed, int8_mode=mode, **options) as engine:
            actual_logits = np.array(engine.debug_prefill_logits(prompt))
            if mode == "reconstruct":
                np.testing.assert_array_equal(actual_logits, expected_logits)
            np.testing.assert_allclose(
                actual_logits, expected_logits, rtol=0.02, atol=0.002
            )
            assert engine.generate(prompt, 8) == expected
            ids = [engine.submit(prompt[:length], 8) for length in [15, 16, 17]]
            engine.cancel(ids[0])
            while (
                engine.stats()["scheduler"]["running"]
                or engine.stats()["scheduler"]["waiting"]
            ):
                engine.step()
            assert engine.stats()["kv_device_bytes"] == 0
            assert engine.stats()["kv_cache"]["allocated_blocks"] == 0
            assert engine.stats()["kv_cache"]["reserved_blocks"] == 0
            assert engine.stats()["quantized_matrices"] == 14
            assert all(
                engine.model.weights[name].dtype == mx.int8
                for name in engine.model.file.quantization
            )
        assert not engine.model.weights
        engine.close()  # Ownership teardown is idempotent.


def test_invalid_int8_dispatch_mode_is_rejected(tmp_path):
    path = tmp_path / "model.engine"
    tiny_artifact(path)
    with pytest.raises(ValueError, match="int8_mode"):
        MlxEngine(path, int8_mode="int4")


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_mixed_precision_reconstruction_matches_composed_mode(tmp_path, family):
    source, packed = tmp_path / "source.engine", tmp_path / "mixed.engine"
    tiny_artifact(source, family)
    quantize_model(source, packed, retain_fp16=["model.layers.0.mlp.down_proj.weight"])
    outputs, logits = [], []
    for mode in ["dequantize", "reconstruct"]:
        with MlxEngine(
            packed, int8_mode=mode, max_model_length=64, kv_cache_bytes=1 << 20
        ) as engine:
            assert (
                engine.model.weights["model.layers.0.mlp.down_proj.weight"].dtype
                == mx.float16
            )
            assert engine.stats()["quantized_matrices"] == 13
            logits.append(np.array(engine.debug_prefill_logits([2, 3, 5, 7, 11])))
            outputs.append(engine.generate([2, 3, 5, 7, 11], 16))
            assert engine.stats()["kv_device_bytes"] == 0
    np.testing.assert_array_equal(logits[0], logits[1])
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_mixed_precision_boundary_batch_outputs_and_cancellation(tmp_path, family):
    source, packed = tmp_path / "source.engine", tmp_path / "mixed.engine"
    tiny_artifact(source, family)
    quantize_model(source, packed, retain_fp16=["model.layers.0.mlp.down_proj.weight"])
    prompts = [
        [(i * 7) % 32 for i in range(length)] for length in [15, 16, 17, 31, 32, 33]
    ]
    options = {
        "max_model_length": 64,
        "kv_cache_bytes": 1 << 20,
        "prefill_chunk_size": 8,
    }
    with MlxEngine(packed, int8_mode="dequantize", **options) as independent:
        expected = [independent.generate(prompt, 8) for prompt in prompts]
    with MlxEngine(
        packed, int8_mode="reconstruct", max_num_sequences=8, **options
    ) as batched:
        cancelled = batched.submit([1, 3, 5], 16)
        requests = [batched.submit(prompt, 8) for prompt in prompts]
        batched.step()
        batched.cancel(cancelled)
        while any(
            not batched.scheduler.request(r).output
            or len(batched.scheduler.request(r).output) < 8
            for r in requests
        ):
            batched.step()
        assert [batched.scheduler.request(r).output for r in requests] == expected
        assert batched.stats()["kv_device_bytes"] == 0
        assert batched.stats()["kv_cache"]["reserved_blocks"] == 0
        assert batched.stats()["retained_fp16_projections"] == [
            "model.layers.0.mlp.down_proj.weight"
        ]
        assert 0 < batched.stats()["quantized_projection_fraction"] < 1


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_offline_calibration_policy_executes_without_held_out_evaluation(
    tmp_path, monkeypatch, family
):
    from types import SimpleNamespace

    pytest.importorskip("transformers")
    from forge_llm.calibrate_int8 import calibrate

    source, packed = tmp_path / "source.engine", tmp_path / "mixed.engine"
    tiny_artifact(source, family)
    calls = []

    class Tokenizer:
        def __call__(self, text, **_):
            calls.append(text)
            return SimpleNamespace(
                input_ids={
                    "calibration one": [2, 3, 5],
                    "calibration two": [7, 11, 13],
                    "guard only": [17, 19, 23],
                }[text]
            )

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *_, **__: Tokenizer()
    )
    report = calibrate(
        source,
        "offline-stub",
        ["calibration one", "calibration two"],
        ["guard only"],
        output_tokens=4,
        positions=(0, 1, 3),
        probe_count=2,
        fractions=(1, 0.5, 0.25),
    )
    assert calls == ["calibration one", "calibration two", "guard only"]
    assert len(report["ranking"]) == 14
    assert report["cache_reclaimed"]
    assert report["policy"]["quantized_projection_fraction"] >= 0.25
    assert all(
        case["prompt"] != "guard only"
        for trial in report["trials"]
        for case in trial["cases"]
    )
    quantize_model(source, packed, policy=report["policy"])
    with MlxEngine(
        packed, int8_mode="reconstruct", max_model_length=64, kv_cache_bytes=1 << 20
    ) as engine:
        assert (
            engine.generate([2, 3, 5], 4)
            == report["trials"][-1]["cases"][0]["candidate_tokens"]
        )
