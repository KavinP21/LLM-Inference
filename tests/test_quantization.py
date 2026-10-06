from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from forge_llm.format import HEADER, ModelConfig, QuantizationSpec, write_engine
from forge_llm.model_file import ModelFile
from forge_llm.quantization import quantize_model, quantize_per_channel


def tiny_artifact(path: Path, family: str = "qwen2") -> None:
    config = ModelConfig(32, 8, 16, 2, 2, 1, 64, 2, 10_000.0, 1e-6)
    if family == "gemma3_text":
        from dataclasses import replace

        config = replace(
            config,
            model_type=family,
            head_dim=4,
            sliding_window=4,
            sliding_window_pattern=2,
            activation="gelu_pytorch_tanh",
            embedding_scale=8**0.5,
            norm_weight_offset=1.0,
        )
    random = np.random.default_rng(17)
    tensors = {
        "model.embed_tokens.weight": random.normal(0, 0.12, (32, 8)).astype(np.float16),
        "model.norm.weight": np.ones(8, dtype=np.float16),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        norms = ["input_layernorm", "post_attention_layernorm"]
        if family == "gemma3_text":
            norms += ["pre_feedforward_layernorm", "post_feedforward_layernorm"]
            for name in ["q_norm", "k_norm"]:
                tensors[prefix + f"self_attn.{name}.weight"] = np.zeros(
                    4, dtype=np.float16
                )
        for norm in norms:
            tensors[prefix + norm + ".weight"] = np.full(
                8, 0 if family == "gemma3_text" else 1, dtype=np.float16
            )
        for group, name, shape in [
            ("self_attn", "q", (8, 8)),
            ("self_attn", "k", (4, 8)),
            ("self_attn", "v", (4, 8)),
            ("self_attn", "o", (8, 8)),
            ("mlp", "gate", (16, 8)),
            ("mlp", "up", (16, 8)),
            ("mlp", "down", (8, 16)),
        ]:
            tensors[prefix + f"{group}.{name}_proj.weight"] = random.normal(
                0, 0.12, shape
            ).astype(np.float16)
            if family == "qwen2" and name in {"q", "k", "v"}:
                tensors[prefix + f"{group}.{name}_proj.bias"] = np.zeros(
                    shape[0], dtype=np.float16
                )
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    write_engine(path, config, tensors, {"lm_head.weight": "model.embed_tokens.weight"})


def test_channel_quantizer_error_bound_and_zero_channels():
    weights = np.random.default_rng(7).normal(size=(17, 131)).astype(np.float16)
    weights[0] = 0
    packed, scales = quantize_per_channel(weights)
    assert packed.dtype == np.int8 and scales.dtype == np.float32
    assert np.min(packed) >= -127 and np.max(packed) <= 127
    assert scales[0] == 1 and not np.any(packed[0])
    error = np.abs(weights.astype(np.float32) - packed * scales[:, None])
    assert np.all(error <= scales[:, None] * 0.5001)
    packed, _ = quantize_per_channel(np.array([[127, 0.5, 1.5, 2.5, -0.5, -1.5]]))
    np.testing.assert_array_equal(packed, [[127, 0, 2, 2, 0, -2]])


@pytest.mark.parametrize(
    "bad", [np.empty((0, 4)), np.ones(4), np.array([[np.nan]]), np.array([[np.inf]])]
)
def test_quantizer_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        quantize_per_channel(bad)


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_quantized_conversion_round_trip(tmp_path, family):
    source, first, second = [
        tmp_path / name for name in ["fp16.engine", "int8.engine", "copy.engine"]
    ]
    tiny_artifact(source, family)
    result = quantize_model(source, first)
    quantize_model(source, second)
    assert first.read_bytes() == second.read_bytes()
    assert result["quantized_matrices"] == 14
    with ModelFile(first) as file:
        assert file.version == 3 and len(file.quantization) == 14
        assert (
            file.tensor_info("lm_head.weight").offset
            == file.tensor_info("model.embed_tokens.weight").offset
        )
        for name, spec in file.quantization.items():
            assert file.tensor_info(name).dtype == np.int8
            assert file.tensor_info(spec.scale_name).dtype == np.float32
    with pytest.raises(FileExistsError):
        quantize_model(source, first)
    with pytest.raises(ValueError, match="overwrite"):
        quantize_model(source, source)
    with pytest.raises(ValueError, match="FP16 source"):
        quantize_model(first, tmp_path / "requant.engine")


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "dtype",
        "shape",
        "zero",
        "nan",
        "range",
        "unregistered",
        "scheme",
        "axis",
    ],
)
def test_quantized_reader_rejects_malformed_artifacts(tmp_path, case):
    path = tmp_path / "bad.engine"
    weights = np.ones((3, 4), dtype=np.int8)
    scales = np.ones(3, dtype=np.float32)
    specs = {"weight": QuantizationSpec("scales")}
    if case == "range":
        weights[0, 0] = -128
    if case == "zero":
        scales[0] = 0
    if case == "nan":
        scales[0] = np.nan
    if case == "shape":
        scales = np.ones(4, dtype=np.float32)
    if case == "dtype":
        weights = weights.astype(np.float16)
    if case == "scheme":
        specs["weight"] = QuantizationSpec("scales", scheme=2)
    if case == "axis":
        specs["weight"] = QuantizationSpec("scales", axis=1)
    tensors = {"weight": weights, "scales": scales}
    if case == "missing":
        del tensors["scales"]
    if case == "unregistered":
        specs = {}
    write_engine(
        path,
        ModelConfig(8, 4, 8, 1, 1, 1, 32, 2, 10000, 1e-6),
        tensors,
        quantization=specs,
    )
    with pytest.raises((ValueError, KeyError)):
        ModelFile(path)
    inspector = os.environ.get("FORGE_INSPECT_MODEL")
    if inspector:
        assert (
            subprocess.run(
                [inspector, path], capture_output=True, check=False
            ).returncode
            != 0
        )


def test_quantized_metadata_truncation_is_checked(tmp_path):
    path = tmp_path / "truncated.engine"
    tiny_artifact(tmp_path / "source.engine")
    quantize_model(tmp_path / "source.engine", path)
    payload = bytearray(path.read_bytes())
    header = list(HEADER.unpack_from(payload))
    header[2] -= 1
    header[5] = hashlib.sha256(payload[HEADER.size : HEADER.size + header[2]]).digest()
    payload[: HEADER.size] = HEADER.pack(*header)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="truncated quantization"):
        ModelFile(path)


def test_cpp_inspector_accepts_quantized_family(tmp_path):
    inspector = os.environ.get("FORGE_INSPECT_MODEL")
    if not inspector:
        pytest.skip("set FORGE_INSPECT_MODEL to cross-check the C++ artifact reader")
    for family in ["qwen2", "gemma3_text"]:
        source, output = (
            tmp_path / f"{family}.engine",
            tmp_path / f"{family}-int8.engine",
        )
        tiny_artifact(source, family)
        quantize_model(source, output)
        result = subprocess.run(
            [inspector, output], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "Forge LLM v3" in result.stdout
