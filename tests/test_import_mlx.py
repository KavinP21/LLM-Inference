from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from forge_llm.import_mlx import _validate_packed, import_mlx_checkpoint
from forge_llm.model_file import ModelFile


def numpy_decode(weight, scales, biases):
    values = ((weight[..., None] >> np.arange(0, 32, 4, dtype=np.uint32)) & 15).reshape(
        weight.shape[0], -1
    )
    values = values.reshape(weight.shape[0], -1, 64).astype(np.float32)
    return (
        (
            values * scales.astype(np.float32)[..., None]
            + biases.astype(np.float32)[..., None]
        )
        .reshape(weight.shape[0], -1)
        .astype(np.float16)
    )


def source_checkpoint(directory):
    from safetensors.numpy import save_file

    directory.mkdir()
    config = {
        "model_type": "qwen2",
        "vocab_size": 32,
        "hidden_size": 64,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "max_position_embeddings": 128,
        "eos_token_id": 2,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "use_sliding_window": False,
        "tie_word_embeddings": False,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text('{"test_tokenizer":true}')
    tensors = {}
    shapes = {
        "model.embed_tokens": (32, 64),
        "lm_head": (32, 64),
        "model.layers.0.self_attn.q_proj": (64, 64),
        "model.layers.0.self_attn.k_proj": (8, 64),
        "model.layers.0.self_attn.v_proj": (8, 64),
        "model.layers.0.self_attn.o_proj": (64, 64),
        "model.layers.0.mlp.gate_proj": (64, 64),
        "model.layers.0.mlp.up_proj": (64, 64),
        "model.layers.0.mlp.down_proj": (64, 64),
    }
    for prefix, shape in shapes.items():
        values = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) % 16
        packed = (
            (values.reshape(shape[0], -1, 8) << np.arange(0, 32, 4, dtype=np.uint32))
            .sum(axis=-1)
            .astype(np.uint32)
        )
        tensors[prefix + ".weight"] = packed
        tensors[prefix + ".scales"] = np.full(
            (shape[0], shape[1] // 64), 0.125, dtype=np.float16
        )
        tensors[prefix + ".biases"] = np.full(
            (shape[0], shape[1] // 64), -1, dtype=np.float16
        )
    for prefix in (
        "model.norm",
        "model.layers.0.input_layernorm",
        "model.layers.0.post_attention_layernorm",
    ):
        tensors[prefix + ".weight"] = np.ones(64, dtype=np.float16)
    for name, size in (("q", 64), ("k", 8), ("v", 8)):
        tensors[f"model.layers.0.self_attn.{name}_proj.bias"] = np.zeros(
            size, dtype=np.float16
        )
    save_file(tensors, directory / "model.safetensors")
    mapping = {name: "model.safetensors" for name in tensors}
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    return tensors


def test_import_round_trip_provenance_and_readonly_source(tmp_path):
    source, output = tmp_path / "source", tmp_path / "reconstructed.engine"
    original = source_checkpoint(source)
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    }
    result = import_mlx_checkpoint(source, output, _dequantize=numpy_decode)
    with ModelFile(output) as model:
        assert model.config.model_type == "qwen2"
        assert model.quantization == {}
        name = "model.layers.0.self_attn.q_proj.weight"
        np.testing.assert_array_equal(
            model.tensor_numpy(name),
            numpy_decode(
                original[name],
                original[name.removesuffix(".weight") + ".scales"],
                original[name.removesuffix(".weight") + ".biases"],
            ),
        )
    assert result["original_fp16_checkpoint"] is False
    assert result["parameter_origin"] == "reconstructed_from_mlx_affine_4bit"
    assert len(result["reconstructed_matrices"]) == 9
    assert all(
        info["sha256"] == before[name] for name, info in result["source_files"].items()
    )
    assert json.loads(output.with_suffix(".engine.json").read_text()) == result
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()
    }
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("which", ["engine", "manifest"])
def test_refuses_overwrite_of_either_artifact(tmp_path, which):
    source, output = tmp_path / "source", tmp_path / "output.engine"
    source_checkpoint(source)
    existing = output if which == "engine" else output.with_suffix(".engine.json")
    existing.write_text("preserve me")
    with pytest.raises(FileExistsError):
        import_mlx_checkpoint(source, output, _dequantize=numpy_decode)
    assert existing.read_text() == "preserve me"


@pytest.mark.parametrize(
    "quantization",
    [
        None,
        {"bits": 8, "group_size": 64},
        {"bits": 4, "group_size": 32},
        {"bits": 4, "group_size": 64, "mode": "nvfp4"},
    ],
)
def test_rejects_unknown_quantization_contract(tmp_path, quantization):
    source = tmp_path / "source"
    source_checkpoint(source)
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = quantization
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="quantization"):
        import_mlx_checkpoint(
            source, tmp_path / "output.engine", _dequantize=numpy_decode
        )


def test_index_and_dequantization_failure_leave_no_outputs(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output.engine"
    source_checkpoint(source)

    def bad(*args):
        return np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="invalid FP16"):
        import_mlx_checkpoint(source, output, _dequantize=bad)
    assert not output.exists()
    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"].pop("lm_head.weight")
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="ownership"):
        import_mlx_checkpoint(source, output, _dequantize=numpy_decode)
    assert not output.exists()


def test_failed_manifest_publication_removes_our_engine(tmp_path, monkeypatch):
    import forge_llm.import_mlx as importer

    source, output = tmp_path / "source", tmp_path / "output.engine"
    source_checkpoint(source)
    link = importer.os.link
    calls = []

    def fail_second(*args):
        calls.append(args)
        if len(calls) == 2:
            raise OSError("injected manifest publication failure")
        return link(*args)

    monkeypatch.setattr(importer.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        import_mlx_checkpoint(source, output, _dequantize=numpy_decode)
    assert not output.exists()
    assert not output.with_suffix(".engine.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_packed_shape_dtype_and_nonfinite_parameters_are_rejected():
    weight = np.zeros((2, 8), dtype=np.uint32)
    scales = np.ones((2, 1), dtype=np.float16)
    biases = np.zeros_like(scales)
    _validate_packed(weight, scales, biases)
    _validate_packed(weight, -scales, biases)  # MLX affine scales may be signed.
    for bad_weight, bad_scales, bad_biases in (
        (weight.astype(np.int32), scales, biases),
        (weight[:, :7], scales, biases),
        (weight, scales.astype(np.float32), biases),
        (weight, scales * np.nan, biases),
    ):
        with pytest.raises(ValueError):
            _validate_packed(bad_weight, bad_scales, bad_biases)
