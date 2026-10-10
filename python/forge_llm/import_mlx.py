"""Import a local MLX affine-4-bit Qwen2 checkpoint as reconstructed FP16.

This is a storage conversion, not a W4 inference backend or recovery of the
original FP16 checkpoint. Source files are read-only; no download is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .format import ModelConfig, write_engine
from .model_contract import validate_model_weights
from .model_file import ModelFile


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _config(source: Path) -> tuple[ModelConfig, dict]:
    raw = json.loads((source / "config.json").read_text())
    quantization = raw.get("quantization")
    if not isinstance(quantization, dict) or quantization != {
        "group_size": 64,
        "bits": 4,
    }:
        raise ValueError(
            "import requires explicit MLX affine quantization: bits=4, group_size=64"
        )
    if raw.get("model_type") != "qwen2" or raw.get("use_sliding_window", False):
        raise ValueError("import supports Qwen2 with ordinary causal attention only")
    if raw.get("hidden_act", "silu") != "silu":
        raise ValueError("Qwen2 import requires the SiLU activation")
    normalized = dict(raw)
    normalized["sliding_window"] = 0
    return ModelConfig.from_huggingface(SimpleNamespace(**normalized)), raw


def _source_files(source: Path) -> tuple[list[Path], dict[str, str] | None]:
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        mapping = index.get("weight_map")
        if (
            not isinstance(mapping, dict)
            or not mapping
            or not all(
                isinstance(k, str)
                and isinstance(v, str)
                and Path(v).name == v
                and v.endswith(".safetensors")
                for k, v in mapping.items()
            )
        ):
            raise ValueError("invalid local safetensors index")
        files = [source / name for name in sorted(set(mapping.values()))]
    else:
        files = sorted(source.glob("*.safetensors"))
        mapping = None
        if len(files) != 1:
            raise ValueError("a multi-shard source requires a safetensors index")
    if not files or any(not file.is_file() for file in files):
        raise ValueError("local checkpoint is missing safetensors data")
    return files, mapping


def _expected_names(config: ModelConfig) -> set[str]:
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        names.update(
            prefix + norm + ".weight"
            for norm in ("input_layernorm", "post_attention_layernorm")
        )
        names.update(
            prefix + name + ".weight"
            for name in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
        names.update(
            prefix + "self_attn." + name + "_proj.bias" for name in ("q", "k", "v")
        )
    return names


def _validate_packed(
    weight: np.ndarray, scales: np.ndarray, biases: np.ndarray
) -> None:
    if weight.dtype != np.dtype("uint32") or weight.ndim != 2:
        raise ValueError("MLX packed weights must be rank-two uint32 matrices")
    if weight.shape[1] % 8 or scales.shape != (weight.shape[0], weight.shape[1] // 8):
        raise ValueError("packed matrix does not match affine group_size=64 scales")
    if (
        scales.dtype != np.float16
        or biases.dtype != np.float16
        or biases.shape != scales.shape
    ):
        raise ValueError("MLX affine scales/biases must be matching FP16 matrices")
    if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(biases)):
        raise ValueError("MLX affine parameters must be finite")


def _mlx_dequantizer(device: str) -> tuple[Callable, str]:
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError(
            "install forge-llm[mlx,export] to reconstruct an MLX checkpoint"
        ) from exc
    selected = mx.cpu if device == "cpu" else mx.gpu

    def decode(weight, scales, biases):
        _validate_packed(weight, scales, biases)
        with mx.stream(selected):
            result = mx.dequantize(
                mx.array(weight),
                mx.array(scales),
                mx.array(biases),
                group_size=64,
                bits=4,
                mode="affine",
                dtype=mx.float16,
            )
            mx.eval(result)
            return np.array(result)

    return decode, mx.__version__


def import_mlx_checkpoint(
    source: Path,
    output: Path,
    *,
    device: str = "cpu",
    _dequantize: Callable | None = None,
) -> dict:
    """Reconstruct all packed matrices through MLX and publish without overwrite.

    The private injection argument supports portable importer validation tests;
    production conversion always uses ``mlx.core.dequantize`` on the requested
    device, with CPU as the default. Output remains FP16 reconstructed from W4.
    """
    source, output = Path(source), Path(output)
    manifest_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(
            "refusing to overwrite an engine or its provenance manifest"
        )
    if device not in {"cpu", "gpu"}:
        raise ValueError("dequantization device must be cpu or gpu")
    config, raw = _config(source)
    shards, mapping = _source_files(source)
    fingerprint_files = shards + [source / "config.json"]
    fingerprint_files += [
        source / name
        for name in (
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        if (source / name).is_file()
    ]
    hashes = {
        file.name: {"sha256": _sha256(file), "bytes": file.stat().st_size}
        for file in fingerprint_files
    }
    if _dequantize is None:
        decode, mlx_version = _mlx_dequantizer(device)
    else:
        decode, mlx_version = _dequantize, "test-injected"
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("install forge-llm[export] to read safetensors") from exc
    tensors, quantized, consumed = {}, [], set()
    with ExitStack() as stack:
        readers = {
            file.name: stack.enter_context(safe_open(str(file), framework="np"))
            for file in shards
        }
        owners = {}
        for filename, reader in readers.items():
            # safe_open exposes keys(), but is not an iterable dictionary.
            for name in reader.keys():  # noqa: SIM118
                if name in owners:
                    raise ValueError("duplicate tensor across safetensors shards")
                owners[name] = filename
        if mapping is not None and mapping != owners:
            raise ValueError("safetensors index disagrees with actual tensor ownership")

        def get(name):
            return readers[owners[name]].get_tensor(name)

        for name in sorted(owners):
            if name.endswith((".scales", ".biases")):
                continue
            value = get(name)
            if value.dtype == np.uint32:
                prefix = name.removesuffix(".weight")
                scale_name, bias_name = prefix + ".scales", prefix + ".biases"
                if (
                    not name.endswith(".weight")
                    or scale_name not in owners
                    or bias_name not in owners
                ):
                    raise ValueError(f"packed tensor lacks affine parameters: {name}")
                scales, biases = get(scale_name), get(bias_name)
                _validate_packed(value, scales, biases)
                restored = decode(value, scales, biases)
                if restored.dtype != np.float16 or restored.shape != (
                    value.shape[0],
                    value.shape[1] * 8,
                ):
                    raise ValueError(
                        f"dequantizer returned an invalid FP16 matrix: {name}"
                    )
                if not np.all(np.isfinite(restored)):
                    raise ValueError(
                        f"dequantization produced non-finite values: {name}"
                    )
                tensors[name] = restored
                consumed.update((name, scale_name, bias_name))
                quantized.append(
                    {
                        "name": name,
                        "packed_shape": list(value.shape),
                        "reconstructed_shape": list(restored.shape),
                        "source_shard": owners[name],
                    }
                )
            elif value.dtype == np.float16 and np.all(np.isfinite(value)):
                tensors[name] = value
                consumed.add(name)
            else:
                raise ValueError(
                    f"unsupported checkpoint dtype or non-finite values: {name}"
                )
        if consumed != set(owners):
            raise ValueError("checkpoint contains unused affine parameters")
    if not quantized:
        raise ValueError("checkpoint contains no MLX packed matrices")
    aliases = None
    if raw.get("tie_word_embeddings", False):
        embed = tensors["model.embed_tokens.weight"]
        if "lm_head.weight" in tensors and not np.array_equal(
            tensors["lm_head.weight"], embed
        ):
            raise ValueError(
                "tied embedding declaration disagrees with reconstructed LM head"
            )
        tensors["lm_head.weight"] = embed
        aliases = {"lm_head.weight": "model.embed_tokens.weight"}
    if set(tensors) != _expected_names(config):
        raise ValueError("checkpoint tensors do not match Forge's exact Qwen2 contract")
    output.parent.mkdir(parents=True, exist_ok=True)
    engine_temp = manifest_temp = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            prefix=output.name + ".", suffix=".tmp", dir=output.parent, delete=False
        ) as temporary:
            engine_temp = Path(temporary.name)
        result = write_engine(engine_temp, config, tensors, aliases=aliases)
        del tensors
        with ModelFile(engine_temp) as artifact:
            validate_model_weights(artifact)
        manifest = {
            "schema": "forge_mlx_reconstructed_fp16_v1",
            "source_directory": str(source.resolve()),
            "parameter_origin": "reconstructed_from_mlx_affine_4bit",
            "original_fp16_checkpoint": False,
            "inference_weight_dtype": "float16",
            "source_quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            "dequantization": {
                "implementation": "mlx.core.dequantize"
                if _dequantize is None
                else "test-injected",
                "mlx_version": mlx_version,
                "device": device,
                "output_dtype": "float16",
            },
            "source_files": hashes,
            "reconstructed_matrices": quantized,
            **result,
            "path": str(output),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=manifest_path.name + ".",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary:
            manifest_temp = Path(temporary.name)
            temporary.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Exclusive links publish fully validated files without replacing a
        # concurrently created destination. Both temporaries are on this FS.
        os.link(engine_temp, output)
        published = True
        os.link(manifest_temp, manifest_path)
        return manifest
    except Exception:
        if published:
            output.unlink(missing_ok=True)
        raise
    finally:
        if engine_temp is not None:
            engine_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="local cached MLX Qwen2 safetensors directory"
    )
    parser.add_argument("output", type=Path, help="new reconstructed-FP16 .engine path")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    result = import_mlx_checkpoint(args.source, args.output, device=args.device)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("source_files", "reconstructed_matrices")
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
