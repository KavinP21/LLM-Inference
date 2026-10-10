"""Offline checkpoint validation for the pinned Qwen3.5-4B VLM adapter."""

from __future__ import annotations

import hashlib
import json
import math
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        raise ValueError("Qwen3.5 requires its multimodal checkpoint configuration")
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, dict) or not isinstance(vision, dict):
        raise ValueError("Qwen3.5 requires text_config and vision_config")
    from .qwen35_state import Qwen35StateGeometry

    Qwen35StateGeometry.from_text_config(text)
    expected = {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
    }
    if any(type(text.get(k)) is not int or text[k] != v for k, v in expected.items()):
        raise ValueError("checkpoint does not match the validated Qwen3.5-4B geometry")
    for key, expected_value in {
        "attention_bias": False,
        "tie_word_embeddings": True,
    }.items():
        if text.get(key) is not expected_value:
            raise ValueError(f"unsupported Qwen3.5 {key}")
    if text.get("attn_output_gate", True) is not True:
        raise ValueError("Qwen3.5 requires gated full attention")
    if (
        text.get("hidden_act", "silu") != "silu"
        or type(text.get("rms_norm_eps", 1e-6)) not in {int, float}
        or text.get("rms_norm_eps", 1e-6) != 1e-6
    ):
        raise ValueError("unsupported Qwen3.5 activation or normalization")
    layers = [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(32)
    ]
    if text.get("layer_types", layers) != layers:
        raise ValueError("unsupported Qwen3.5 layer ordering")
    context = text.get("max_position_embeddings")
    if type(context) is not int or not 1 <= context <= 262144:
        raise ValueError("invalid Qwen3.5 context limit")
    eos = text.get("eos_token_id")
    if type(eos) is not int or eos != 248044:
        raise ValueError("unsupported Qwen3.5 EOS metadata")
    rope = text.get("rope_parameters")
    if not isinstance(rope, dict) or (
        rope.get("rope_type", rope.get("type")) != "default"
        or rope.get("mrope_section") != [11, 11, 10]
        or type(rope.get("rope_theta")) not in {int, float}
        or rope["rope_theta"] != 10000000
        or type(rope.get("partial_rotary_factor")) not in {int, float}
        or rope["partial_rotary_factor"] != 0.25
        or rope.get("mrope_interleaved", True) is not True
    ):
        raise ValueError("unsupported Qwen3.5 multimodal RoPE configuration")
    vision_expected = {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 2304,
        "in_channels": 3,
    }
    if any(
        type(vision.get(k)) is not int or vision[k] != v
        for k, v in vision_expected.items()
    ):
        raise ValueError("unsupported Qwen3.5-4B vision geometry")
    if vision.get("deepstack_visual_indexes", []) != []:
        raise ValueError("Qwen3.5 deepstack vision is unsupported")
    if vision.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise ValueError("unsupported Qwen3.5 vision activation")
    for key, value in {
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
    }.items():
        if type(config.get(key)) is not int or config[key] != value:
            raise ValueError(f"unsupported Qwen3.5 {key}")
    quant = config.get("quantization")
    if (
        not isinstance(quant, dict)
        or type(quant.get("bits")) is not int
        or quant["bits"] != 4
        or type(quant.get("group_size")) is not int
        or quant["group_size"] != 64
        or quant.get("mode", "affine") != "affine"
    ):
        raise ValueError("Qwen3.5 adapter requires local affine 4-bit MLX weights")
    if config.get("quantization_config", quant) != quant:
        raise ValueError("Qwen3.5 quantization metadata disagrees")
    return text


@dataclass(frozen=True)
class Qwen35Checkpoint:
    path: Path
    config: dict[str, Any]
    tensors: dict[str, dict[str, Any]]
    identity: dict[str, str]
    weight_bytes: int
    signatures: tuple[tuple[Path, int, int, int], ...]

    def unchanged(self) -> bool:
        try:
            expected = {p.name for p, *_ in self.signatures}
            current = {
                p.name
                for p in self.path.iterdir()
                if p.is_file() and (p.suffix in {".json", ".jinja", ".safetensors"})
            }
            return current == expected and all(
                (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
                == (inode, size, modified)
                for p, inode, size, modified in self.signatures
            )
        except OSError:
            return False


def inspect_checkpoint(path: str | Path) -> Qwen35Checkpoint:
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Qwen3.5 requires an existing local checkpoint directory")
    with (directory / "config.json").open("rb") as stream:
        raw_config = stream.read((1 << 20) + 1)
    if len(raw_config) > 1 << 20:
        raise ValueError("checkpoint configuration exceeds 1 MiB")
    config = json.loads(raw_config)
    validate_config(config)
    if (directory / "offload_index.json").exists():
        raise ValueError(
            "Qwen3.5 expert offload is outside this adapter's checkpoint contract"
        )
    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix in {".json", ".jinja", ".safetensors"}
    )
    shards = [p for p in files if p.suffix == ".safetensors"]
    if not shards or len(shards) > 128 or len(files) > 256:
        raise ValueError("checkpoint has missing or excessive local weight shards")
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open("rb") as stream:
            index_raw = stream.read((16 << 20) + 1)
        if len(index_raw) > 16 << 20:
            raise ValueError("checkpoint shard index is excessive")
        index = json.loads(index_raw)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        shard_names = {p.name for p in shards}
        if (
            not isinstance(weight_map, dict)
            or not weight_map
            or any(
                not isinstance(name, str) or name not in shard_names
                for name in weight_map.values()
            )
        ):
            raise ValueError(
                "checkpoint index must reference only its local weight shards"
            )
    for required in (
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ):
        if not (directory / required).is_file():
            raise ValueError(f"full Qwen3.5 checkpoint requires {required}")
    signatures, manifest, tensors = [], [], {}
    sizes = {"F16": 2, "BF16": 2, "F32": 4, "U32": 4}
    for selected in files:
        before = selected.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint files must be regular files")
        if selected.suffix != ".safetensors" and before.st_size > 64 << 20:
            raise ValueError("checkpoint metadata file exceeds 64 MiB")
        digest = hashlib.sha256()
        with selected.open("rb") as stream:
            if selected.suffix == ".safetensors":
                length_bytes = stream.read(8)
                if len(length_bytes) != 8:
                    raise ValueError("truncated safetensors header")
                length = struct.unpack("<Q", length_bytes)[0]
                if not 1 <= length <= 16 << 20 or 8 + length > before.st_size:
                    raise ValueError("invalid safetensors header size")
                header = json.loads(stream.read(length))
                if not isinstance(header, dict):
                    raise ValueError("invalid safetensors header")
                ranges = []
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    if (
                        name in tensors
                        or not isinstance(info, dict)
                        or info.get("dtype") not in sizes
                    ):
                        raise ValueError("unsupported or duplicate checkpoint tensor")
                    shape, offsets = info.get("shape"), info.get("data_offsets")
                    if (
                        not isinstance(shape, list)
                        or not shape
                        or len(shape) > 8
                        or any(
                            type(n) is not int or not 0 < n <= 1_000_000_000
                            for n in shape
                        )
                        or not isinstance(offsets, list)
                        or len(offsets) != 2
                        or any(type(n) is not int or n < 0 for n in offsets)
                        or offsets[1] - offsets[0]
                        != math.prod(shape) * sizes[info["dtype"]]
                        or offsets[1] > before.st_size - 8 - length
                    ):
                        raise ValueError(
                            "invalid safetensors tensor dimensions or byte range"
                        )
                    ranges.append(tuple(offsets))
                    tensors[name] = info
                ordered = sorted(ranges)
                if any(b[0] < a[1] for a, b in zip(ordered, ordered[1:])):
                    raise ValueError("overlapping safetensors tensors")
                stream.seek(0)
            while chunk := stream.read(4 << 20):
                digest.update(chunk)
        after = selected.stat()
        signature = (before.st_ino, before.st_size, before.st_mtime_ns)
        if signature != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("checkpoint changed while hashing")
        signatures.append((selected, *signature))
        manifest.append(
            {
                "name": selected.name,
                "bytes": before.st_size,
                "sha256": digest.hexdigest(),
            }
        )

    def expect(name: str, shape: list[int], dtypes: set[str]) -> None:
        info = tensors.get(name, {})
        if info.get("shape") != shape or info.get("dtype") not in dtypes:
            raise ValueError(f"unsupported checkpoint tensor {name}")

    prefixes = ("language_model.model.", "model.language_model.", "model.")
    prefix = next(
        (p for p in prefixes if p + "layers.3.self_attn.k_proj.scales" in tensors), None
    )
    if prefix is None:
        raise ValueError("checkpoint has no validated full-attention weights")
    for i in range(32):
        layer = f"{prefix}layers.{i}."
        if (i + 1) % 4 == 0:
            for projection in ("k_proj", "v_proj"):
                expect(
                    layer + f"self_attn.{projection}.scales",
                    [1024, 40],
                    {"F16", "BF16"},
                )
        else:
            conv = tensors.get(layer + "linear_attn.conv1d.weight", {})
            if conv.get("shape") not in ([8192, 4, 1], [8192, 1, 4]) or conv.get(
                "dtype"
            ) not in {"F16", "BF16"}:
                raise ValueError("unsupported recurrent convolution weights")
            expect(layer + "linear_attn.A_log", [32], {"F32"})
    vision_prefix = next(
        (
            p
            for p in ("vision_tower.", "model.visual.")
            if p + "patch_embed.proj.weight" in tensors
        ),
        None,
    )
    if vision_prefix is None:
        raise ValueError("full Qwen3.5 checkpoint is missing its vision tower")
    patch = tensors[vision_prefix + "patch_embed.proj.weight"]
    if patch.get("shape") not in (
        [1024, 2, 16, 16, 3],
        [1024, 3, 2, 16, 16],
    ) or patch.get("dtype") not in {"F16", "BF16"}:
        raise ValueError("unsupported Qwen3.5 vision patch weights")
    identity = {
        "model_data_sha256": hashlib.sha256(
            json.dumps(
                [m for m in manifest if m["name"].endswith(".safetensors")],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "model_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "processor_signature": hashlib.sha256(
            json.dumps(
                [m for m in manifest if not m["name"].endswith(".safetensors")],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "tokenizer_signature": hashlib.sha256(
            json.dumps(
                [
                    m
                    for m in manifest
                    if m["name"]
                    in {
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "chat_template.jinja",
                        "vocab.json",
                    }
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    return Qwen35Checkpoint(
        directory,
        config,
        tensors,
        identity,
        sum(p.stat().st_size for p in shards),
        tuple(signatures),
    )
