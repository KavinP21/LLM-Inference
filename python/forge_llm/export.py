from __future__ import annotations

import argparse
import json
from pathlib import Path

from .format import ModelConfig, write_engine, write_manifest
from .model_contract import validate_model_weights
from .model_file import ModelFile


def export_model(model_name_or_path: str, output: Path) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("install forge-llm[export] before exporting") from exc

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model_type = getattr(model.config, "model_type", None)
    if model_type not in {"qwen2", "gemma3_text"}:
        raise ValueError("Forge exports Qwen2 or text-only Gemma 3 checkpoints")
    if bool(getattr(model.config, "use_bidirectional_attention", False)):
        raise ValueError("Gemma 3 bidirectional/image attention is not supported")
    if model_type == "gemma3_text" and bool(
        getattr(model.config, "attention_bias", False)
    ):
        raise ValueError("Gemma 3 attention projections must be bias-free")
    config = ModelConfig.from_huggingface(model.config)
    tensors = {
        name: value.detach().cpu().to(dtype=torch.float16).numpy()
        for name, value in model.state_dict().items()
    }
    # Some safetensors checkpoints omit the duplicated tied alias. The runtime keeps an explicit
    # LM-head entry so its on-disk contract does not depend on Transformers aliasing behavior.
    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if "lm_head.weight" not in tensors and tied:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    aliases = {"lm_head.weight": "model.embed_tokens.weight"} if tied else None
    result = write_engine(output, config, tensors, aliases=aliases)
    with ModelFile(output) as artifact:
        validate_model_weights(artifact)
    write_manifest(
        output.with_suffix(output.suffix + ".json"), result, model_name_or_path
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a Qwen2 or text-only Gemma 3 checkpoint for Forge LLM"
    )
    parser.add_argument("model", help="Hugging Face model id or local checkpoint")
    parser.add_argument("output", type=Path, help="output .engine file")
    args = parser.parse_args()
    print(json.dumps(export_model(args.model, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
