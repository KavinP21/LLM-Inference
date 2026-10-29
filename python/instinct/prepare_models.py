"""Download the MLX checkpoints the Studio and agent configs expect.

python -m instinct.prepare_models              # Gemma 4 driver + Qwen3.5 sub-agents
python -m instinct.prepare_models qwen3.5-4b   # just one
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# name -> (Hugging Face repo, pinned revision, local directory)
MODELS = {
    "gemma-4-26b-a4b": (
        "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
        "0e3cbab38ce568cf6e23543010d08d03b731910c",
        "models/gemma-4-26b-a4b-mlx",
    ),
    "gemma-4-26b-a4b-assistant": (
        "mlx-community/gemma-4-26B-A4B-it-assistant-bf16",
        "cda74908f1dbe7d3dbd3030e66576a7d4094144f",
        "models/gemma-4-26b-a4b-assistant",
    ),
    "qwen3.5-4b": (
        "mlx-community/Qwen3.5-4B-4bit",
        "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c",
        "models/qwen3.5-4b-mlx",
    ),
}
DEFAULT = ("gemma-4-26b-a4b", "gemma-4-26b-a4b-assistant", "qwen3.5-4b")


def prepare(name: str, root: Path = Path(".")) -> dict:
    from huggingface_hub import snapshot_download

    repo, revision, directory = MODELS[name]
    target = root / directory
    snapshot_download(
        repo,
        revision=revision,
        local_dir=str(target),
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
        max_workers=8,
    )
    size = sum(f.stat().st_size for f in target.glob("*.safetensors"))
    return {
        "name": name,
        "repo": repo,
        "revision": revision,
        "directory": str(target),
        "bytes": size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("names", nargs="*", help=f"any of: {', '.join(MODELS)}")
    args = parser.parse_args()
    unknown = set(args.names) - set(MODELS)
    if unknown:
        parser.error(f"unknown model(s): {', '.join(sorted(unknown))}")
    for name in args.names or DEFAULT:
        print(json.dumps(prepare(name)), flush=True)


if __name__ == "__main__":
    main()
