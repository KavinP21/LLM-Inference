"""Prepare the pinned full Qwen3.5 daily-driver checkpoint locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_REPO = "mlx-community/Qwen3.5-4B-4bit"
MODEL_REVISION = "0e7ffd5c629ef7719d4cbc04069232580bfa9d9c"


def prepare(directory: str | Path, *, download: bool = True) -> dict:
    selected = Path(directory).expanduser().resolve()
    if download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=str(selected),
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
            max_workers=3,
        )
    from .qwen35_checkpoint import inspect_checkpoint

    checkpoint = inspect_checkpoint(selected)
    return {
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "directory": str(selected),
        "identity": checkpoint.identity,
        "weight_storage_bytes": checkpoint.weight_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="models/qwen3.5-4b-mlx")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory, download=not args.verify_only), indent=2))


if __name__ == "__main__":
    main()
