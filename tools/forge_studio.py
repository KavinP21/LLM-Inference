#!/usr/bin/env python3
"""Convenient repository launcher with the local Studio worker profile."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from forge_llm.harness.cli import main  # noqa: E402 - repository path bootstrap

if __name__ == "__main__":
    os.chdir(ROOT)
    if not any(
        arg == "--workspace" or arg.startswith("--workspace=") for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--workspace", str(ROOT)])
    if "--demo" not in sys.argv and not any(
        arg == "--config" or arg.startswith("--config=") for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--config", str(ROOT / "examples/agents-studio.json")])
    main()
