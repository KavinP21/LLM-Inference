from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_gpu_workflow_lease_blocks_competing_driver_but_allows_children(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    monkeypatch.delenv("FORGE_GPU_EVIDENCE_FD", raising=False)
    from gpu_guard import exclusive_gpu_workflow

    command = [
        sys.executable,
        "-c",
        (
            "from gpu_guard import exclusive_gpu_workflow\n"
            "with exclusive_gpu_workflow():\n"
            "    print('inherited lease works')\n"
        ),
    ]
    environment = {**os.environ, "PYTHONPATH": str(root / "benchmarks")}
    with exclusive_gpu_workflow() as run:
        child = run(
            command, env=environment, capture_output=True, text=True, check=True
        )
        assert "inherited lease works" in child.stdout
        competitor = subprocess.run(
            command, env=environment, capture_output=True, text=True, check=False
        )
        assert competitor.returncode != 0
        assert "Another Forge evidence workflow" in competitor.stderr
    # Closing the owning workflow really releases the lease; no busy loop/retry.
    subsequent = subprocess.run(
        command, env=environment, capture_output=True, text=True, check=False
    )
    assert subsequent.returncode == 0


def test_invalid_inherited_gpu_lease_fails_closed(monkeypatch):
    import pytest

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    from gpu_guard import exclusive_gpu_workflow

    monkeypatch.setenv("FORGE_GPU_EVIDENCE_FD", "-1")
    with pytest.raises((OSError, ValueError)), exclusive_gpu_workflow():
        raise AssertionError("invalid lease was accepted")
