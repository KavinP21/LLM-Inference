"""Fail-fast, inherited process lease for cooperating GPU evidence drivers.

This protects calibration/measurement workflows, not the inference scheduler or
unrelated applications. It is intentionally outside the timed runtime path.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_gpu_workflow():
    import fcntl

    path = Path(tempfile.gettempdir()) / f"forge-gpu-evidence-{os.getuid()}.lock"
    inherited = os.environ.get("FORGE_GPU_EVIDENCE_FD")
    owned = inherited is None
    descriptor = None
    try:
        if owned:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        else:
            descriptor = int(inherited)
            expected, actual = path.stat(), os.fstat(descriptor)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise RuntimeError("invalid inherited GPU evidence lease")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Another Forge evidence workflow owns the GPU lease; wait for it to finish."
            ) from exc

        def run(command, **kwargs):
            environment = dict(kwargs.pop("env", os.environ))
            environment["FORGE_GPU_EVIDENCE_FD"] = str(descriptor)
            passed = set(kwargs.pop("pass_fds", ())) | {descriptor}
            return subprocess.run(
                command,
                env=environment,
                pass_fds=tuple(passed),
                check=kwargs.pop("check", False),
                **kwargs,
            )

        yield run
    finally:
        # An inherited file description shares its lock with the parent. Never
        # unlock it in the child; the owning workflow releases it on exit.
        if owned and descriptor is not None:
            os.close(descriptor)
