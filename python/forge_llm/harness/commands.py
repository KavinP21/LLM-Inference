"""Reviewed argv execution with confined cwd and owned process-group cleanup.

This is an execution tool, not an OS sandbox. The approval boundary lives in
ApprovalGate. No shell, credential environment or automatic retry is supplied.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from ..agents.tools import ToolValidationError


class CommandTools:
    schemas = {
        "run_command": {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 64,
                },
                "cwd": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
            },
            "required": ["argv"],
            "additionalProperties": False,
        }
    }

    def __init__(self, workspace: Path, *, max_output_bytes: int = 128_000):
        self.workspace = Path(workspace).resolve(strict=True)
        self.max_output_bytes = max_output_bytes

    def mapping(self):
        return {"run_command": self.run_command}

    def tool_descriptions(self):
        return {
            "run_command": {
                "description": "Run a command as an argv array in the workspace. Operator reviews the exact command before execution. Use this for builds, tests, linters and developer commands. No shell expansion; use separate arguments. The launcher Python environment is on PATH. Output is bounded and exit status is reported. This runs with the operator's OS privileges.",
                "parameters": self.schemas["run_command"],
            }
        }

    def _validate(self, args):
        if not isinstance(args, dict) or args.keys() - {
            "argv",
            "cwd",
            "timeout_seconds",
        }:
            raise ToolValidationError("command accepts argv, cwd and timeout_seconds")
        argv = args.get("argv")
        if (
            not isinstance(argv, list)
            or not 1 <= len(argv) <= 64
            or any(
                not isinstance(part, str)
                or not part
                or "\x00" in part
                or len(part) > 8192
                for part in argv
            )
        ):
            raise ToolValidationError(
                "argv must contain 1 to 64 nonempty bounded strings"
            )
        if sum(len(part.encode("utf-8")) for part in argv) > 32_768:
            raise ToolValidationError("combined command arguments exceed 32 KiB")
        relative = args.get("cwd", ".")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ToolValidationError("command cwd must be workspace-relative")
        try:
            cwd = (self.workspace / relative).resolve(strict=True)
        except OSError as exc:
            raise ToolValidationError(
                "command cwd must be an existing workspace directory"
            ) from exc
        if not cwd.is_dir() or not cwd.is_relative_to(self.workspace):
            raise ToolValidationError("command cwd must stay inside the workspace")
        timeout = args.get("timeout_seconds", 60)
        if type(timeout) is not int or not 1 <= timeout <= 120:
            raise ToolValidationError("command timeout must be 1 to 120 seconds")
        return argv, cwd, timeout

    async def _stop(self, process):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 0.5)
        except TimeoutError:
            pass
        # Descendants can keep pipes open after their parent exits.
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.returncode is None:
            process.kill()
        await process.wait()

    async def run_command(self, args):
        argv, cwd, timeout = self._validate(args)
        environment = {
            name: os.environ[name]
            for name in (
                "PATH",
                "LANG",
                "LC_ALL",
                "LC_CTYPE",
                "TMPDIR",
                "TEMP",
                "TMP",
                "SYSTEMROOT",
                "WINDIR",
            )
            if name in os.environ
        }
        environment["PATH"] = (
            str(Path(sys.executable).parent)
            + os.pathsep
            + environment.get("PATH", os.defpath)
        )
        environment["PYTHONPATH"] = str(self.workspace / "python")
        environment["PYTHONUNBUFFERED"] = "1"
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        except OSError as exc:
            raise ToolValidationError(
                f"command could not start: {exc.strerror}"
            ) from exc
        output = bytearray()
        truncated, timed_out = False, False

        async def drain():
            nonlocal truncated
            while data := await process.stdout.read(16_384):
                remaining = self.max_output_bytes - len(output)
                output.extend(data[: max(0, remaining)])
                if len(data) > remaining:
                    truncated = True
                    await self._stop(process)
                    break
            await process.wait()

        try:
            await asyncio.wait_for(drain(), timeout)
        except TimeoutError:
            timed_out = True
            await self._stop(process)
        except BaseException:
            await self._stop(process)
            raise
        return json.dumps(
            {
                "argv": argv,
                "cwd": str(cwd.relative_to(self.workspace)) or ".",
                "exit_code": process.returncode,
                "output": output.decode("utf-8", "replace"),
                "timed_out": timed_out,
                "truncated": truncated,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
            ensure_ascii=False,
        )
