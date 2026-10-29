import asyncio
import json
import os
import sys

import pytest
from instinct.agents.tools import ToolValidationError
from instinct.harness.commands import CommandTools


def test_real_command_output_exit_code_cwd_and_no_shell_expansion(tmp_path):
    async def check():
        tools = CommandTools(tmp_path)
        result = json.loads(
            await tools.run_command(
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import sys; print(sys.argv[1]); print('problem',file=sys.stderr); sys.exit(3)",
                        "$(whoami)",
                    ]
                }
            )
        )
        assert result["exit_code"] == 3
        assert "$(whoami)" in result["output"] and "problem" in result["output"]
        assert not result["timed_out"] and not result["truncated"]
        assert result["cwd"] == "."

    asyncio.run(check())


def test_command_does_not_inherit_credential_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("INSTINCT_TEST_PRIVATE_KEY", "test-fixture-not-a-secret")

    async def check():
        result = json.loads(
            await CommandTools(tmp_path).run_command(
                {
                    "argv": [
                        sys.executable,
                        "-c",
                        "import os; print('INSTINCT_TEST_PRIVATE_KEY' in os.environ)",
                    ]
                }
            )
        )
        assert result["output"].strip() == "False"

    asyncio.run(check())


def test_output_and_deadline_are_bounded(tmp_path):
    async def check():
        tools = CommandTools(tmp_path, max_output_bytes=1000)
        huge = json.loads(
            await tools.run_command(
                {"argv": [sys.executable, "-c", "print('x'*10000)"]}
            )
        )
        assert huge["truncated"] and len(huge["output"].encode()) <= 1000
        slow = json.loads(
            await tools.run_command(
                {
                    "argv": [sys.executable, "-c", "import time; time.sleep(20)"],
                    "timeout_seconds": 1,
                }
            )
        )
        assert slow["timed_out"] and slow["exit_code"] != 0

    asyncio.run(check())


@pytest.mark.parametrize(
    "args",
    [
        {"argv": "echo hi"},
        {"argv": []},
        {"argv": ["echo"], "cwd": "../"},
        {"argv": ["echo"], "cwd": "missing-directory"},
        {"argv": ["echo"], "timeout_seconds": True},
        {"argv": ["echo"], "shell": True},
        {"argv": ["echo", "\x00"]},
    ],
)
def test_invalid_command_rejected_before_execution(tmp_path, args):
    async def check():
        with pytest.raises(ToolValidationError):
            await CommandTools(tmp_path).run_command(args)

    asyncio.run(check())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_cancellation_stops_owned_child(tmp_path):
    async def check():
        marker = tmp_path / "started"
        code = "import os,time,pathlib; pathlib.Path('started').write_text(str(os.getpid())); time.sleep(30)"
        task = asyncio.create_task(
            CommandTools(tmp_path).run_command({"argv": [sys.executable, "-c", code]})
        )
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)
        pid = int(marker.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    asyncio.run(check())
