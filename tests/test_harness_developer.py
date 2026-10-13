"""Developer inspection uses confined disk reads and actual journal receipts."""

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from urllib.parse import quote

import pytest
from forge_llm.agents.protocol import Generation
from forge_llm.agents.tools import ToolValidationError
from forge_llm.harness.developer import MAX_FILE_BYTES, DeveloperWorkspace
from forge_llm.harness.service import StudioService


class ScriptBackend:
    def __init__(self, actions=None):
        self.actions = actions or [{"action": "finish", "result": "done"}]

    async def generate(self, messages, max_tokens, request_id):
        await asyncio.sleep(0.001)
        step = int(request_id.rsplit(".", 1)[-1])
        return Generation(json.dumps(self.actions[step - 1]), 10, 10)

    async def cancel(self, request_id):
        return None

    async def close(self):
        return None


def service_for(tmp_path, backend=None):
    return StudioService(
        tmp_path,
        tmp_path / ".forge",
        demo=True,
        backend=backend or ScriptBackend(),
        config={"runtime": {"max_agents": 1}},
    )


async def run_id(service, chat_id):
    for _ in range(200):
        row = service.conversations._latest(chat_id)
        if row["run_id"]:
            return row["run_id"]
        await asyncio.sleep(0.005)
    pytest.fail("conversation did not start")


async def approval(service, run):
    for _ in range(200):
        rows = service.gates[run].pending(run)
        if rows:
            return rows[0]
        await asyncio.sleep(0.005)
    pytest.fail("approval did not arrive")


def test_workspace_browser_search_content_hash_limits_and_privacy(tmp_path):
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "example.py"
    source.write_text("value = 'hello'\nprint(value)\n")
    (tmp_path / ".env").write_text("private")
    for ignored in ("models", ".venv", "secrets", "node_modules"):
        (tmp_path / ignored).mkdir()
        (tmp_path / ignored / "hidden.txt").write_text("private")
    (tmp_path / "binary.dat").write_bytes(b"a\x00b")
    (tmp_path / "large.txt").write_bytes(b"a" * (MAX_FILE_BYTES + 1))

    async def scenario():
        service = service_for(tmp_path)
        developer = DeveloperWorkspace(service)
        try:
            listing = await developer.dispatch(
                "POST", "/api/workspace/files", {"query": "EXAMPLE", "limit": 1}
            )
            assert [entry["path"] for entry in listing["files"]] == ["src/example.py"]
            assert (
                await service.dispatch(
                    "POST", "/api/workspace/files", {"query": "EXAMPLE"}
                )
            )["files"] == listing["files"]
            all_files = await developer.dispatch("GET", "/api/workspace/files", {})
            assert not any(
                any(
                    part in entry["path"]
                    for part in (
                        ".env",
                        ".forge",
                        "models",
                        ".venv",
                        "secrets",
                        "node_modules",
                    )
                )
                for entry in all_files["files"]
            )
            result = await developer.dispatch(
                "GET", "/api/workspace/file/" + quote("src/example.py", safe=""), {}
            )
            assert result["content"] == source.read_text()
            assert result["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
            assert result["total_lines"] == 2 and result["language"] == "python"
            assert result["source"] == "current_workspace"
            for path in (
                ".env",
                ".ENV",
                "models/hidden.txt",
                ".venv/hidden.txt",
                ".forge/studio.sqlite",
                "secrets/hidden.txt",
                "../escape",
            ):
                with pytest.raises(ToolValidationError):
                    await developer.dispatch(
                        "GET", "/api/workspace/file/" + quote(path, safe=""), {}
                    )
            with pytest.raises(ToolValidationError, match="binary"):
                await developer.dispatch("GET", "/api/workspace/file/binary.dat", {})
            with pytest.raises(ToolValidationError, match="limit"):
                await developer.dispatch("GET", "/api/workspace/file/large.txt", {})
            with pytest.raises(ToolValidationError):
                await developer.dispatch(
                    "POST", "/api/workspace/files", {"limit": True}
                )
            with pytest.raises(ToolValidationError):
                await developer.dispatch(
                    "POST", "/api/workspace/files", {"path": "models"}
                )
        finally:
            await service.close()

    asyncio.run(scenario())


def test_workspace_browser_rejects_symlinks_to_private_or_outside_files(tmp_path):
    async def scenario():
        service = service_for(tmp_path)
        outside = tmp_path.with_name(tmp_path.name + "-outside.txt")
        outside.write_text("outside")
        (tmp_path / "escape.txt").symlink_to(outside)
        (tmp_path / "private.txt").symlink_to(service.state_dir / "studio.sqlite")
        developer = DeveloperWorkspace(service)
        try:
            for path in ("escape.txt", "private.txt"):
                with pytest.raises(ToolValidationError):
                    await developer.dispatch("GET", "/api/workspace/file/" + path, {})
            files = (await developer.dispatch("GET", "/api/workspace/files", {}))[
                "files"
            ]
            assert not any(
                entry["path"] in {"escape.txt", "private.txt"} for entry in files
            )
        finally:
            await service.close()
            outside.unlink()

    asyncio.run(scenario())


def test_proposed_and_executed_edits_are_scoped_to_exact_invocations(tmp_path):
    source = tmp_path / "example.txt"
    source.write_text("old\n")
    args = {"path": "example.txt", "content": "new\n"}
    backend = ScriptBackend(
        [
            {"action": "read_file", "args": {"path": "example.txt"}},
            {"action": "write_file", "args": args},
            {"action": "read_file", "args": {"path": "example.txt"}},
            {"action": "write_file", "args": args},
            {"action": "finish", "result": "done"},
        ]
    )

    async def scenario():
        service = service_for(tmp_path, backend)
        developer = DeveloperWorkspace(service)
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Edit", "policy": "review"}
                )
            )["chat_id"]
            run = await run_id(service, chat)
            first = await approval(service, run)
            before = await developer.chat(chat)
            assert before["files"] == [] and len(before["reads"]) == 1
            assert before["changes"][0]["status"] == "proposed"
            assert before["changes"][0]["approval_id"] == first["id"]
            assert before["changes"][0]["verified"] is False
            assert "-old\n+new\n" in before["changes"][0]["diff"]
            await service.decide(first["id"], {"approved": True})
            second = await approval(service, run)
            while second["id"] == first["id"]:
                await asyncio.sleep(0.005)
                second = await approval(service, run)
            during = await developer.chat(chat)
            executed = next(
                row for row in during["changes"] if row["status"] == "executed"
            )
            proposed = next(
                row for row in during["changes"] if row["status"] == "proposed"
            )
            assert executed["approval_id"] == first["id"]
            assert proposed["approval_id"] == second["id"]
            assert executed["verified"] is True
            assert during["files"][0]["change"] == "modified"
            assert during["files"][0]["current_matches_receipt"] is True
            await service.decide(second["id"], {"approved": False})
            await asyncio.wait_for(service.jobs[run], 2)
            source.write_text("external edit\n")
            after = await developer.chat(chat)
            assert after["files"][0]["current_matches_receipt"] is False
            assert after["files"][0]["sha256"] == hashlib.sha256(b"new\n").hexdigest()
            assert any(change["status"] == "denied" for change in after["changes"])
            assert after["usage"]["total_tokens"] > 0
            assert after["agents"][0]["run_id"] == run
            assert after["agents"][0]["can_message"] is False
            assert source.read_text() == "external edit\n"
        finally:
            await service.close()

    asyncio.run(scenario())


def test_artifact_and_command_receipts_do_not_include_another_chat(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_fixture.py").write_text(
        "def test_fixture():\n    assert True\n"
    )
    backend = ScriptBackend(
        [
            {
                "action": "save_artifact",
                "args": {"path": "report.md", "content": "# Actual artifact\n"},
            },
            {"action": "run_tests", "args": {"paths": ["tests/test_fixture.py"]}},
            {
                "action": "run_command",
                "args": {
                    "argv": [sys.executable, "-c", "print('controlled terminal')"]
                },
            },
            {"action": "finish", "result": "done"},
        ]
    )

    async def scenario():
        service = service_for(tmp_path, backend)
        developer = DeveloperWorkspace(service)
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Inspect", "policy": "review"}
                )
            )["chat_id"]
            run = await run_id(service, chat)
            artifact = await approval(service, run)
            await service.decide(artifact["id"], {"approved": True})
            command = await approval(service, run)
            assert command["tool"] == "run_tests"
            pending_command = (await developer.chat(chat))["commands"][0]
            assert (
                pending_command["status"] == "proposed"
                and pending_command["output"] is None
            )
            await service.decide(command["id"], {"approved": True})
            controlled = await approval(service, run)
            assert controlled["tool"] == "run_command"
            await service.decide(controlled["id"], {"approved": True})
            await asyncio.wait_for(service.jobs[run], 5)
            actual = await service.dispatch("GET", f"/api/chats/{chat}/developer", {})
            assert actual["files"][0]["origin"] == "artifact"
            assert actual["files"][0]["change"] == "created"
            assert actual["commands"][0]["status"] == "completed"
            assert "1 passed" in actual["commands"][0]["output"]
            assert actual["commands"][0]["verified"] is True
            assert actual["commands"][1]["tool"] == "run_command"
            assert actual["commands"][1]["status"] == "completed"
            assert actual["commands"][1]["output"] == "controlled terminal\n"
            service.backend = ScriptBackend()
            other = (
                await service.conversations.create(
                    {"message": "Separate", "policy": "read_only"}
                )
            )["chat_id"]
            other_run = await run_id(service, other)
            await service.jobs[other_run]
            unrelated = await developer.chat(other)
            assert (
                unrelated["files"] == []
                and unrelated["commands"] == []
                and unrelated["changes"] == []
            )
            assert {agent["run_id"] for agent in unrelated["agents"]} == {other_run}
            with pytest.raises(ValueError, match="does not exist"):
                await developer.chat("unknown-chat")
        finally:
            await service.close()

    asyncio.run(scenario())


def test_per_agent_message_and_cancel_use_the_live_scheduler(tmp_path):
    class WaitingBackend(ScriptBackend):
        def __init__(self):
            self.entered = asyncio.Event()

        async def generate(self, *args):
            self.entered.set()
            await asyncio.Event().wait()

    async def scenario():
        backend = WaitingBackend()
        service = service_for(tmp_path, backend)
        developer = DeveloperWorkspace(service)
        try:
            chat = (
                await service.conversations.create(
                    {"message": "Work", "policy": "read_only"}
                )
            )["chat_id"]
            run = await run_id(service, chat)
            await backend.entered.wait()
            root = (await developer.chat(chat))["agents"][0]
            assert root["can_message"] and root["can_cancel"]
            endpoint = f"/api/runs/{run}/agents/root/"
            response = await developer.dispatch(
                "POST", endpoint + "message", {"message": "Focus on evidence"}
            )
            assert response["delivered"] is True
            assert (
                service.store.events(run, kinds=("operator_message",))[0]["agent_id"]
                == "root"
            )
            await developer.dispatch("POST", endpoint + "cancel", {})
            await asyncio.wait_for(service.jobs[run], 2)
            root = (await developer.chat(chat))["agents"][0]
            assert root["status"] == "cancelled" and root["can_cancel"] is False
            with pytest.raises(ToolValidationError, match="active mission"):
                await developer.dispatch("POST", endpoint + "cancel", {})
        finally:
            await service.close()

    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is optional")
def test_git_inspection_filters_private_paths_and_disables_configured_programs(
    tmp_path,
):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "example.txt").write_text("old\n")
    (tmp_path / ".env").write_text("private old\n")
    (tmp_path / "*.py").write_text("literal old\n")
    (tmp_path / "other.py").write_text("other old\n")
    (tmp_path / ".gitattributes").write_text(
        "example.txt diff=fixture filter=fixture\n"
    )
    git("add", "example.txt", ".env", ".gitattributes", "*.py", "other.py")
    git("commit", "-m", "fixture")
    marker = tmp_path / "configured-program-executed"
    program = tmp_path / "git-helper.sh"
    program.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 0\n")
    program.chmod(0o755)
    git("config", "core.fsmonitor", str(program))
    git("config", "diff.external", str(program))
    git("config", "diff.fixture.textconv", str(program))
    git("config", "filter.fixture.clean", str(program))
    git("config", "filter.fixture.required", "true")
    (tmp_path / "example.txt").write_text("new\n")
    (tmp_path / ".env").write_text("private new\n")
    (tmp_path / "*.py").write_text("literal new\n")
    (tmp_path / "other.py").write_text("other new\n")

    async def scenario():
        service = service_for(tmp_path)
        developer = DeveloperWorkspace(service)
        try:
            status = await developer.dispatch("GET", "/api/workspace/git", {})
            assert status["available"] and any(
                row["path"] == "example.txt" for row in status["files"]
            )
            assert not any(row["path"] == ".env" for row in status["files"])
            diff = await developer.dispatch(
                "GET", "/api/workspace/diff/example.txt", {}
            )
            assert diff["available"] and "-old\n+new\n" in diff["diff"]
            assert not marker.exists()
            literal = await developer.dispatch("GET", "/api/workspace/diff/%2A.py", {})
            assert (
                "literal new" in literal["diff"] and "other new" not in literal["diff"]
            )
            with pytest.raises(ToolValidationError):
                await developer.dispatch("GET", "/api/workspace/diff/.env", {})
        finally:
            await service.close()

    asyncio.run(scenario())
