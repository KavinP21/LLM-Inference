"""Operator lifecycle against the real agent journal and scheduler, without GPU."""

import asyncio
import json

import pytest
from forge_llm.agents.context import tool_actor
from forge_llm.agents.tools import ToolValidationError
from forge_llm.harness.extensions import SkillRegistry
from forge_llm.harness.service import StudioService


def make_service(tmp_path, **kwargs):
    (tmp_path / "README.md").write_text("# Fixture\nObserved evidence.\n")
    service = StudioService(tmp_path, tmp_path / ".forge", demo=True, **kwargs)
    service.backend.delay = 0.01
    return service


async def settle(service, run_id):
    await asyncio.wait_for(service.jobs[run_id], 5)
    return await service.detail(run_id)


def test_durable_subagents_export_and_followup(tmp_path):
    async def check():
        service = make_service(tmp_path)
        try:
            first = (await service.create_run({"task": "Inspect evidence"}))["run_id"]
            detail = await settle(service, first)
            assert detail["run"]["status"] == "completed", detail["error"]
            assert len(detail["agents"]) == 3
            assert all(a["status"] == "completed" for a in detail["agents"])
            assert "without model inference" in detail["output"]
            assert detail["usage"]["reserved_tokens"] == 0
            assert detail["usage"]["total_tokens"] > 0
            assert not any("write_file" in a["tools"] for a in detail["agents"])
            events = (await service.dispatch("GET", f"/api/runs/{first}/export", {}))[
                "events"
            ]
            assert any(e["kind"] == "tool_result" for e in events)
            second = (
                await service.dispatch(
                    "POST",
                    f"/api/runs/{first}/fork",
                    {"task": "Explain the inspection"},
                )
            )["run_id"]
            assert (await settle(service, second))["lineage"]["parent_id"] == first
            assert "Previous mission result" in service.store.agent(second, "root").task
            with pytest.raises(ValueError, match="Only interrupted"):
                await service.resume(first)
        finally:
            await service.close()
        restarted = StudioService(tmp_path, tmp_path / ".forge", demo=True)
        try:
            state = await restarted.state()
            assert len(state["runs"]) == 2
            assert all(r["status"] == "completed" for r in state["runs"])
        finally:
            await restarted.close()

    asyncio.run(check())


def test_review_requires_exact_approval_and_artifact_executes_once(tmp_path):
    async def check():
        service = make_service(tmp_path)
        try:
            run_id = (
                await service.create_run(
                    {"task": "Exercise review", "policy": "review"}
                )
            )["run_id"]
            for _ in range(300):
                detail = await service.detail(run_id)
                if detail["approvals"]:
                    break
                await asyncio.sleep(0.01)
            assert len(detail["approvals"]) == 1
            approval = detail["approvals"][0]
            assert approval["tool"] == "save_artifact"
            assert "demo-inspection.md" in approval["preview"]["diff"]
            artifact = tmp_path / ".forge" / "artifacts" / run_id / "demo-inspection.md"
            assert not artifact.exists()
            await service.decide(approval["id"], {"approved": True})
            with pytest.raises(ValueError, match="no longer pending"):
                await service.decide(approval["id"], {"approved": True})
            finished = await settle(service, run_id)
            assert finished["run"]["status"] == "completed"
            assert "Saved `demo-inspection.md`" in finished["output"]
            assert str(service.state_dir) not in finished["output"]
            assert artifact.is_file()
            assert "No model inference" in artifact.read_text()
            downloaded = await service.dispatch(
                "GET", f"/api/runs/{run_id}/artifacts/demo-inspection.md", {}
            )
            assert downloaded["content"] == artifact.read_text()
            with pytest.raises(ToolValidationError, match="inside"):
                await service.dispatch(
                    "GET", f"/api/runs/{run_id}/artifacts/%2e%2e%2fstudio.sqlite", {}
                )
            (artifact.parent / "escape.md").symlink_to(tmp_path / "README.md")
            with pytest.raises(ToolValidationError, match="symlink"):
                await service.dispatch(
                    "GET", f"/api/runs/{run_id}/artifacts/escape.md", {}
                )
            assert not (await service.detail(run_id))["approvals"]
        finally:
            await service.close()

    asyncio.run(check())


def test_steering_and_cancel_are_durable(tmp_path):
    async def check():
        service = make_service(tmp_path)
        service.backend.delay = 1
        try:
            run_id = (await service.create_run({"task": "Inspect evidence"}))["run_id"]
            await service.message(run_id, {"message": "Focus on actual file receipts"})
            assert (
                "Focus on actual file receipts"
                in service.store.messages(run_id, "root")[-1].content
            )
            await service.cancel(run_id)
            detail = await settle(service, run_id)
            assert detail["run"]["status"] == "cancelled"
            assert any(e["kind"] == "operator_message" for e in detail["events"])
            with pytest.raises(ValueError, match="active agent"):
                await service.message(run_id, {"message": "late message"})
        finally:
            await service.close()

    asyncio.run(check())


def test_recovery_preserves_manifest_and_rejects_changed_backend(tmp_path):
    async def check():
        service = make_service(tmp_path)
        service.backend.delay = 1
        run_id = (await service.create_run({"task": "Inspect evidence"}))["run_id"]
        await asyncio.sleep(0.02)
        await service.close()
        service = StudioService(tmp_path, tmp_path / ".forge", demo=True)
        service.backend.delay = 0.01
        try:
            assert (await service.detail(run_id))["run"]["status"] == "interrupted"
            service.config = {"runtime": {"max_agents": 8}}
            with pytest.raises(ValueError, match="original workspace"):
                await service.resume(run_id)
            service.config = {}
            await service.resume(run_id)
            detail = await settle(service, run_id)
            assert detail["run"]["status"] == "completed", detail["error"]
            assert detail["usage"]["uncertain_tokens"] > 0
        finally:
            await service.close()

    asyncio.run(check())


def test_private_state_and_read_only_mutations_are_unavailable(tmp_path):
    async def check():
        service = make_service(tmp_path)
        try:
            run_id = (await service.create_run({"task": "Inspect evidence"}))["run_id"]
            runtime = service.runtimes[run_id]
            assert (
                not {"save_artifact", "write_file", "run_tests"} & runtime.tools.keys()
            )
            token = tool_actor.set((run_id, "root"))
            try:
                with pytest.raises(ToolValidationError, match="private"):
                    await runtime.tools["read_file"]({"path": ".forge/studio.sqlite"})
                files = json.loads(await runtime.tools["list_files"]({"path": "."}))[
                    "files"
                ]
                assert not any(".forge" in name for name in files)
            finally:
                tool_actor.reset(token)
            await service.cancel(run_id)
            await settle(service, run_id)
        finally:
            await service.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    "payload",
    [
        {"task": ""},
        {"task": "valid", "policy": "unrestricted"},
        {"task": "valid", "max_agents": True},
        {"task": "valid", "max_agents": 17},
        {"task": "valid", "max_total_tokens": 10},
        {"task": "valid", "shell": "anything"},
    ],
)
def test_invalid_mission_rejected_before_run_creation(tmp_path, payload):
    async def check():
        service = make_service(tmp_path)
        try:
            with pytest.raises(ValueError):
                await service.create_run(payload)
            assert not (await service.state())["runs"]
        finally:
            await service.close()

    asyncio.run(check())


def test_operator_selected_skill_loaded_and_cataloged(tmp_path):
    async def check():
        skills_path = tmp_path / "skills"
        skill = skills_path / "review"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: review\ndescription: Inspect actual evidence\n---\nCite receipts.\n"
        )
        registry = SkillRegistry([skills_path])
        service = make_service(tmp_path, skills=registry)
        try:
            run_id = (
                await service.create_run(
                    {"task": "Inspect evidence", "skill": "review"}
                )
            )["run_id"]
            assert "Cite receipts" in service.store.agent(run_id, "root").task
            assert (await service.state())["skills"][0]["name"] == "review"
            assert "skill_load" in service.runtimes[run_id].tools
            await settle(service, run_id)
        finally:
            await service.close()

    asyncio.run(check())
