"""Portable policy tests; approvals control invocation, not model capabilities."""

import asyncio
import hashlib
import json
import time
from contextlib import contextmanager

import pytest
from instinct.agents.context import tool_actor
from instinct.agents.store import AgentStore
from instinct.agents.tools import ToolValidationError, WorkspaceTools
from instinct.harness.policy import ApprovalGate, HookRegistry, workspace_preview


@pytest.fixture
def store(tmp_path):
    journal = AgentStore(tmp_path / "agents.sqlite")
    journal.create_run("one", "task", {"deadline_seconds": 30}, "system")
    journal.create_run("two", "task", {"deadline_seconds": 30}, "system")
    yield journal
    journal.close()


@contextmanager
def actor(run_id="one", agent_id="root"):
    token = tool_actor.set((run_id, agent_id))
    try:
        yield
    finally:
        tool_actor.reset(token)


async def requested(gate, run_id="one"):
    for _ in range(100):
        if rows := gate.pending(run_id):
            return rows[0]
        await asyncio.sleep(0.005)
    raise AssertionError("tool did not request approval")


def status(store, approval_id):
    return store.connection.execute(
        "SELECT status FROM harness_approvals WHERE id=?", (approval_id,)
    ).fetchone()[0]


def test_reads_bypass_approval_and_read_only_rejects_mutations(store):
    calls = []

    async def tool(args):
        calls.append(args)
        return "ok"

    async def scenario():
        gate = ApprovalGate(store, "read_only", run_id="one")
        with actor():
            assert await gate.wrap("read", tool)({"path": "source"}) == "ok"
            with pytest.raises(ToolValidationError, match="read_only") as caught:
                await gate.wrap("write", tool, mutating=True)({})
        assert caught.value.work_started is False
        assert not gate.pending("one")
        assert len(calls) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("stop_mode", ["requested", "parent_failed"])
def test_agent_stop_request_invalidates_review_before_scheduler_cleanup(
    store, stop_mode
):
    async def scenario():
        child = store.spawn(
            "one", "root", "test-child", "child task", "reviewer", [], "system"
        )
        calls = []

        async def tool(args):
            calls.append(args)
            return "must not execute"

        gate = ApprovalGate(store, run_id="one")
        with actor(agent_id=child):
            task = asyncio.create_task(gate.wrap("write", tool, mutating=True)({}))
            row = await requested(gate)
            if stop_mode == "requested":
                store.event("one", child, "agent_cancel_requested", {"agent_id": child})
            else:
                store.update_agent("one", "root", status="failed")
            with pytest.raises(ToolValidationError, match="cancellation|parent"):
                gate.decide(row["id"], True)
            with pytest.raises(ToolValidationError, match="cancellation|parent"):
                await asyncio.wait_for(task, 2)
        assert not calls

    asyncio.run(scenario())


def test_approval_is_exact_one_shot_and_durable(store):
    calls = []

    async def tool(args):
        calls.append(args)
        return "receipt"

    async def scenario():
        gate = ApprovalGate(store, run_id="one")
        wrapped = gate.wrap("write", tool, mutating=True)
        original = {"z": 1, "nested": {"value": [1, 2]}}
        with actor():
            task = asyncio.create_task(wrapped(original))
            row = await requested(gate)
            assert calls == []
            assert row["agent_id"] == "root" and row["status"] == "pending"
            canonical = '{"nested":{"value":[1,2]},"z":1}'
            assert row["args_hash"] == hashlib.sha256(canonical.encode()).hexdigest()
            original["nested"]["value"].append(99)
            gate.decide(row["id"], True)
            assert calls == []  # decide only wakes the existing task.
            with pytest.raises(ToolValidationError, match="already"):
                gate.decide(row["id"], True)
            assert await task == "receipt"
            assert calls == [{"nested": {"value": [1, 2]}, "z": 1}]
            assert status(store, row["id"]) == "completed"
            with pytest.raises(ToolValidationError, match="stale"):
                gate.decide(row["id"], True)
            again = asyncio.create_task(wrapped({"z": 1, "nested": {"value": [1, 2]}}))
            next_row = await requested(gate)
            assert next_row["id"] != row["id"]
            gate.decide(next_row["id"], False)
            with pytest.raises(ToolValidationError, match="denied") as caught:
                await again
            assert caught.value.work_started is False
            assert status(store, next_row["id"]) == "denied"
            assert len(calls) == 1
        events = store.events("one", kinds=("approval_requested",))
        assert len(events) == 2 and events[0]["payload"]["args"] == calls[0]

    asyncio.run(scenario())
    with AgentStore(store.path) as reopened:
        rows = list(reopened.connection.execute("SELECT status FROM harness_approvals"))
        assert {row[0] for row in rows} == {"completed", "denied"}


@pytest.mark.parametrize(
    "args",
    [{"bad": float("nan")}, {"bad": float("inf")}, {1: "bad"}, {"bad": (1, 2)}, [1]],
)
def test_non_json_arguments_never_reach_tool(store, args):
    async def tool(args):
        pytest.fail("invalid arguments reached tool")

    with pytest.raises(ToolValidationError):
        asyncio.run(ApprovalGate(store).wrap("read", tool)(args))


def test_timeout_cancellation_and_cancelled_run_never_execute(store):
    async def tool(args):
        pytest.fail("unapproved tool executed")

    async def scenario():
        gate = ApprovalGate(store, timeout_seconds=0.03, run_id="one")
        with actor():
            task = asyncio.create_task(gate.wrap("write", tool, mutating=True)({}))
            row = await requested(gate)
            with pytest.raises(ToolValidationError, match="expired"):
                await task
            assert status(store, row["id"]) == "expired"
            with pytest.raises(ToolValidationError):
                gate.decide(row["id"], True)
            gate = ApprovalGate(store, run_id="one")
            task = asyncio.create_task(gate.wrap("write", tool, mutating=True)({}))
            row = await requested(gate)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert status(store, row["id"]) == "cancelled"
            task = asyncio.create_task(gate.wrap("write", tool, mutating=True)({}))
            row = await requested(gate)
            store.request_cancel("one")
            assert gate.pending("one") == []
            with pytest.raises(ToolValidationError):
                await task
            assert status(store, row["id"]) == "cancelled"

    asyncio.run(scenario())


def test_expired_decision_is_persisted_immediately(store):
    async def tool(args):
        pytest.fail("expired tool executed")

    async def scenario():
        gate = ApprovalGate(store, run_id="one")
        with actor():
            task = asyncio.create_task(gate.wrap("write", tool, mutating=True)({}))
            row = await requested(gate)
            store.connection.execute(
                "UPDATE harness_approvals SET expires=? WHERE id=?",
                (time.time() - 1, row["id"]),
            )
            with pytest.raises(ToolValidationError, match="expired"):
                gate.decide(row["id"], True)
            assert status(store, row["id"]) == "expired"
            with pytest.raises(ToolValidationError):
                await task

    asyncio.run(scenario())


def test_restart_invalidates_only_its_run_and_grants_cannot_cross_gates(store):
    async def tool(args):
        return "ok"

    async def scenario():
        one = ApprovalGate(store, run_id="one")
        with actor():
            task = asyncio.create_task(one.wrap("write", tool, mutating=True)({}))
            row = await requested(one)
            two = ApprovalGate(store, run_id="two")
            assert status(store, row["id"]) == "pending"
            with pytest.raises(ToolValidationError, match="another gate"):
                two.decide(row["id"], True)
            recovered = ApprovalGate(store, run_id="one")
            assert status(store, row["id"]) == "stale"
            assert recovered.pending("one") == []
            with pytest.raises(ToolValidationError):
                await task

    asyncio.run(scenario())


def test_actor_and_decision_loop_cannot_be_substituted(store):
    async def tool(args):
        return "ok"

    async def foreign_decision(gate, approval_id):
        gate.decide(approval_id, True)

    async def scenario():
        gate = ApprovalGate(store, run_id="one")
        with actor("two"):
            with pytest.raises(ToolValidationError, match="another policy run"):
                await gate.wrap("write", tool, mutating=True)({})
        with actor():
            task = asyncio.create_task(gate.wrap("write", tool, mutating=True)({}))
            row = await requested(gate)
            with pytest.raises(ToolValidationError, match="another gate"):
                await asyncio.to_thread(
                    lambda: asyncio.run(foreign_decision(gate, row["id"]))
                )
            store.connection.execute(
                "UPDATE harness_approvals SET args='{}',args_hash='tampered' WHERE id=?",
                (row["id"],),
            )
            with pytest.raises(ToolValidationError, match="exact invocation"):
                gate.decide(row["id"], True)
            with pytest.raises(ToolValidationError):
                await task
            assert status(store, row["id"]) == "stale"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "tool_name,args",
    [
        ("write_file", {"path": "example.txt", "content": "first\nNEW\nthird\n"}),
        (
            "replace_text",
            {"path": "example.txt", "old_text": "second", "new_text": "NEW"},
        ),
        (
            "edit_lines",
            {"path": "example.txt", "start_line": 2, "end_line": 2, "content": "NEW"},
        ),
    ],
)
def test_workspace_diff_matches_executed_content(store, tmp_path, tool_name, args):
    path = tmp_path / "example.txt"
    path.write_text("first\nsecond\nthird\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)

    async def scenario():
        gate = ApprovalGate(
            store,
            run_id="one",
            preview=lambda name, args: workspace_preview(tools, name, args),
        )
        with actor():
            await tools.read_file({"path": "example.txt"})
            task = asyncio.create_task(
                gate.wrap(tool_name, getattr(tools, tool_name), mutating=True)(args)
            )
            row = await requested(gate)
            assert "-second\n+NEW\n" in row["preview"]["diff"]
            assert path.read_text() == "first\nsecond\nthird\n"
            gate.decide(row["id"], True)
            await task
            assert path.read_text() == "first\nNEW\nthird\n"
            assert (
                row["preview"]["after_sha256"]
                == hashlib.sha256(path.read_bytes()).hexdigest()
            )

    asyncio.run(scenario())


def test_reviewed_diff_cannot_change_when_a_new_read_updates_receipt(store, tmp_path):
    path = tmp_path / "example.txt"
    path.write_text("old\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)

    async def scenario():
        gate = ApprovalGate(
            store,
            run_id="one",
            preview={
                "write_file": lambda args: workspace_preview(tools, "write_file", args)
            },
        )
        with actor():
            await tools.read_file({"path": "example.txt"})
            task = asyncio.create_task(
                gate.wrap("write_file", tools.write_file, mutating=True)(
                    {"path": "example.txt", "content": "new\n"}
                )
            )
            row = await requested(gate)
            path.write_text("other actor changed it\n")
            await tools.read_file({"path": "example.txt"})
            gate.decide(row["id"], True)
            with pytest.raises(ToolValidationError, match="preview changed"):
                await task
            assert path.read_text() == "other actor changed it\n"
            assert status(store, row["id"]) == "stale"

    asyncio.run(scenario())


def test_preview_validates_confinement_protection_and_python(store, tmp_path):
    path = tmp_path / "example.py"
    path.write_text("value = 1\n")
    tools = WorkspaceTools(tmp_path, tmp_path / "artifacts", allow_write=True)
    with actor():
        asyncio.run(tools.read_file({"path": "example.py"}))
        with pytest.raises(ToolValidationError, match="syntax"):
            workspace_preview(
                tools, "write_file", {"path": "example.py", "content": "return 1"}
            )
        with pytest.raises(ToolValidationError, match="selected directory"):
            workspace_preview(
                tools, "write_file", {"path": "../escape.txt", "content": "bad"}
            )
        tools.protect_files(["example.py"])
        with pytest.raises(ToolValidationError, match="protected"):
            workspace_preview(
                tools, "write_file", {"path": "example.py", "content": "value = 2\n"}
            )
    assert path.read_text() == "value = 1\n"


@pytest.mark.parametrize("behavior", ["reject", "exception", "timeout"])
def test_failed_before_hook_rejects_without_tool_or_approval(store, behavior):
    async def before(payload):
        if behavior == "reject":
            return False
        if behavior == "exception":
            raise RuntimeError("extension failed")
        await asyncio.sleep(1)

    async def tool(args):
        pytest.fail("rejected hook allowed execution")

    async def scenario():
        hooks = HookRegistry(timeout_seconds=0.02)
        await hooks.register("before_tool", before)
        gate = ApprovalGate(store, run_id="one", hooks=hooks)
        with actor():
            with pytest.raises(ToolValidationError) as caught:
                await gate.wrap("write", tool, mutating=True)({})
            assert caught.value.work_started is False
            assert gate.pending("one") == []

    asyncio.run(scenario())


def test_hooks_are_ordered_isolated_and_after_failure_preserves_receipt(store):
    observed = []

    async def first(payload):
        observed.append("first")
        payload["args"]["value"] = "tampered"

    async def second(payload):
        observed.append(payload["args"]["value"])

    async def after(payload):
        observed.append(payload["result"])
        raise RuntimeError("after failed")

    async def tool(args):
        observed.append(args["value"])
        return json.dumps({"executed": True})

    async def scenario():
        hooks = HookRegistry()
        await hooks.register("before_tool", first)
        await hooks.register("before_tool", second)
        await hooks.register("after_tool", after)
        with pytest.raises(TypeError, match="async"):
            await hooks.register("before_tool", lambda payload: None)
        gate = ApprovalGate(store, run_id="one", hooks=hooks)
        with actor():
            assert json.loads(await gate.wrap("read", tool)({"value": "original"})) == {
                "executed": True
            }
        assert observed == ["first", "original", "original", '{"executed": true}']
        assert (
            store.events("one", kinds=("hook_failed",))[0]["payload"]["event"]
            == "after_tool"
        )

    asyncio.run(scenario())
