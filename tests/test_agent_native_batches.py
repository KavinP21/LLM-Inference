"""Native batches are bounded, fully checked and durable by individual effect."""

import asyncio
import json

import pytest
from forge_llm.agents.runtime import AgentRuntime, RuntimeConfig
from forge_llm.agents.store import AgentStore
from test_agent_runtime_native import READ_SCHEMA, NativeBackend, call


def test_two_native_spawns_are_one_model_turn_with_independent_children():
    backend = NativeBackend(
        {
            "root": [
                call("spawn", task="Inspect A") + call("spawn", task="Inspect B"),
                call("wait", agents=["agent-0001", "agent-0002"]),
                call("finish", result="Both reports synthesized"),
            ],
            "agent-0001": [call("finish", result="Report A")],
            "agent-0002": [call("finish", result="Report B")],
        }
    )
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_agents=3, max_depth=1))
    result = asyncio.run(runtime.run("Delegate two independent reviews."))
    assert result.status == "completed" and len(result.agents) == 3
    assert result.agents[0].steps == 3
    assert (
        runtime.store.effect(f"{result.run_id}/root/1/effect/batch/0")["agent_id"]
        == "agent-0001"
    )
    assert (
        runtime.store.effect(f"{result.run_id}/root/1/effect/batch/1")["agent_id"]
        == "agent-0002"
    )
    assert (
        len(
            [
                event
                for event in runtime.store.events(result.run_id)
                if event["kind"] == "batch_completed"
            ]
        )
        == 1
    )


def test_read_only_batch_records_each_result_with_two_matching_tool_responses():
    read = []
    backend = NativeBackend(
        {
            "root": [
                call("read_file", path="a") + call("read_file", path="b"),
                call("finish", result="Evidence A and B used"),
            ]
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={
            "read_file": lambda args: (
                read.append(args["path"]) or "Evidence " + args["path"]
            )
        },
        tool_schemas={"read_file": READ_SCHEMA},
        replay_safe_tools={"read_file"},
    )
    result = asyncio.run(runtime.run("Read two independent files"))
    assert result.status == "completed" and read == ["a", "b"]
    assert len(backend.calls) == 2
    messages = backend.calls[1][1]
    assert len([message for message in messages if message.role == "tool"]) == 2
    assert any(
        "Completed 2 batch calls" in message.content and message.role == "user"
        for message in messages
    )


@pytest.mark.parametrize(
    "batch",
    [
        call("spawn", task="Would create child")
        + call("finish", result="Mixed finish is forbidden"),
        call("read_file", path="a") + call("write_file", path="b"),
        call("read_file", path="same") + call("read_file", path="same"),
        call("read_file", path="a")
        + '<tool_call>{"name":"read_file","arguments":{"path":"b","unknown":1}}</tool_call>',
    ],
)
def test_forbidden_or_late_invalid_batch_has_no_partial_effects(batch):
    effects = []
    backend = NativeBackend(
        {"root": [batch, call("finish", result="Invalid batch rejected")]}
    )
    runtime = AgentRuntime(
        backend,
        tools={
            "read_file": lambda args: effects.append("read") or "data",
            "write_file": lambda args: effects.append("write") or "changed",
        },
        tool_schemas={"read_file": READ_SCHEMA, "write_file": READ_SCHEMA},
        replay_safe_tools={"read_file"},
    )
    result = asyncio.run(runtime.run("Task"))
    assert result.status == "completed" and not effects
    assert len(result.agents) == 1


def test_spawn_batch_capacity_preflight_creates_no_children():
    backend = NativeBackend(
        {
            "root": [
                call("spawn", task="A") + call("spawn", task="B"),
                call("finish", result="Capacity diagnostic accepted"),
            ]
        }
    )
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_agents=2))
    result = asyncio.run(runtime.run("Task"))
    assert result.status == "completed" and len(result.agents) == 1
    assert any(
        "remaining agent capacity" in event["payload"].get("error", "")
        for event in runtime.store.events(result.run_id)
    )


def test_resume_read_batch_skips_recorded_first_item_and_replays_safe_interrupted_item(
    tmp_path,
):
    async def scenario():
        started = asyncio.Event()
        counts = {"a": 0, "b": 0}
        interrupted = True

        async def read(args):
            nonlocal interrupted
            path = args["path"]
            counts[path] += 1
            if path == "b" and interrupted:
                started.set()
                await asyncio.sleep(10)
            return "Evidence " + path

        backend = NativeBackend(
            {
                "root": [
                    call("read_file", path="a") + call("read_file", path="b"),
                    call("finish", result="Resumed both results"),
                ]
            }
        )
        store = AgentStore(tmp_path / "batch.sqlite")
        options = {
            "tools": {"read_file": read},
            "tool_schemas": {"read_file": READ_SCHEMA},
            "replay_safe_tools": {"read_file"},
        }
        runtime = AgentRuntime(backend, store, **options)
        job = asyncio.create_task(runtime.run("Read files", run_id="batch-resume"))
        await asyncio.wait_for(started.wait(), 1)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert store.agent("batch-resume", "root").pending_action["action"] == "batch"
        assert store.effect("batch-resume/root/1/effect/batch/0") is not None
        interrupted = False
        result = await AgentRuntime(backend, store, **options).resume("batch-resume")
        assert result.status == "completed" and counts == {"a": 1, "b": 2}
        assert len(backend.calls) == 2
        assert (
            len(
                [
                    event
                    for event in store.events("batch-resume")
                    if event["kind"] == "tool_result"
                ]
            )
            == 2
        )
        store.close()

    asyncio.run(scenario())


def test_native_finish_object_normalizes_without_exposing_expected_answers():
    answer = {"measurement": 4.5, "qualified": False}
    backend = NativeBackend({"root": [call("finish", result=answer)]})
    result = asyncio.run(AgentRuntime(backend).run("Return a JSON report"))
    assert result.status == "completed" and json.loads(result.output) == answer


def test_repeated_read_batches_keep_every_effect_then_reject_no_progress():
    batch = call("read_file", path="a") + call("read_file", path="b")
    backend = NativeBackend(
        {"root": [batch] * 3 + [call("finish", result="Used the existing evidence")]}
    )
    runtime = AgentRuntime(
        backend,
        tools={"read_file": lambda args: "Unchanged " + args["path"]},
        tool_schemas={"read_file": READ_SCHEMA},
        replay_safe_tools={"read_file"},
    )
    result = asyncio.run(runtime.run("Read independent evidence"))
    assert result.status == "completed"
    events = runtime.store.events(result.run_id)
    assert len([event for event in events if event["kind"] == "tool_result"]) == 6
    assert len([event for event in events if event["kind"] == "batch_completed"]) == 3
    assert len([event for event in events if event["kind"] == "batch_no_progress"]) == 2
    rejected = [event for event in events if event["kind"] == "action_rejected"]
    assert (
        len(rejected) == 1 and "same result 3 times" in rejected[0]["payload"]["error"]
    )
    assert runtime.store.effect(f"{result.run_id}/root/3/effect/batch/0") is not None
    assert runtime.store.effect(f"{result.run_id}/root/3/effect/batch/1") is not None
    assert "no new evidence" in backend.calls[3][1][-1].content


def test_spawn_batch_resume_skips_committed_child_and_creates_only_missing_child():
    config = RuntimeConfig(max_agents=3, max_depth=1)
    backend = NativeBackend(
        {
            "root": [
                "unused",
                call("wait", agents=["agent-0001", "agent-0002"]),
                call("finish", result="Both resumed reports"),
            ],
            "agent-0001": [call("finish", result="A")],
            "agent-0002": [call("finish", result="B")],
        }
    )
    store = AgentStore()
    runtime = AgentRuntime(backend, store, config=config)
    store.create_run(
        "spawn-resume", "Task", vars(config), "system", tools=["journal_read"]
    )
    items = [{"action": "spawn", "task": "A"}, {"action": "spawn", "task": "B"}]
    store.update_agent(
        "spawn-resume",
        "root",
        steps=1,
        pending_action={"action": "batch", "items": items, "next_index": 0},
        action_phase="batch",
    )
    store.spawn(
        "spawn-resume",
        "root",
        "spawn-resume/root/1/effect/batch/0",
        "A",
        "Reviewer",
        [],
        "system",
        tools=["journal_read"],
        preserve_pending=True,
    )
    result = asyncio.run(runtime.resume("spawn-resume"))
    assert result.status == "completed" and len(result.agents) == 3
    assert [agent.task for agent in result.agents[1:]] == ["A", "B"]
    assert (
        len(
            [
                event
                for event in store.events("spawn-resume")
                if event["kind"] == "spawn"
            ]
        )
        == 2
    )
