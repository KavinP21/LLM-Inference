"""Native catalog negotiation/replay and evidence diagnostics; no model skill claims."""

import asyncio
import json

import pytest
from forge_llm.agents.protocol import Generation, conservative_input_tokens
from forge_llm.agents.runtime import AgentRuntime, RuntimeConfig
from forge_llm.agents.store import AgentStore


def call(name, **arguments):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
    )


class NativeBackend:
    supports_native_tools = True

    def __init__(self, scripts, delay=0):
        self.scripts = scripts
        self.calls = []
        self.delay = delay

    async def generate(self, *args, **kwargs):
        raise AssertionError("native negotiation must not invoke the DSL backend")

    async def generate_action(self, messages, max_tokens, request_id, tool_specs):
        self.calls.append(
            (request_id, tuple(messages), json.loads(json.dumps(tool_specs)))
        )
        await asyncio.sleep(self.delay)
        agent, step = request_id.rsplit(".", 2)[-2:]
        return Generation(self.scripts[agent][int(step) - 1], 20, 10)

    async def cancel(self, request_id):
        pass


READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def names(specs):
    return {spec["function"]["name"] for spec in specs}


def test_native_calls_use_scoped_functions_tool_responses_and_parent_synthesis():
    backend = NativeBackend(
        {
            "root": [
                call("spawn", task="Review source.py"),
                call("wait", agents=["agent-0001"]),
                call("finish", result="Reviewed child evidence"),
            ],
            "agent-0001": [
                call("read_file", path="source.py"),
                call("finish", result="Verified source evidence"),
            ],
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={
            "read_file": lambda args: "Actual source",
            "write_file": lambda args: "Forbidden child edit",
        },
        tool_schemas={"read_file": READ_SCHEMA, "write_file": READ_SCHEMA},
        replay_safe_tools={"read_file"},
        config=RuntimeConfig(max_agents=2, max_depth=1),
    )
    result = asyncio.run(runtime.run("Review source.py; children must inspect only."))
    assert result.status == "completed"
    first = backend.calls[0]
    assert "PROVIDED FUNCTIONS" in first[1][0].content
    assert "CURRENT ACTION MENU" not in first[1][-1].content
    assert "exactly one JSON object" not in first[1][0].content
    assert "verified text or a JSON object" in first[1][0].content
    assert "Prefer an object" in first[1][0].content
    assert "never invent source contents" in first[1][0].content
    assert "For reviews omit spawn.tools" in first[1][0].content
    child = next(item for item in backend.calls if ".agent-0001.1" in item[0])
    assert "write_file" not in names(child[2]) and "spawn" not in names(child[2])
    child_finish = next(item for item in backend.calls if ".agent-0001.2" in item[0])
    assert any(
        message.role == "tool" and "Actual source" in message.content
        for message in child_finish[1]
    )
    waiting = next(item for item in backend.calls if ".root.2" in item[0])
    assert "finish" not in names(waiting[2]) and "wait" in names(waiting[2])
    final = next(item for item in backend.calls if ".root.3" in item[0])
    assert "wait" not in names(final[2]) and "finish" in names(final[2])
    assert any(
        message.role == "tool" and "Verified source evidence" in message.content
        for message in final[1]
    )


def test_native_resume_replays_exact_catalog_before_reviewing_new_messages(tmp_path):
    async def scenario():
        backend = NativeBackend(
            {
                "root": [
                    call("finish", result="Old answer"),
                    call("finish", result="Updated answer"),
                ]
            },
            delay=10,
        )
        store = AgentStore(tmp_path / "native.sqlite")
        tools = {"read_file": lambda args: "source"}
        runtime = AgentRuntime(
            backend,
            store,
            tools=tools,
            tool_schemas={"read_file": READ_SCHEMA},
            tool_descriptions={"read_file": "Original description"},
        )
        job = asyncio.create_task(runtime.run("Original task", run_id="native-replay"))
        while not backend.calls:
            await asyncio.sleep(0.001)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        store.add_message("native-replay", "root", "user", "New requirement")
        backend.delay = 0
        resumed = AgentRuntime(
            backend,
            store,
            tools=tools,
            tool_schemas={"read_file": READ_SCHEMA},
            tool_descriptions={"read_file": "Changed future description"},
        )
        result = await resumed.resume("native-replay")
        assert result.status == "completed" and result.output == "Updated answer"
        assert backend.calls[0] == backend.calls[1]
        assert backend.calls[0][2] != backend.calls[2][2]
        assert any(
            "New requirement" in message.content for message in backend.calls[2][1]
        )
        assert (
            store.generation_tool_specs("native-replay", "native-replay.root.1")
            == backend.calls[0][2]
        )
        store.close()

    asyncio.run(scenario())


def test_native_catalog_is_reserved_before_generation_and_context_bounded():
    backend = NativeBackend({"root": [call("finish", result="Unused")]})
    result = asyncio.run(
        AgentRuntime(backend, config=RuntimeConfig(max_total_tokens=1000)).run("Task")
    )
    assert result.status == "failed" and not backend.calls
    assert "token budget" in result.error
    backend = NativeBackend({"root": [call("finish", result="Valid")]})
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_context_tokens=8000))
    result = asyncio.run(runtime.run("Task"))
    request, messages, specs = backend.calls[0]
    assert result.status == "completed"
    assert (
        conservative_input_tokens(messages)
        + runtime._schema_input_bound(specs)
        + runtime.config.max_output_tokens
        <= 8000
    )
    assert runtime.store.generation_tool_specs(result.run_id, request) == specs


def test_native_backend_requires_explicit_parameter_schemas():
    with pytest.raises(ValueError, match="explicit parameter schemas"):
        AgentRuntime(NativeBackend({}), tools={"read_file": lambda args: "source"})


def test_native_invalid_function_is_repaired_without_dsl_instructions():
    backend = NativeBackend(
        {
            "root": [
                call("unregistered", command="bad"),
                call("finish", result="Recovered through supplied function"),
            ]
        }
    )
    result = asyncio.run(AgentRuntime(backend).run("Task"))
    assert result.status == "completed"
    feedback = backend.calls[1][1][-1].content
    assert "PROVIDED FUNCTION" in feedback and "CURRENT ACTION MENU" not in feedback


def test_diagnostic_excerpt_preserves_head_and_tail_within_limit():
    text = (
        "TESTS START\n"
        + "body\n" * 2000
        + "SyntaxError: invalid syntax\nFAILED tests/test_stats.py"
    )
    clipped = AgentRuntime._context_excerpt(text, 1000)
    assert len(clipped) <= 1000
    assert clipped.startswith("TESTS START") and clipped.endswith(
        "FAILED tests/test_stats.py"
    )
    assert "SyntaxError: invalid syntax" in clipped and "TRUNCATED middle" in clipped
    byte_clipped = AgentRuntime._byte_excerpt(
        "α" * 1000 + "DIAGNOSTIC END", 200, "[TRUNCATED middle]"
    )
    assert len(byte_clipped.encode()) <= 200 and byte_clipped.endswith("DIAGNOSTIC END")


def test_journal_defaults_to_evidence_and_control_is_explicit():
    runtime = AgentRuntime(NativeBackend({}))
    runtime.store.create_run("journal", "Task", vars(runtime.config), "system")
    runtime.store.event("journal", "root", "generation", {"request_id": "metadata"})
    runtime.store.event(
        "journal", "root", "tool_result", {"name": "read", "result": "Actual evidence"}
    )
    default = json.loads(runtime._read_journal("journal", "root", {}))
    assert [event["kind"] for event in default["events"]] == ["tool_result"]
    audit = json.loads(
        runtime._read_journal("journal", "root", {"include_control": True})
    )
    assert {event["kind"] for event in audit["events"]} == {
        "run_created",
        "generation",
        "tool_result",
    }
    with pytest.raises(TypeError):
        runtime._read_journal("journal", "root", {"include_control": "true"})


def test_default_journal_skips_recall_copies_without_recursive_payloads():
    runtime = AgentRuntime(NativeBackend({}))
    runtime.store.create_run("recall", "Task", vars(runtime.config), "system")
    runtime.store.event(
        "recall",
        "root",
        "tool_result",
        {"name": "read_file", "result": "Original source evidence"},
    )
    first = runtime._read_journal("recall", "root", {})
    runtime.store.event(
        "recall", "root", "tool_result", {"name": "journal_read", "result": first}
    )
    runtime.store.event("recall", "root", "generation", {"request_id": "metadata"})
    second = json.loads(runtime._read_journal("recall", "root", {}))
    assert len(second["events"]) == 1
    assert "Original source evidence" in second["events"][0]["payload_excerpt"]
    assert "journal_read" not in second["events"][0]["payload_excerpt"]
    assert second["next_after_event_id"] == runtime.store.events("recall")[-1]["id"]
    full = json.loads(
        runtime._read_journal("recall", "root", {"include_control": True, "limit": 5})
    )
    assert any("journal_read" in event["payload_excerpt"] for event in full["events"])


def test_journal_empty_pages_advance_over_skipped_rows_with_bounded_scan():
    runtime = AgentRuntime(NativeBackend({}))
    runtime.store.create_run("skip", "Task", vars(runtime.config), "system")
    for index in range(70):
        runtime.store.event(
            "skip",
            "root",
            "tool_result",
            {"name": "journal_read", "result": str(index)},
        )
    runtime.store.event(
        "skip",
        "root",
        "tool_result",
        {"name": "read_file", "result": "Next original evidence"},
    )
    page = json.loads(runtime._read_journal("skip", "root", {"limit": 1}))
    assert page["events"] == [] and page["next_after_event_id"] > 0
    # Exactly 64 own rows were visited; the following page can reach real data.
    assert page["next_after_event_id"] == runtime.store.events("skip")[63]["id"]
    next_page = json.loads(
        runtime._read_journal(
            "skip", "root", {"limit": 1, "after_event_id": page["next_after_event_id"]}
        )
    )
    assert len(next_page["events"]) == 1
    assert "Next original evidence" in next_page["events"][0]["payload_excerpt"]


def test_journal_offsets_still_page_original_evidence_across_skipped_records():
    runtime = AgentRuntime(NativeBackend({}))
    runtime.store.create_run("offset", "Task", vars(runtime.config), "system")
    runtime.store.event(
        "offset",
        "root",
        "tool_result",
        {"name": "journal_read", "result": "A copy to skip"},
    )
    payload = {"name": "read_file", "result": "0123456789" * 100}
    runtime.store.event("offset", "root", "tool_result", payload)
    first = json.loads(
        runtime._read_journal("offset", "root", {"limit": 1, "max_chars": 256})
    )
    event = first["events"][0]
    second = json.loads(
        runtime._read_journal(
            "offset",
            "root",
            {
                "limit": 1,
                "max_chars": 256,
                "offset": 256,
                "after_event_id": event["event_id"] - 1,
            },
        )
    )
    assert (
        event["payload_excerpt"] + second["events"][0]["payload_excerpt"]
        == json.dumps(payload)[:512]
    )
    assert second["events"][0]["event_id"] == event["event_id"]


def test_repeated_identical_tool_result_produces_no_progress_feedback():
    backend = NativeBackend(
        {
            "root": [call("read_file", path="same")] * 3
            + [call("finish", result="Existing evidence used")]
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={"read_file": lambda args: "Identical evidence"},
        tool_schemas={"read_file": READ_SCHEMA},
        replay_safe_tools={"read_file"},
    )
    result = asyncio.run(runtime.run("Read evidence"))
    assert result.status == "completed"
    assert "same result 3 times" in backend.calls[3][1][-1].content
    rejected = [
        event
        for event in runtime.store.events(result.run_id)
        if event["kind"] == "action_rejected"
    ]
    assert len(rejected) == 1 and "no new evidence" in rejected[0]["payload"]["error"]
