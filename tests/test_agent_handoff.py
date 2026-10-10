"""Child reports are claims; handoffs carry bounded, scoped source observations."""

import asyncio
import json
from dataclasses import replace

import pytest
from forge_llm.agents.protocol import Generation, conservative_input_tokens
from forge_llm.agents.runtime import AgentRuntime, RuntimeConfig
from forge_llm.agents.store import AgentStore


def response(native, name, **args):
    if native:
        return (
            "<tool_call>"
            + json.dumps({"name": name, "arguments": args})
            + "</tool_call>"
        )
    return json.dumps({"action": name, **args})


class Backend:
    def __init__(self, native):
        self.supports_native_tools = native
        self.calls = []
        self.scripts = {
            "root": [
                response(native, "spawn", task="Review observed source"),
                response(native, "wait", agents=["agent-0001"]),
                response(native, "finish", result="Source contradicts the report"),
            ],
            "agent-0001": [
                response(
                    native,
                    "read_file",
                    **(
                        {"path": "source.txt"}
                        if native
                        else {"args": {"path": "source.txt"}}
                    ),
                ),
                response(native, "finish", result="The observation was approved"),
            ],
        }

    async def generate(self, messages, max_tokens, request_id):
        self.calls.append((request_id, tuple(messages)))
        agent, step = request_id.rsplit(".", 2)[-2:]
        return Generation(self.scripts[agent][int(step) - 1], 20, 10)

    async def generate_action(self, messages, max_tokens, request_id, tool_specs):
        return await self.generate(messages, max_tokens, request_id)


READ_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("native", [False, True])
def test_false_child_claim_and_actual_source_survive_handoff_projection_and_resume(
    tmp_path, native
):
    observed = {
        "path": "source.txt",
        "start_line": 1,
        "total_lines": 2,
        "truncated": False,
        "content": "1: Observation: rejected\n2: Reason: calibration failed",
    }
    backend = Backend(native)
    path = tmp_path / "handoff.sqlite"
    store = AgentStore(path)
    config = RuntimeConfig(max_agents=2, max_depth=1, max_tool_result_chars=2000)
    tools = {"read_file": lambda args: json.dumps(observed)}
    runtime = AgentRuntime(
        backend,
        store,
        tools=tools,
        replay_safe_tools={"read_file"},
        tool_schemas={"read_file": READ_SCHEMA},
        config=config,
    )
    result = asyncio.run(runtime.run("Review actual observations", run_id="handoff"))
    assert result.status == "completed"
    messages = store.messages("handoff", "root")
    delivered = next(
        message for message in messages if message.content.startswith("Child outcomes")
    )
    summary = json.loads(delivered.content.split("\n", 1)[1])
    assert summary["result"] == "The observation was approved"
    assert summary["report_kind"] == "unverified_child_claim"
    (receipt,) = summary["observed_evidence"]
    assert receipt["args"] == {"path": "source.txt"}
    assert receipt["observed_source"]["content_excerpt"] == observed["content"]
    assert receipt["observed_source"]["start_line"] == 1
    assert "observed_result_excerpt" not in receipt  # File JSON is unpacked.
    page = json.loads(
        runtime._read_journal("handoff", "root", receipt["journal_cursor"])
    )
    assert page["events"][0]["event_id"] == receipt["event_id"]
    assert "calibration failed" in page["events"][0]["payload_excerpt"]
    prompt = messages[0].content
    assert "Child conclusions are unverified claims" in prompt
    assert "Missing measurements remain unknown" in prompt

    # Project a long earlier history while retaining the latest actual handoff.
    store.connection.execute(
        "UPDATE messages SET content=? WHERE run_id=? AND agent_id=? AND role='assistant' AND id=(SELECT MIN(id) FROM messages WHERE run_id=? AND agent_id=? AND role='assistant')",
        ("Earlier dialogue " * 2000, "handoff", "root", "handoff", "root"),
    )
    history = store.messages("handoff", "root")
    compact = replace(
        config, max_context_tokens=conservative_input_tokens(history[:2]) + 5000
    )
    projected = runtime._model_context("handoff", "root", compact)
    combined = "\n".join(message.content for message in projected)
    assert "History projection" in combined
    assert "The observation was approved" in combined
    assert (
        observed["content"] not in combined
    )  # JSON wire escapes preserve line breaks.
    assert "Observation: rejected" in combined and "calibration failed" in combined
    assert str(receipt["event_id"]) in combined
    assert (
        conservative_input_tokens(projected) + compact.max_output_tokens
        <= compact.max_context_tokens
    )

    # Outcome delivery is a durable snapshot; restart never regenerates it.
    store.event(
        "handoff",
        "agent-0001",
        "tool_result",
        {"name": "read_file", "args": {"path": "late.txt"}, "result": "Later receipt"},
    )
    store.close()
    reopened = AgentStore(path)
    resumed = AgentRuntime(
        backend,
        reopened,
        tools=tools,
        replay_safe_tools={"read_file"},
        tool_schemas={"read_file": READ_SCHEMA},
        config=config,
    )
    assert asyncio.run(resumed.resume("handoff")).status == "completed"
    outcomes = [
        message
        for message in reopened.messages("handoff", "root")
        if message.content.startswith("Child outcomes")
    ]
    assert outcomes == [delivered]
    assert len(backend.calls) == 5


def test_handoff_filters_noise_mutations_other_agents_and_other_runs_and_bounds_payload():
    config = RuntimeConfig()
    runtime = AgentRuntime(
        Backend(False),
        tools={
            name: lambda args: "unused"
            for name in ("read_file", "search", "list_files", "write_file")
        },
        replay_safe_tools={"read_file", "search", "list_files", "write_file"},
    )
    store = runtime.store
    store.create_run("one", "Root", vars(config), "system")
    first = store.spawn(
        "one", "root", "first", "Inspect source", "review", [], "system"
    )
    second = store.spawn("one", "root", "second", "Other scope", "review", [], "system")
    grandchild = store.spawn(
        "one", first, "grandchild", "Nested scope", "review", [], "system"
    )
    store.create_run("two", "Unrelated run", vars(config), "system")
    other = store.spawn("two", "root", "other", "Other run", "review", [], "system")
    eligible = []
    for index in range(5):
        store.event(
            "one",
            first,
            "tool_result",
            {
                "name": "read_file",
                "args": {"path": f"record-{index}.txt"},
                "result": json.dumps(
                    {
                        "content": f"Record {index}: actual observation\n"
                        + "long source " * 2000,
                        "start_line": 1,
                    }
                ),
            },
        )
        eligible.append(store.events("one")[-1]["id"])
    for run_id, agent_id in (("one", second), ("one", grandchild), ("two", other)):
        store.event(
            run_id,
            agent_id,
            "tool_result",
            {"name": "read_file", "args": {}, "result": "SCOPED_SECRET"},
        )
    for name, value in (
        ("journal_read", "Recursive journal noise"),
        ("list_files", "Listing noise"),
        ("search", '{"matches": []}'),
        ("write_file", "Mutation result"),
        ("read_file", "Tool failed: unavailable"),
    ):
        store.event(
            "one", first, "tool_result", {"name": name, "args": {}, "result": value}
        )
    store.update_agent(
        "one", first, status="completed", result="Unverified claim " * 1000
    )
    summary_text = runtime._summary(store.agent("one", first), 2000)
    assert len(summary_text) <= 2000
    summary = json.loads(summary_text)
    receipts = summary["observed_evidence"]
    assert 2 <= len(receipts) <= 3
    assert {item["event_id"] for item in receipts} <= set(eligible[-3:])
    assert len(runtime._handoff_receipts(store.agent("one", first), 500)) == 3
    assert all(item["tool"] == "read_file" for item in receipts)
    assert "SCOPED_SECRET" not in summary_text
    assert (
        "Recursive journal noise" not in summary_text
        and "Mutation result" not in summary_text
    )
    assert "TRUNCATED" in summary_text
    assert summary["omitted_evidence_event_ids"]
    assert runtime._summary(store.agent("one", first), 2000) == summary_text
