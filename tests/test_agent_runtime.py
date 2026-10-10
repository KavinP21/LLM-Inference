"""Portable coordinator tests: scripted decisions test control, not model skill."""

import asyncio
import json
import time

import pytest
from forge_llm.agents.context import tool_actor
from forge_llm.agents.protocol import Generation, ProtocolError, parse_action
from forge_llm.agents.runtime import AgentRuntime, RuntimeConfig
from forge_llm.agents.store import AgentStore, BudgetError, LeaseError


def action(kind, **kwargs):
    return json.dumps({"action": kind, **kwargs})


class ScriptBackend:
    def __init__(self, scripts, delay=0):
        self.scripts = scripts
        self.delay = delay
        self.calls = []
        self.active = 0
        self.peak = 0
        self.cancelled = []

    async def generate(self, messages, max_tokens, request_id):
        agent_id, turn = request_id.rsplit(".", 2)[-2:]
        self.calls.append((request_id, tuple(messages)))
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            response = self.scripts[agent_id][int(turn) - 1]
            if isinstance(response, Exception):
                raise response
            return Generation(response, input_tokens=20, output_tokens=10)
        finally:
            self.active -= 1

    async def cancel(self, request_id):
        self.cancelled.append(request_id)


def run(runtime, task="Inspect evidence and report.", **kwargs):
    return asyncio.run(runtime.run(task, **kwargs))


def test_finish_and_journal_round_trip(tmp_path):
    path = tmp_path / "agents.sqlite"
    backend = ScriptBackend(
        {"root": [action("finish", result="Evidence-backed answer")]}
    )
    store = AgentStore(path)
    result = run(AgentRuntime(backend, store), run_id="one")
    assert result.status == "completed"
    assert result.output == "Evidence-backed answer"
    assert result.usage.total_tokens == 30
    assert "finish" in [event["kind"] for event in store.events("one")]
    store.close()
    reopened = AgentStore(path)
    assert reopened.agent("one", "root").result == result.output
    assert (
        asyncio.run(AgentRuntime(backend, reopened).resume("one")).output
        == result.output
    )
    assert len(backend.calls) == 1
    reopened.close()


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"action":"finish","result":"x"}\n```',
        '{"action":"finish","result":"a","result":"b"}',
        '{"action":"finish","result":"x","extra":1}',
        '{"action":"tool","name":"x","args":[],"extra":0}',
        '{"action":"wait","agents":["x","x"]}',
        '{"action":"wait","agents":[]}',
        '{"action":"finish","result":42}',
        '{"action":"finish","result":""}',
        '{"action":"tool","name":"x","args":{"number":NaN}}',
        '{"action":"tool","name":"x","args":{"number":1e309}}',
        '[{"action":"finish","result":"x"}]',
        '{"action":"finish","result":"x"} {"action":"finish","result":"y"}',
    ],
)
def test_strict_protocol_rejects_ambiguous_output(text):
    with pytest.raises(ProtocolError):
        parse_action(text)


def test_registered_direct_tool_alias_is_exact_and_closed():
    expected = {"action": "tool", "name": "read_file", "args": {"path": "source.py"}}
    assert (
        parse_action(
            action("read_file", args={"path": "source.py"}), allowed_tools={"read_file"}
        )
        == expected
    )
    with pytest.raises(ProtocolError, match="unknown action"):
        parse_action(action("read_file", args={"path": "source.py"}))
    with pytest.raises(ProtocolError, match="exactly action and args"):
        parse_action(
            action("read_file", args={}, name="source.py"), allowed_tools={"read_file"}
        )
    with pytest.raises(ProtocolError):
        parse_action(action("read_fil", args={}), allowed_tools={"read_file"})
    with pytest.raises(ProtocolError):
        parse_action(action("read_file", args=[]), allowed_tools={"read_file"})


def test_nullable_optional_role_selects_default_but_other_fields_stay_strict():
    parsed = parse_action(action("spawn", task="Inspect evidence", role=None))
    assert parsed == {"action": "spawn", "task": "Inspect evidence"}
    with pytest.raises(ProtocolError):
        parse_action(action("spawn", task="Inspect evidence", role=""))
    with pytest.raises(ProtocolError):
        parse_action(action("spawn", task="Inspect evidence", dependencies=None))
    with pytest.raises(ProtocolError):
        parse_action(action("spawn", task="Inspect evidence", tools=None))


def test_repair_is_bounded_and_can_recover():
    backend = ScriptBackend(
        {"root": ["bad JSON", action("finish", result="Recovered")]}
    )
    runtime = AgentRuntime(backend)
    result = run(runtime)
    assert result.status == "completed"
    assert "Action rejected" in backend.calls[1][1][-1].content
    broken = AgentRuntime(ScriptBackend({"root": ["bad"] * 10}))
    failed = run(broken)
    assert failed.status == "failed"
    assert len(broken.backend.calls) == 3
    assert "repair limit" in failed.error


def menu_from(messages):
    content = messages[-1].content
    start = content.rindex("CURRENT ACTION MENU")
    body = (
        content[start:]
        .split("\n", 1)[1]
        .rsplit("\nReply with exactly one action object.", 1)[0]
    )
    return json.loads(body)


def test_legal_menu_uses_real_tools_and_omits_unavailable_agent_actions():
    backend = ScriptBackend(
        {
            "root": [
                action("fix", task="stats.py"),
                action("tool", name="read_file", args={"path": "stats.py"}),
                action("finish", result="Source reviewed"),
            ]
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={"read_file": lambda args: "Actual source"},
        replay_safe_tools={"read_file"},
        config=RuntimeConfig(max_agents=1),
    )
    result = run(runtime, "Read stats.py and report findings.")
    assert result.status == "completed"
    initial = menu_from(backend.calls[0][1])
    assert set(initial["allowed_action_schemas"]) == {
        "read_file",
        "journal_read",
        "finish",
    }
    assert initial["your_agent_id"] == "root"
    assert initial["direct_children"] == [] and initial["active_recipients"] == []
    example = initial["valid_syntax_examples_for_this_state"][0]
    assert example == {
        "action": "read_file",
        "args": {"path": "stats.py", "max_lines": 40},
    }
    assert "Choose ONE action" in backend.calls[1][1][-1].content
    assert "finish.result as a STRING" in backend.calls[1][1][-1].content


def test_parent_menu_names_actual_children_and_requires_outcome_delivery():
    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Review evidence"),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Synthesized child evidence"),
            ],
            "agent-0001": [action("finish", result="Child evidence")],
        },
        delay=0.001,
    )
    result = run(AgentRuntime(backend, config=RuntimeConfig(max_agents=2)))
    assert result.status == "completed"
    parent_calls = [
        messages for request, messages in backend.calls if ".root." in request
    ]
    second = menu_from(parent_calls[1])
    assert second["direct_children"][0]["id"] == "agent-0001"
    assert "finish" not in second["allowed_action_schemas"]
    assert "spawn" not in second["allowed_action_schemas"]
    assert {"action": "wait", "agents": ["agent-0001"]} in second[
        "valid_syntax_examples_for_this_state"
    ]
    third = menu_from(parent_calls[2])
    assert third["direct_children"][0]["outcome_delivered"]
    assert "finish" in third["allowed_action_schemas"]


def test_repeat_wait_does_not_reset_repair_or_replay_outcomes():
    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Review"),
                action("wait", agents=["agent-0001"]),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Synthesis after rejecting idle wait"),
            ],
            "agent-0001": [action("finish", result="Evidence")],
        }
    )
    runtime = AgentRuntime(backend)
    result = run(runtime)
    assert result.status == "completed"
    events = runtime.store.events(result.run_id)
    assert len([event for event in events if event["kind"] == "children_settled"]) == 1
    assert any(
        "already delivered" in event["payload"].get("error", "")
        for event in events
        if event["kind"] == "action_rejected"
    )
    parent_calls = [
        messages for request, messages in backend.calls if ".root." in request
    ]
    assert "wait" not in menu_from(parent_calls[2])["allowed_action_schemas"]


def test_child_inherits_goal_constraints_but_default_cannot_edit():
    edited = []
    backend = ScriptBackend(
        {
            "root": [
                action(
                    "spawn", task="Inspect stats.py and propose correction; do not edit"
                ),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Review collected"),
            ],
            "agent-0001": [
                action("replace_text", args={"path": "stats.py"}),
                action("finish", result="Report proposed ValueError correction only"),
            ],
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={
            "replace_text": lambda args: edited.append(args) or "changed",
            "read_file": lambda args: "source",
        },
    )
    result = run(
        runtime,
        "Fix arithmetic mean. Empty input MUST raise ValueError. Children review only.",
    )
    assert result.status == "completed" and edited == []
    child = result.agents[1]
    assert "replace_text" not in child.tools and "read_file" in child.tools
    first_child = next(
        messages for request, messages in backend.calls if ".agent-0001.1" in request
    )
    assert "Empty input MUST raise ValueError" in first_child[1].content
    assert "YOUR ASSIGNED SCOPE" in first_child[1].content
    assert "replace_text" not in menu_from(first_child)["available_tools"]
    assert "replace_text" not in first_child[0].content


def test_explicit_child_grants_are_durable_and_cannot_escalate_nested_scope(tmp_path):
    path = tmp_path / "grants.sqlite"
    edited = []
    backend = ScriptBackend(
        {
            "root": [
                action(
                    "spawn", task="Implement fix", tools=["replace_text", "read_file"]
                ),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Implementation reviewed"),
            ],
            "agent-0001": [
                action("replace_text", args={"path": "stats.py"}),
                action("spawn", task="Illegal escalation", tools=["unregistered"]),
                action("spawn", task="Inspect independent evidence"),
                action("wait", agents=["agent-0002"]),
                action("finish", result="Implementation and review complete"),
            ],
            "agent-0002": [
                action("replace_text", args={}),
                action("finish", result="Read-only review complete"),
            ],
        }
    )
    store = AgentStore(path)
    runtime = AgentRuntime(
        backend,
        store,
        tools={
            "replace_text": lambda args: edited.append(args) or "edited",
            "read_file": lambda args: "evidence",
        },
    )
    result = run(runtime, run_id="grant-run")
    assert result.status == "completed" and len(edited) == 1
    assert set(result.agents[1].tools) == {"replace_text", "read_file", "journal_read"}
    assert set(result.agents[2].tools) == {"read_file", "journal_read"}
    store.close()
    reopened = AgentStore(path)
    assert reopened.agents("grant-run")[1].tools == result.agents[1].tools
    resumed = asyncio.run(
        AgentRuntime(backend, reopened, tools=runtime.tools).resume("grant-run")
    )
    assert resumed.agents[2].tools == result.agents[2].tools
    reopened.close()


def test_resume_enforces_persisted_child_grant_even_if_default_policy_changes(tmp_path):
    path = tmp_path / "active-grants.sqlite"
    config = RuntimeConfig()
    backend = ScriptBackend(
        {
            "root": ["unused", action("finish", result="Reviewed")],
            "agent-0001": ["unused", action("finish", result="No unauthorized edit")],
        }
    )
    edited = []
    tools = {
        "read_file": lambda args: "source",
        "write_file": lambda args: edited.append(args) or "edited",
    }
    store = AgentStore(path)
    runtime = AgentRuntime(backend, store, tools=tools)
    store.create_run(
        "active-grants",
        "Review only",
        vars(config),
        runtime._system_prompt(config),
        tools=["read_file", "write_file", "journal_read"],
    )
    store.spawn(
        "active-grants",
        "root",
        "spawn",
        "Review child",
        "Reviewer",
        [],
        "system",
        tools=["read_file", "journal_read"],
    )
    store.update_agent(
        "active-grants", "root", status="waiting", steps=1, wait_for=["agent-0001"]
    )
    store.update_agent(
        "active-grants",
        "agent-0001",
        steps=1,
        pending_action={"action": "tool", "name": "write_file", "args": {}},
        action_phase="prepared",
    )
    store.close()
    reopened = AgentStore(path)
    resumed = asyncio.run(
        AgentRuntime(backend, reopened, tools=tools, mutating_tools=set()).resume(
            "active-grants"
        )
    )
    assert resumed.status == "completed" and edited == []
    assert set(resumed.agents[1].tools) == {"read_file", "journal_read"}
    assert any(
        event["kind"] == "action_rejected"
        and "granted allowlist" in event["payload"]["error"]
        for event in reopened.events("active-grants")
    )
    reopened.close()


def test_tool_actor_isolated_across_threads_agents_and_cleanup():
    seen = []

    def inspect_actor(args):
        seen.append(tool_actor.get())
        return "Evidence"

    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Inspect independently"),
                action("inspect_actor", args={}),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Both contexts observed"),
            ],
            "agent-0001": [
                action("inspect_actor", args={}),
                action("finish", result="Child observed"),
            ],
        },
        delay=0.001,
    )
    result = run(
        AgentRuntime(
            backend,
            tools={"inspect_actor": inspect_actor},
            replay_safe_tools={"inspect_actor"},
        ),
        run_id="actors",
    )
    assert result.status == "completed"
    assert set(seen) == {("actors", "root"), ("actors", "agent-0001")}
    assert tool_actor.get() == ("direct", "direct")


def test_inherited_long_goals_are_visibly_bounded_without_durable_loss():
    runtime = AgentRuntime(ScriptBackend({}))
    original = "Important long goal. " * 200
    runtime.store.create_run("goal", original, vars(runtime.config), "system")
    context = runtime._inherited_goals("goal", "root")
    assert len(context) < 2400 and "TRUNCATED" in context
    assert runtime.store.agent("goal", "root").task == original


def test_parallel_children_messages_dependency_evidence_and_synthesis():
    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Read component A", role="Code reviewer"),
                action("spawn", task="Read component B", role="Code reviewer"),
                action(
                    "send", to="agent-0002", message="Check the boundary cases too."
                ),
                action(
                    "spawn",
                    task="Compare A and B",
                    dependencies=["agent-0001", "agent-0002"],
                ),
                action("wait", agents=["agent-0001", "agent-0002", "agent-0003"]),
                action("finish", result="A and B verified; comparison incorporated."),
            ],
            "agent-0001": [
                action("tool", name="read", args={"path": "A"}),
                action("finish", result="A result"),
            ],
            "agent-0002": [
                action("tool", name="read", args={"path": "B"}),
                action("tool", name="read", args={"path": "B-boundary"}),
                action("finish", result="B result"),
            ],
            "agent-0003": [action("finish", result="Comparison result")],
        },
        delay=0.01,
    )

    async def read(args):
        await asyncio.sleep(0.025)
        return "Evidence for " + args["path"]

    runtime = AgentRuntime(backend, tools={"read": read}, replay_safe_tools={"read"})
    result = run(runtime)
    assert result.status == "completed"
    assert len(result.agents) == 4
    assert backend.peak == 2
    child_messages = runtime.store.messages(result.run_id, "agent-0002")
    assert any("boundary cases" in message.content for message in child_messages)
    comparison_context = next(
        messages for request, messages in backend.calls if ".agent-0003." in request
    )
    assert (
        "A result" in comparison_context[-1].content
        and "B result" in comparison_context[-1].content
    )
    assert (
        "Read component A" not in comparison_context[1].content
    )  # Focused task; no full parent transcript.
    final_context = backend.calls[-1][1]
    assert any("Comparison result" in message.content for message in final_context)


def test_parent_must_read_child_outcomes_before_finish():
    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Child task"),
                action("finish", result="Premature"),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Synthesis"),
            ],
            "agent-0001": [action("finish", result="Child evidence")],
        },
        delay=0.005,
    )
    result = run(AgentRuntime(backend))
    assert result.status == "completed"
    assert result.output == "Synthesis"
    parent_calls = [
        messages for request, messages in backend.calls if ".root." in request
    ]
    assert "Action rejected" in parent_calls[2][-1].content
    assert "Child evidence" in parent_calls[3][-1].content


def test_failed_child_blocks_dependent_and_parent_receives_failure():
    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Failing task"),
                action(
                    "spawn", task="Depends on failed task", dependencies=["agent-0001"]
                ),
                action("wait", agents=["agent-0001", "agent-0002"]),
                action("finish", result="Could not verify; child failures reported."),
            ],
            "agent-0001": ["invalid"] * 3,
        },
        delay=0.005,
    )
    result = run(AgentRuntime(backend))
    assert result.status == "completed_with_failures"
    assert result.agents[2].error == "dependency failed: agent-0001"
    assert not any(".agent-0002." in request for request, _ in backend.calls)
    assert "failed" in backend.calls[-1][1][-1].content


def test_dependency_cannot_reference_parent_or_unknown_agent():
    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Bad dependency", dependencies=["root"]),
                action("finish", result="Handled invalid dependency"),
            ]
        }
    )
    result = run(AgentRuntime(backend))
    assert len(result.agents) == 1
    assert "existing direct children" in backend.calls[1][1][-1].content


def test_hierarchy_agents_and_steps_are_bounded():
    config = RuntimeConfig(max_agents=1, max_depth=0, max_steps_per_agent=2)
    backend = ScriptBackend(
        {"root": [action("spawn", task="Too many"), action("spawn", task="Again")]}
    )
    result = run(AgentRuntime(backend, config=config))
    assert result.status == "failed"
    assert len(result.agents) == 1
    assert "step limit" in result.error


def test_token_reservations_prevent_concurrent_overspend():
    store = AgentStore()
    store.create_run("budget", "task", {"deadline_seconds": 60}, "system")
    store.reserve("budget", "a", 80, 100)
    with pytest.raises(BudgetError):
        store.reserve("budget", "b", 30, 100)
    store.settle("budget", "a", 10, 10)
    store.reserve("budget", "b", 50, 100)
    store.uncertain("budget", "b")
    assert store.run("budget")["uncertain_tokens"] == 50
    with pytest.raises(BudgetError):
        store.reserve("budget", "c", 31, 100)
    store.close()
    backend = ScriptBackend({"root": [action("finish", result="Never invoked")]})
    result = run(AgentRuntime(backend, config=RuntimeConfig(max_total_tokens=100)))
    assert result.status == "failed"
    assert backend.calls == []
    assert "token budget exhausted" in result.error


def test_context_limit_is_explicit_and_history_remains_in_journal():
    backend = ScriptBackend({"root": [action("finish", result="Unused")]})
    runtime = AgentRuntime(backend, config=RuntimeConfig(max_context_tokens=100))
    result = run(runtime, "Long task " * 200)
    assert result.status == "failed"
    assert "context limit" in result.error
    assert (
        runtime.store.messages(result.run_id, "root")[1].content == "Long task " * 200
    )


def test_context_projection_preserves_task_bounds_and_recall():
    backend = ScriptBackend(
        {
            "root": [
                action("tool", name="read", args={"path": "first"}),
                action("tool", name="read", args={"path": "second"}),
                action(
                    "tool",
                    name="journal_read",
                    args={"after_event_id": 3, "limit": 1, "max_chars": 256},
                ),
                action(
                    "finish", result="Evidence reviewed with explicit bounded history"
                ),
            ]
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={"read": lambda args: args["path"] + " evidence " * 500},
        replay_safe_tools={"read"},
        config=RuntimeConfig(max_context_tokens=7800, max_tool_result_chars=6000),
    )
    result = run(runtime, "Original task must survive every projection")
    assert result.status == "completed"
    assert len(runtime.store.messages(result.run_id, "root")) == 9
    projected = runtime.store.events(result.run_id)
    assert any(event["kind"] == "context_projected" for event in projected)
    from forge_llm.agents.protocol import conservative_input_tokens

    for _, messages in backend.calls:
        assert messages[1].content == "Original task must survive every projection"
        assert conservative_input_tokens(messages) + 768 <= 7800
    assert any(
        "History projection" in message.content for message in backend.calls[-1][1]
    )
    results = [
        event["payload"] for event in projected if event["kind"] == "tool_result"
    ]
    recall = next(item for item in results if item["name"] == "journal_read")
    assert json.loads(recall["result"])["events"]


def test_journal_reader_paginates_payload_and_restricts_scope():
    runtime = AgentRuntime(ScriptBackend({}))
    config = RuntimeConfig()
    runtime.store.create_run("read", "Task", vars(config), "system")
    runtime.store.event("read", "root", "large", {"evidence": "0123456789" * 100})
    events = runtime.store.events("read")
    event_id = events[-1]["id"]
    first = json.loads(
        runtime._read_journal(
            "read",
            "root",
            {
                "after_event_id": event_id - 1,
                "limit": 1,
                "max_chars": 256,
                "include_control": True,
            },
        )
    )
    second = json.loads(
        runtime._read_journal(
            "read",
            "root",
            {
                "after_event_id": event_id - 1,
                "limit": 1,
                "max_chars": 256,
                "offset": 256,
                "include_control": True,
            },
        )
    )
    original = json.dumps(events[-1]["payload"])
    assert (
        first["events"][0]["payload_excerpt"] + second["events"][0]["payload_excerpt"]
        == original[:512]
    )
    assert first["events"][0]["next_offset"] == 256
    with pytest.raises(ValueError):
        runtime._read_journal("read", "root", {"limit": 1, "max_chars": 100})
    runtime.store.spawn("read", "root", "spawn", "Child", "Reviewer", [], "system")
    with pytest.raises(ValueError, match="own or a direct child's"):
        runtime._read_journal("read", "agent-0001", {"agent_id": "root"})


def test_tool_allowlist_and_truncation_preserve_full_evidence():
    backend = ScriptBackend(
        {
            "root": [
                action("tool", name="shell", args={"command": "bad"}),
                action("tool", name="read", args={}),
                action("finish", result="Partial evidence acknowledged"),
            ]
        }
    )
    runtime = AgentRuntime(
        backend,
        tools={"read": lambda args: "x" * 500},
        replay_safe_tools={"read"},
        config=RuntimeConfig(max_tool_result_chars=100),
    )
    result = run(runtime)
    assert result.status == "completed"
    assert "allowlist" in backend.calls[1][1][-1].content
    assert "TRUNCATED" in backend.calls[2][1][-1].content
    tool_result = next(
        event
        for event in runtime.store.events(result.run_id)
        if event["kind"] == "tool_result"
    )
    assert len(tool_result["payload"]["result"]) == 500


def test_unsafe_tool_preflight_failure_can_be_repaired_without_replay():
    class ToolValidationError(ValueError):
        work_started = False

    async def save(args):
        raise ToolValidationError("artifact path is required")

    backend = ScriptBackend(
        {
            "root": [
                action("tool", name="save", args={}),
                action("finish", result="Missing argument acknowledged"),
            ]
        }
    )
    result = run(AgentRuntime(backend, tools={"save": save}))
    assert result.status == "completed"
    assert "artifact path is required" in backend.calls[1][1][-1].content


def test_backend_failure_retry_keeps_request_id_and_charges_uncertainty():
    class Flaky(ScriptBackend):
        async def generate(self, messages, max_tokens, request_id):
            if not self.calls:
                self.calls.append((request_id, tuple(messages)))
                raise ConnectionError("worker disconnected")
            return await super().generate(messages, max_tokens, request_id)

    backend = Flaky({"root": [action("finish", result="Recovered")]})
    result = run(AgentRuntime(backend))
    assert result.status == "completed"
    assert backend.calls[0][0] == backend.calls[1][0]
    assert result.usage.uncertain_tokens > 0
    assert result.usage.reserved_tokens == 0


def test_final_retry_abandonment_cancels_potential_remote_work():
    backend = ScriptBackend({"root": [ConnectionError("lost remote poll")]})
    result = run(AgentRuntime(backend, config=RuntimeConfig(max_backend_retries=1)))
    assert result.status == "failed"
    assert len(backend.calls) == 2
    assert backend.cancelled == [backend.calls[0][0]]
    assert result.usage.reserved_tokens == 0 and result.usage.uncertain_tokens > 0


def test_timeout_and_run_deadline_stop_work():
    backend = ScriptBackend({"root": [action("finish", result="Late")]}, delay=1)
    result = run(
        AgentRuntime(
            backend,
            config=RuntimeConfig(
                generation_timeout_seconds=0.01, max_backend_retries=1
            ),
        )
    )
    assert result.status == "failed"
    assert "TimeoutError" in result.error
    assert backend.active == 0 and len(backend.calls) == 2
    backend = ScriptBackend({"root": [action("finish", result="Late")]}, delay=1)
    result = run(AgentRuntime(backend, config=RuntimeConfig(deadline_seconds=0.05)))
    assert result.status == "failed"
    assert "deadline" in result.error
    assert backend.active == 0


def test_cancel_propagates_to_worker_and_all_children():
    async def scenario():
        backend = ScriptBackend({"root": [action("finish", result="Late")]}, delay=1)
        runtime = AgentRuntime(backend)
        job = asyncio.create_task(runtime.run("Task", run_id="cancel"))
        while not backend.calls:
            await asyncio.sleep(0.001)
        await runtime.cancel("cancel")
        result = await job
        assert result.status == "cancelled"
        assert all(agent.status == "cancelled" for agent in result.agents)
        assert backend.cancelled and backend.active == 0
        assert result.usage.reserved_tokens == 0

    asyncio.run(scenario())


def test_worker_coroutine_cancel_returns_cancelled_run_not_coordinator_interruption():
    class CancellingBackend(ScriptBackend):
        def __init__(self):
            super().__init__({"root": [action("finish", result="Late")]}, delay=10)
            self.task = None

        async def generate(self, messages, max_tokens, request_id):
            self.task = asyncio.current_task()
            return await super().generate(messages, max_tokens, request_id)

        async def cancel(self, request_id):
            await super().cancel(request_id)
            self.task.cancel()

    async def scenario():
        backend = CancellingBackend()
        runtime = AgentRuntime(backend)
        job = asyncio.create_task(runtime.run("Task", run_id="worker-cancel"))
        while not backend.calls:
            await asyncio.sleep(0.001)
        await runtime.cancel("worker-cancel")
        result = await job
        assert result.status == "cancelled"
        assert result.usage.reserved_tokens == 0
        assert all(agent.status == "cancelled" for agent in result.agents)

    asyncio.run(scenario())


def test_failed_parent_cancels_descendants_before_root_synthesis():
    class SlowGrandchild(ScriptBackend):
        async def generate(self, messages, max_tokens, request_id):
            if ".agent-0002." in request_id:
                await asyncio.sleep(10)
            return await super().generate(messages, max_tokens, request_id)

    backend = SlowGrandchild(
        {
            "root": [
                action("spawn", task="Child"),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Failure reported"),
            ],
            "agent-0001": [
                action("spawn", task="Grandchild"),
                "invalid",
                "invalid",
                "invalid",
            ],
            "agent-0002": [action("finish", result="Unreachable")],
        },
        delay=0.001,
    )
    result = run(AgentRuntime(backend))
    assert result.status == "completed_with_failures"
    assert result.agents[2].status == "cancelled"
    assert all(
        agent.status in {"completed", "failed", "cancelled"} for agent in result.agents
    )


def test_message_arriving_during_finish_requires_next_turn_review():
    class SlowChild(ScriptBackend):
        async def generate(self, messages, max_tokens, request_id):
            if ".agent-0001.1" in request_id:
                await asyncio.sleep(0.04)
            return await super().generate(messages, max_tokens, request_id)

    backend = SlowChild(
        {
            "root": [
                action("spawn", task="Child"),
                action("send", to="agent-0001", message="New requirement"),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Synthesized updated result"),
            ],
            "agent-0001": [
                action("finish", result="Stale answer"),
                action("finish", result="Updated answer"),
            ],
        },
        delay=0.001,
    )
    result = run(AgentRuntime(backend))
    assert result.status == "completed"
    assert result.agents[1].result == "Updated answer"
    child_second = next(
        messages for request, messages in backend.calls if ".agent-0001.2" in request
    )
    assert "new messages arrived" in child_second[-1].content
    assert any("New requirement" in message.content for message in child_second)


def test_interruption_resume_and_stale_reservation_accounting(tmp_path):
    async def scenario():
        store = AgentStore(tmp_path / "resume.sqlite")
        backend = ScriptBackend(
            {"root": [action("finish", result="After resume")]}, delay=10
        )
        runtime = AgentRuntime(backend, store)
        job = asyncio.create_task(runtime.run("Task", run_id="resume"))
        while not backend.calls:
            await asyncio.sleep(0.001)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):
            await job
        assert store.run("resume")["status"] == "interrupted"
        assert store.run("resume")["uncertain_tokens"] > 0
        backend.delay = 0
        result = await AgentRuntime(backend, store).resume("resume")
        assert result.status == "completed"
        assert backend.calls[0][0] == backend.calls[1][0]
        assert result.agents[0].steps == 1
        store.close()

    asyncio.run(scenario())


def test_resume_reuses_exact_inflight_prompt_then_reviews_new_messages():
    store = AgentStore()
    config = RuntimeConfig()
    backend = ScriptBackend(
        {
            "root": [
                action("finish", result="Original answer"),
                action("finish", result="New requirement incorporated"),
            ]
        }
    )
    runtime = AgentRuntime(backend, store, config=config)
    store.create_run(
        "snapshot", "Original task", vars(config), runtime._system_prompt(config)
    )
    messages = store.messages("snapshot", "root")
    store.save_generation_input("snapshot", "snapshot.root.1", messages, len(messages))
    store.update_agent(
        "snapshot",
        "root",
        status="running",
        steps=1,
        action_phase="generating",
        backend_attempts=1,
    )
    store.add_message(
        "snapshot", "root", "user", "New requirement arrived before recovery"
    )
    result = asyncio.run(runtime.resume("snapshot"))
    assert result.status == "completed"
    assert result.output == "New requirement incorporated"
    assert backend.calls[0][0] == "snapshot.root.1"
    assert backend.calls[0][1] == tuple(messages)
    assert any(
        "New requirement arrived" in message.content for message in backend.calls[1][1]
    )
    assert "new messages arrived" in backend.calls[1][1][-1].content
    store.close()


def test_nonretryable_preflight_failure_releases_unused_reservation():
    class PreflightError(RuntimeError):
        retryable = False
        work_started = False

    backend = ScriptBackend({"root": [PreflightError("model context mismatch")]})
    result = run(AgentRuntime(backend))
    assert result.status == "failed"
    assert len(backend.calls) == 1
    assert result.usage.total_tokens == 0
    assert "model context mismatch" in result.error


@pytest.mark.parametrize("run_id", ["x/y", "", "x" * 97, "é", "a.b", False, 0])
def test_run_ids_are_safe_for_worker_request_ids(run_id):
    backend = ScriptBackend({"root": [action("finish", result="Unused")]})
    with pytest.raises(ValueError, match="run_id"):
        run(AgentRuntime(backend), run_id=run_id)


@pytest.mark.parametrize("safe", [False, True])
def test_crashed_tool_replay_is_explicit_and_safe_only(safe):
    store = AgentStore()
    config = RuntimeConfig()
    backend = ScriptBackend(
        {"root": ["unused", action("finish", result="Read resumed")]}
    )
    runtime = AgentRuntime(
        backend,
        store,
        tools={"read": lambda args: "Evidence"},
        config=config,
        replay_safe_tools={"read"} if safe else set(),
    )
    store.create_run("crash", "Task", vars(config), runtime._system_prompt(config))
    store.update_agent(
        "crash",
        "root",
        status="executing",
        steps=1,
        pending_action={"action": "tool", "name": "read", "args": {}},
        action_phase="executing",
    )
    result = asyncio.run(runtime.resume("crash"))
    assert result.status == ("completed" if safe else "failed")
    if safe:
        assert "Evidence" in backend.calls[0][1][-1].content
    else:
        assert "automatic replay refused" in result.error and not backend.calls
    store.close()


def test_lease_blocks_two_coordinators(tmp_path):
    first = AgentStore(tmp_path / "lease.sqlite")
    second = AgentStore(tmp_path / "lease.sqlite")
    first.create_run("lease", "Task", {"deadline_seconds": 60}, "system")
    first.acquire_lease("lease", "owner-one", 30)
    with pytest.raises(LeaseError):
        second.acquire_lease("lease", "owner-two", 30)
    first.release_lease("lease", "owner-one")
    second.acquire_lease("lease", "owner-two", 30)
    with pytest.raises(LeaseError):
        first.renew_lease("lease", "owner-one", 30)
    first.close()
    second.close()


def test_lost_lease_prevents_stale_model_response_from_committing():
    async def scenario():
        backend = ScriptBackend(
            {"root": [action("finish", result="Only new owner may commit")]}, delay=0.03
        )
        store = AgentStore()
        runtime = AgentRuntime(backend, store)
        job = asyncio.create_task(runtime.run("Task", run_id="takeover"))
        while not backend.calls:
            await asyncio.sleep(0.001)
        store.connection.execute(
            "UPDATE runs SET lease_owner='new-owner',lease_until=? WHERE run_id='takeover'",
            (time.time() + 30,),
        )
        with pytest.raises(LeaseError):
            await job
        assert store.agent("takeover", "root").result is None
        assert store.run("takeover")["lease_owner"] == "new-owner"
        assert store.run("takeover")["reserved_tokens"] > 0
        store.release_lease("takeover", "new-owner")
        backend.delay = 0
        result = await AgentRuntime(backend, store).resume("takeover")
        assert result.status == "completed"
        assert result.usage.uncertain_tokens > 0 and result.usage.reserved_tokens == 0
        store.close()

    asyncio.run(scenario())


def test_spawn_effect_is_idempotent():
    store = AgentStore()
    store.create_run("spawn", "Task", {"deadline_seconds": 60}, "system")
    first = store.spawn(
        "spawn", "root", "spawn/root/1", "Child", "Reviewer", [], "system"
    )
    second = store.spawn(
        "spawn", "root", "spawn/root/1", "Child", "Reviewer", [], "system"
    )
    assert first == second and len(store.agents("spawn")) == 2
    assert (
        len([event for event in store.events("spawn") if event["kind"] == "spawn"]) == 1
    )
    store.close()


def test_concurrency_bound_applies_across_runs():
    async def scenario():
        backend = ScriptBackend({"root": [action("finish", result="Done")]}, delay=0.02)
        runtime = AgentRuntime(
            backend, config=RuntimeConfig(max_concurrent_generations=1)
        )
        results = await asyncio.gather(
            *(runtime.run("Task", run_id=f"parallel-{index}") for index in range(3))
        )
        assert all(result.status == "completed" for result in results)
        assert backend.peak == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_agents": True},
        {"max_agents": 0},
        {"max_depth": -1},
        {"deadline_seconds": float("nan")},
        {"lease_seconds": 1},
        {"max_tool_result_chars": 100, "max_stored_tool_chars": 50},
    ],
)
def test_config_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        RuntimeConfig(**kwargs)
