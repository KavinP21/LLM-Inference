"""Operator-selected completion checks validate work without solving it."""

import asyncio
import json
import sys

import pytest
from forge_llm.agents.runtime import AgentRuntime, CompletionCheck, RuntimeConfig
from forge_llm.agents.store import AgentStore
from forge_llm.agents.tools import WorkspaceTools
from test_agent_runtime import ScriptBackend, action


def test_real_pytest_rejection_then_actual_edit_and_verified_completion(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "stats.py").write_text(
        "def mean(values):\n    return sum(values) / (len(values) + 1)\n"
    )
    (workspace / "test_stats.py").write_text(
        "import pytest\nfrom stats import mean\ndef test_mean():\n    assert mean([2, 4]) == 3\n    assert mean([7]) == 7\ndef test_empty():\n    with pytest.raises(ValueError): mean([])\n"
    )
    tools = WorkspaceTools(
        workspace,
        tmp_path / "artifacts",
        allow_write=True,
        allow_tests=True,
        python=sys.executable,
    )
    checked = []

    async def validator(run_id, agent_id, result):
        report = json.loads(await tools.run_tests({"paths": ["test_stats.py"]}))
        checked.append(report)
        return CompletionCheck(report["exit_code"] == 0, report["output"])

    correct = "def mean(values):\n    if not values:\n        raise ValueError('empty')\n    return sum(values) / len(values)\n"
    backend = ScriptBackend(
        {
            "root": [
                action("finish", result="The tests failed; ending early."),
                action("read_file", args={"path": "stats.py"}),
                action("write_file", args={"path": "stats.py", "content": correct}),
                action("finish", result="Correction made and actual tests verified."),
            ]
        }
    )
    runtime = AgentRuntime(
        backend,
        tools=tools.mapping(),
        tool_descriptions=tools.tool_descriptions(),
        replay_safe_tools=tools.replay_safe_tools,
        completion_validator=validator,
    )
    result = asyncio.run(runtime.run("Fix the mean implementation and validate it."))
    assert result.status == "completed" and result.completion_verified
    assert (
        result.acceptance_passed is True and result.agents[0].completion_rejections == 1
    )
    assert [report["exit_code"] for report in checked] == [1, 0]
    assert "assert mean([2, 4]) == 3" in backend.calls[1][1][-1].content
    assert (workspace / "stats.py").read_text() == correct


def test_failed_checks_are_bounded_and_never_mark_verified():
    called = []

    async def check(run_id, agent_id, result):
        called.append((agent_id, result))
        return CompletionCheck(False, "Requested checks still fail.")

    backend = ScriptBackend({"root": [action("finish", result="Premature")] * 5})
    result = asyncio.run(
        AgentRuntime(
            backend,
            completion_validator=check,
            config=RuntimeConfig(max_completion_rejections=2),
        ).run("Task")
    )
    assert result.status == "failed" and len(called) == 2
    assert result.acceptance_passed is False and not result.completion_verified
    assert result.output is None and "2 rejected finishes" in result.error


def test_default_validator_scope_checks_root_only():
    called = []

    async def check(run_id, agent_id, result):
        called.append(agent_id)
        return CompletionCheck(True, "Parent criteria passed.")

    backend = ScriptBackend(
        {
            "root": [
                action("spawn", task="Review only"),
                action("wait", agents=["agent-0001"]),
                action("finish", result="Parent synthesis"),
            ],
            "agent-0001": [action("finish", result="Local review")],
        }
    )
    result = asyncio.run(AgentRuntime(backend, completion_validator=check).run("Task"))
    assert result.status == "completed" and called == ["root"]
    assert result.agents[1].completion_passed is None


def test_persisted_validator_outcome_is_replayed_without_reexecuting_callback(tmp_path):
    store = AgentStore(tmp_path / "checks.sqlite")
    config = RuntimeConfig()
    store.create_run(
        "checked", "Task", vars(config), "system", completion_required=True
    )
    store.update_agent(
        "checked",
        "root",
        status="executing",
        steps=1,
        pending_action={"action": "finish", "result": "Candidate"},
        action_phase="validating",
    )
    store.record_effect(
        "checked/root/1/effect/completion_check",
        {"passed": True, "feedback": "Already checked successfully."},
    )
    store.close()
    reopened = AgentStore(tmp_path / "checks.sqlite")

    async def should_not_run(*args):
        raise AssertionError("a recorded callback result must not be recomputed")

    runtime = AgentRuntime(
        ScriptBackend({}), reopened, completion_validator=should_not_run
    )
    result = asyncio.run(runtime.resume("checked"))
    assert result.status == "completed" and result.completion_verified
    assert result.output == "Candidate" and not runtime.backend.calls
    reopened.close()


def test_interrupted_unknown_validation_refuses_duplicate_callback():
    store = AgentStore()
    config = RuntimeConfig()
    store.create_run(
        "unknown", "Task", vars(config), "system", completion_required=True
    )
    store.update_agent(
        "unknown",
        "root",
        status="executing",
        steps=1,
        pending_action={"action": "finish", "result": "Candidate"},
        action_phase="validating",
    )
    calls = []

    async def check(*args):
        calls.append(args)
        return CompletionCheck(True)

    result = asyncio.run(
        AgentRuntime(ScriptBackend({}), store, completion_validator=check).resume(
            "unknown"
        )
    )
    assert result.status == "failed" and not calls
    assert "callback replay refused" in result.error
    assert not result.completion_verified


def test_recorded_rejection_is_applied_once_then_new_finish_gets_new_check():
    store = AgentStore()
    config = RuntimeConfig()
    store.create_run(
        "rejected", "Task", vars(config), "system", completion_required=True
    )
    store.update_agent(
        "rejected",
        "root",
        status="executing",
        steps=1,
        pending_action={"action": "finish", "result": "Early"},
        action_phase="validating",
    )
    store.record_effect(
        "rejected/root/1/effect/completion_check",
        {"passed": False, "feedback": "Actual tests still fail."},
    )
    calls = []

    async def check(run_id, agent_id, result):
        calls.append(result)
        return CompletionCheck(True, "New checks passed.")

    backend = ScriptBackend(
        {"root": ["unused", action("finish", result="Revised candidate")]}
    )
    result = asyncio.run(
        AgentRuntime(backend, store, completion_validator=check).resume("rejected")
    )
    assert result.status == "completed" and result.completion_verified
    assert (
        calls == ["Revised candidate"] and result.agents[0].completion_rejections == 1
    )
    assert (
        len(
            [
                event
                for event in store.events("rejected")
                if event["kind"] == "completion_rejected"
            ]
        )
        == 1
    )


def test_resume_cannot_drop_required_checker_or_change_scope():
    store = AgentStore()
    store.create_run(
        "required", "Task", vars(RuntimeConfig()), "system", completion_required=True
    )
    with pytest.raises(ValueError, match="original configured completion validator"):
        asyncio.run(AgentRuntime(ScriptBackend({}), store).resume("required"))

    async def check(*args):
        return CompletionCheck(True)

    with pytest.raises(ValueError, match="preserve.*scope"):
        asyncio.run(
            AgentRuntime(
                ScriptBackend({}),
                store,
                completion_validator=check,
                completion_root_only=False,
            ).resume("required")
        )


def test_completion_timeout_is_diagnostic_and_bounded():
    async def check(*args):
        await asyncio.sleep(10)
        return CompletionCheck(True)

    backend = ScriptBackend({"root": [action("finish", result="Premature")] * 2})
    result = asyncio.run(
        AgentRuntime(
            backend,
            completion_validator=check,
            config=RuntimeConfig(
                max_completion_rejections=1, completion_timeout_seconds=0.01
            ),
        ).run("Task")
    )
    assert result.status == "failed" and "TimeoutError" in result.error
    assert not result.completion_verified and result.usage.reserved_tokens == 0


def test_unchecked_completion_is_explicitly_unverified():
    result = asyncio.run(
        AgentRuntime(
            ScriptBackend({"root": [action("finish", result="Model session ended")]})
        ).run("Task")
    )
    assert result.status == "completed" and not result.completion_verified
    assert result.acceptance_passed is None
