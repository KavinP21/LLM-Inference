"""Cross-layer control tests; scripted decisions are not model-quality evidence.

These tests exercise the real durable coordinator, confined workspace tools,
replica pool, HTTP client/server, and worker job lifecycle together. Scripted
model responses deliberately isolate transport and orchestration correctness.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar

import pytest
from forge_llm.agents import cli
from forge_llm.agents.backends import (
    LocalProcessBackend,
    RemoteWorkerBackend,
    RequestValidationError,
    WorkerConfig,
    WorkerPool,
    WorkerUnavailableError,
)
from forge_llm.agents.profiles import CompletionProfile
from forge_llm.agents.protocol import Generation
from forge_llm.agents.runtime import AgentRuntime, CompletionCheck, RuntimeConfig
from forge_llm.agents.store import AgentStore
from forge_llm.agents.tools import WorkspaceTools
from forge_llm.agents.worker import WorkerHTTPServer, WorkerService

BEARER = "control-test-credential-never-used-outside-loopback"


def action(kind, **fields):
    return json.dumps({"action": kind, **fields})


def native_call(name, **arguments):
    return (
        "<tool_call>"
        + json.dumps({"name": name, "arguments": arguments})
        + "</tool_call>"
    )


async def eventually(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("cross-layer control condition timed out")
        await asyncio.sleep(0.002)


class ScriptState:
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = []
        self.cancelled = []
        self.active = 0
        self.peak = 0


class ScriptedReplica:
    def __init__(self, name, scripts, state, *, delays=None):
        self.name, self.scripts, self.state = name, scripts, state
        self.delays = delays or {}
        self.calls = []
        self.active = 0
        self.peak = 0

    def health(self):
        return {
            "worker_id": self.name,
            "backend": "scripted-control",
            "model": "scripted-test-fixture",
            "model_data_sha256": "fixture-weights-v1",
            "model_config_sha256": "fixture-config-v1",
            "tokenizer_signature": "fixture-tokenizer-v1",
            "draft_model_data_sha256": None,
        }

    async def generate(self, messages, max_tokens, request_id):
        _, agent, turn = request_id.rsplit(".", 2)
        with self.state.lock:
            self.calls.append(request_id)
            self.state.calls.append((self.name, request_id, tuple(messages)))
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.state.active += 1
            self.state.peak = max(self.state.peak, self.state.active)
        try:
            await asyncio.sleep(self.delays.get(agent, 0.008))
            response = self.scripts[agent][int(turn) - 1]
            if callable(response):
                response = response(messages)
            # Counts belong to this fixture; no tokenizer measurement is claimed.
            return Generation(
                response, 20, min(10, max_tokens), "stop", "scripted-control"
            )
        except asyncio.CancelledError:
            with self.state.lock:
                self.state.cancelled.append(request_id)
            raise
        finally:
            with self.state.lock:
                self.active -= 1
                self.state.active -= 1


class NativeScriptedReplica(ScriptedReplica):
    supports_native_tools = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalogs = {}

    async def generate_action(self, messages, max_tokens, request_id, tool_specs):
        self.catalogs[request_id] = tool_specs
        return await self.generate(messages, max_tokens, request_id)


@contextmanager
def serve(replica):
    service = WorkerService(replica, worker_id=replica.name)
    server = None
    thread = None
    try:
        service.start()
        server = WorkerHTTPServer(("127.0.0.1", 0), service, BEARER)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        yield f"http://127.0.0.1:{server.server_port}", service
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=2.0)
        service.close()


def tool_result(messages, name):
    prefix = f"Tool {name} result:\n"
    payload = next(
        m.content[len(prefix) :]
        for m in reversed(messages)
        if m.content.startswith(prefix)
    )
    # The runtime appends an untrusted-evidence framing reminder after the JSON.
    return json.JSONDecoder().raw_decode(payload)[0]


def test_actual_http_workers_tools_spawn_message_wait_and_synthesis(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    originals = {"A.txt": "owner=A\nbudget=12\n", "B.txt": "owner=B\nbudget=7\n"}
    for name, contents in originals.items():
        (workspace / name).write_text(contents)
    state = ScriptState()
    summaries = {}

    def child_result(name):
        def respond(messages):
            result = tool_result(messages, "read_file")
            expected_hash = hashlib.sha256(originals[name].encode()).hexdigest()
            assert result["sha256"] == expected_hash
            assert result["path"] == name
            assert f"owner={name[0]}" in result["content"]
            if name == "B.txt":
                assert any("Compare owner labels too." in m.content for m in messages)
            summaries[name] = result
            return action(
                "finish",
                result=f"{name}: {result['content']}; sha256={result['sha256']}",
            )

        return respond

    def save_synthesis(messages):
        outcomes = next(
            m.content
            for m in reversed(messages)
            if m.content.startswith("Child outcomes")
        )
        assert "budget=12" in outcomes and "budget=7" in outcomes
        assert all(result["sha256"] in outcomes for result in summaries.values())
        return action(
            "tool",
            name="save_artifact",
            args={"path": "report.md", "content": outcomes + "\nCombined budget=19\n"},
        )

    scripts = {
        "root": [
            action("spawn", task="Read A.txt and report its owner and budget."),
            action("spawn", task="Read B.txt and report its owner and budget."),
            action("send", to="agent-0002", message="Compare owner labels too."),
            action("wait", agents=["agent-0001", "agent-0002"]),
            save_synthesis,
            action(
                "finish",
                result="Verified A=12 and B=7 from both child reports; combined budget=19. Saved report.md.",
            ),
        ],
        "agent-0001": [
            action("tool", name="read_file", args={"path": "A.txt"}),
            child_result("A.txt"),
        ],
        "agent-0002": [
            action("tool", name="read_file", args={"path": "B.txt"}),
            child_result("B.txt"),
        ],
    }
    first, second = (
        ScriptedReplica("control-a", scripts, state),
        ScriptedReplica("control-b", scripts, state),
    )
    with serve(first) as (url_a, service_a), serve(second) as (url_b, service_b):

        async def run():
            pool = WorkerPool(
                {
                    "a": RemoteWorkerBackend(url_a, BEARER, poll_interval=0.002),
                    "b": RemoteWorkerBackend(url_b, BEARER, poll_interval=0.002),
                }
            )
            tools = WorkspaceTools(workspace, tmp_path / "artifacts")
            mappings = tools.mapping()
            store = AgentStore(tmp_path / "journal.sqlite")
            runtime = None

            async def coordinated_read(args):
                if args["path"] == "B.txt":
                    # Hold B's first real read until durable messaging has been
                    # delivered, eliminating scheduler-timing assumptions.
                    await eventually(
                        lambda: any(
                            "Compare owner labels too." in m.content
                            for m in store.messages("cross-layer", "agent-0002")
                        )
                    )
                return await tools.read_file(args)

            mappings["read_file"] = coordinated_read
            runtime = AgentRuntime(
                pool,
                store,
                mappings,
                RuntimeConfig(max_output_tokens=512, deadline_seconds=10),
                tool_descriptions=tools.descriptions,
                replay_safe_tools=tools.replay_safe_tools,
            )
            try:
                result = await runtime.run(
                    "Read both files, delegate independently, then save a sourced synthesis.",
                    run_id="cross-layer",
                )
                assert result.status == "completed"
                assert result.output.endswith("Saved report.md.")
                assert len(result.agents) == 3
                assert all(agent.status == "completed" for agent in result.agents)
                events = store.events("cross-layer")
                assert [e["kind"] for e in events].count("spawn") == 2
                assert [e["kind"] for e in events].count("send") == 1
                assert [e["kind"] for e in events].count("children_settled") == 1
                generations = [
                    e["payload"]["request_id"]
                    for e in events
                    if e["kind"] == "generation"
                ]
                assert len(generations) == len(set(generations)) == 10
                assert all(
                    "/" not in request and request.startswith("cross-layer.")
                    for request in generations
                )
                assert (
                    result.usage.input_tokens == 200
                    and result.usage.output_tokens == 100
                )
                assert (
                    result.usage.reserved_tokens == result.usage.uncertain_tokens == 0
                )
                assert state.peak == 2 and first.peak == second.peak == 1
                assert first.calls and second.calls
                assert service_a.health()["active"] == service_b.health()["active"] == 0
                report = (tmp_path / "artifacts" / "report.md").read_text()
                assert "Combined budget=19" in report
                assert all(result["sha256"] in report for result in summaries.values())
                assert {
                    name: (workspace / name).read_text() for name in originals
                } == originals
                call_count = len(state.calls)
                assert (await runtime.resume("cross-layer")).output == result.output
                assert (
                    len(state.calls) == call_count
                )  # Terminal resume never reexecutes inference/tools.
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_native_runtime_catalogs_and_tool_roles_cross_two_http_workers(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "A.txt").write_text("A: 12 units\n")
    (workspace / "B.txt").write_text("B: 7 units\n")
    state = ScriptState()

    def child_finish(messages):
        evidence = tool_result(messages, "read_file")
        assert any(
            m.role == "tool" and m.content.startswith("Tool read_file result:")
            for m in messages
        )
        return native_call(
            "finish", result=evidence["content"] + "; source=" + evidence["sha256"]
        )

    def synthesis(messages):
        evidence = next(
            m.content
            for m in reversed(messages)
            if m.content.startswith("Child outcomes")
        )
        assert "12 units" in evidence and "7 units" in evidence
        return native_call(
            "finish",
            result="Both native tool-backed child reads verified; total is 19 units.",
        )

    scripts = {
        "root": [
            native_call(
                "spawn",
                task="Read A.txt and report the actual evidence.",
                tools=["read_file"],
            ),
            native_call(
                "spawn",
                task="Read B.txt and report the actual evidence.",
                tools=["read_file"],
            ),
            native_call("wait", agents=["agent-0001", "agent-0002"]),
            synthesis,
        ],
        "agent-0001": [native_call("read_file", path="A.txt"), child_finish],
        "agent-0002": [native_call("read_file", path="B.txt"), child_finish],
    }
    first = NativeScriptedReplica("native-control-a", scripts, state)
    second = NativeScriptedReplica("native-control-b", scripts, state)
    with serve(first) as (url_a, _), serve(second) as (url_b, _):

        async def run():
            a = RemoteWorkerBackend(url_a, BEARER, poll_interval=0.002)
            b = RemoteWorkerBackend(url_b, BEARER, poll_interval=0.002)
            await asyncio.gather(a.health(), b.health())
            pool = WorkerPool({"a": a, "b": b})
            tools = WorkspaceTools(workspace, tmp_path / "artifacts")
            store = AgentStore(tmp_path / "native.sqlite")
            runtime = AgentRuntime(
                pool,
                store,
                tools.mapping(),
                RuntimeConfig(max_output_tokens=512),
                tool_descriptions=tools.tool_descriptions(),
                replay_safe_tools=tools.replay_safe_tools,
            )
            try:
                assert runtime.native_mode and pool.supports_native_tools
                result = await runtime.run(
                    "Read both files using focused delegated evidence.",
                    run_id="native-wire",
                )
                assert result.status == "completed"
                assert result.output.endswith("19 units.")
                assert len(result.agents) == 3
                assert first.calls and second.calls
                assert state.peak == 2
                catalogs = {**first.catalogs, **second.catalogs}
                assert len(catalogs) == 8
                for request, catalog in catalogs.items():
                    if ".agent-" in request:
                        names = {entry["function"]["name"] for entry in catalog}
                        assert "read_file" in names
                        assert (
                            not {
                                "save_artifact",
                                "write_file",
                                "replace_text",
                                "edit_lines",
                                "run_tests",
                            }
                            & names
                        )
                assert not any(
                    "Every reply MUST be exactly one JSON object" in m.content
                    for _, _, messages in state.calls
                    for m in messages
                    if m.role == "system"
                )
                assert (
                    result.usage.uncertain_tokens == result.usage.reserved_tokens == 0
                )
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_native_batch_spawn_and_completion_checker_correct_real_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "A.txt").write_text("12\n")
    (workspace / "B.txt").write_text("7\n")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_total.py").write_text(
        "from pathlib import Path\n"
        "def test_combined_total():\n"
        "    expected = int(Path('A.txt').read_text()) + int(Path('B.txt').read_text())\n"
        "    assert int(Path('total.txt').read_text()) == expected\n"
    )
    state = ScriptState()
    checks = []

    def correction_after_real_failure(messages):
        assert any(
            "Finish rejected by configured acceptance checks" in m.content
            for m in messages
        )
        assert any("FileNotFoundError" in m.content for m in messages)
        return native_call("write_file", path="total.txt", content="19\n")

    scripts = {
        "root": [
            native_call(
                "spawn",
                task="Read A.txt and report its exact value.",
                tools=["read_file"],
            )
            + "\n"
            + native_call(
                "spawn",
                task="Read B.txt and report its exact value.",
                tools=["read_file"],
            ),
            native_call("wait", agents=["agent-0001", "agent-0002"]),
            native_call(
                "finish", result="Both child reads are done; the total is ready."
            ),
            correction_after_real_failure,
            native_call(
                "finish",
                result="Saved total.txt with 19; the configured independent test passes.",
            ),
        ],
        "agent-0001": [
            native_call("read_file", path="A.txt"),
            native_call("finish", result="A.txt contains 12."),
        ],
        "agent-0002": [
            native_call("read_file", path="B.txt"),
            native_call("finish", result="B.txt contains 7."),
        ],
    }
    first = NativeScriptedReplica("batch-control-a", scripts, state)
    second = NativeScriptedReplica("batch-control-b", scripts, state)
    with serve(first) as (url_a, _), serve(second) as (url_b, _):

        async def run():
            a = RemoteWorkerBackend(url_a, BEARER, poll_interval=0.002)
            b = RemoteWorkerBackend(url_b, BEARER, poll_interval=0.002)
            await asyncio.gather(a.health(), b.health())
            pool = WorkerPool({"a": a, "b": b})
            tools = WorkspaceTools(
                workspace, tmp_path / "artifacts", allow_write=True, allow_tests=True
            )
            protected = tools.protect_files(["tests/test_total.py"])
            store = AgentStore(tmp_path / "batch-completion.sqlite")

            async def completion(run_id, agent_id, answer):
                assert tools.protected_file_hashes(["tests/test_total.py"]) == protected
                actual = json.loads(
                    await tools.run_tests({"paths": ["tests/test_total.py"]})
                )
                assert tools.protected_file_hashes(["tests/test_total.py"]) == protected
                checks.append((run_id, agent_id, actual))
                return CompletionCheck(actual["exit_code"] == 0, actual["output"])

            runtime = AgentRuntime(
                pool,
                store,
                tools.mapping(),
                RuntimeConfig(max_output_tokens=512),
                tool_descriptions=tools.tool_descriptions(),
                replay_safe_tools=tools.replay_safe_tools,
                completion_validator=completion,
            )
            try:
                result = await runtime.run(
                    "Use two independent child reads and write total.txt; finish only after acceptance passes.",
                    run_id="batch-checked",
                )
                assert result.status == "completed"
                assert (
                    result.acceptance_passed is True
                    and result.completion_verified is True
                )
                assert len(result.agents) == 3
                assert result.agents[0].completion_rejections == 1
                assert result.agents[0].completion_passed is True
                assert [check[2]["exit_code"] for check in checks] == [1, 0]
                assert all(agent_id == "root" for _, agent_id, _ in checks)
                assert (workspace / "total.txt").read_text() == "19\n"
                assert (workspace / "A.txt").read_text() == "12\n" and (
                    workspace / "B.txt"
                ).read_text() == "7\n"
                events = store.events("batch-checked")
                assert len([event for event in events if event["kind"] == "spawn"]) == 2
                assert (
                    len(
                        [
                            event
                            for event in events
                            if event["kind"] == "completion_rejected"
                        ]
                    )
                    == 1
                )
                ids = [request for _, request, _ in state.calls]
                assert len(ids) == len(set(ids)) == 9
                assert (
                    ids.count("batch-checked.root.1") == 1
                )  # One model frame created both children.
                assert first.calls and second.calls
                previous = len(state.calls)
                assert (await runtime.resume("batch-checked")).completion_verified
                assert len(state.calls) == previous and len(checks) == 2
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_native_http_whole_line_edit_passes_protected_independent_tests(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "value.py"
    path.write_text("# keep header\ndef answer():\n    return 41\n# keep footer\n")
    (workspace / "tests").mkdir()
    test_path = workspace / "tests" / "test_value.py"
    original_test = (
        "from value import answer\ndef test_answer():\n    assert answer() == 42\n"
    )
    test_path.write_text(original_test)
    state = ScriptState()
    replica = NativeScriptedReplica(
        "line-edit-control",
        {
            "root": [
                native_call("read_file", path="value.py"),
                native_call(
                    "edit_lines",
                    path="value.py",
                    start_line=2,
                    end_line=3,
                    content="def answer():\n    value = 42\n    return value",
                ),
                native_call(
                    "finish",
                    result="Replaced the whole function and preserved neighboring lines; independent test passes.",
                ),
            ],
        },
        state,
    )
    with serve(replica) as (url, _):

        async def run():
            remote = RemoteWorkerBackend(url, BEARER, poll_interval=0.002)
            await remote.health()
            pool = WorkerPool([remote])
            tools = WorkspaceTools(
                workspace, tmp_path / "artifacts", allow_write=True, allow_tests=True
            )
            store = AgentStore(tmp_path / "line-edit.sqlite")
            profile = CompletionProfile(
                {"tests": ["tests/test_value.py"]}, tools, store
            )
            runtime = AgentRuntime(
                pool,
                store,
                tools.mapping(),
                RuntimeConfig(max_output_tokens=512),
                tool_descriptions=tools.tool_descriptions(),
                replay_safe_tools=tools.replay_safe_tools,
                completion_validator=profile,
            )
            try:
                result = await runtime.run(
                    "Replace a complete source function and independently check the result.",
                    run_id="line-edit-wire",
                )
                assert result.status == "completed" and result.completion_verified
                assert result.acceptance_passed is True
                assert (
                    path.read_text()
                    == "# keep header\ndef answer():\n    value = 42\n    return value\n# keep footer\n"
                )
                assert test_path.read_text() == original_test
                assert replica.calls == [
                    "line-edit-wire.root.1",
                    "line-edit-wire.root.2",
                    "line-edit-wire.root.3",
                ]
                catalog = replica.catalogs["line-edit-wire.root.2"]
                line_spec = next(
                    entry["function"]
                    for entry in catalog
                    if entry["function"]["name"] == "edit_lines"
                )
                assert (
                    not {"read_receipt", "expected_sha256"}
                    & line_spec["parameters"]["properties"].keys()
                )
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_runtime_cancel_reaches_remote_child_job_and_settles_journal(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = ScriptState()
    scripts = {
        "root": [
            action("spawn", task="Long child inference."),
            action("wait", agents=["agent-0001"]),
        ],
        "agent-0001": [action("finish", result="Never delivered after cancellation.")],
    }
    replica = ScriptedReplica(
        "cancel-control", scripts, state, delays={"agent-0001": 5.0}
    )
    with serve(replica) as (url, service):

        async def run():
            pool = WorkerPool([RemoteWorkerBackend(url, BEARER, poll_interval=0.002)])
            store = AgentStore(tmp_path / "cancel.sqlite")
            tools = WorkspaceTools(workspace, tmp_path / "artifacts")
            runtime = AgentRuntime(
                pool, store, tools.mapping(), RuntimeConfig(deadline_seconds=10)
            )
            task = asyncio.create_task(
                runtime.run("Delegate and wait for child work.", run_id="cancel-wire")
            )
            try:
                await eventually(
                    lambda: any(
                        ".agent-0001." in request for _, request, _ in state.calls
                    )
                )
                await runtime.cancel("cancel-wire")
                result = await asyncio.wait_for(task, timeout=2.0)
                assert result.status == "cancelled"
                assert all(agent.status == "cancelled" for agent in result.agents)
                await eventually(lambda: "cancel-wire.agent-0001.1" in state.cancelled)
                assert service.health()["active"] == 0
                assert result.usage.reserved_tokens == 0
                assert result.usage.uncertain_tokens > 0
                assert not any(
                    e["kind"] == "finish" for e in store.events("cancel-wire")
                )
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await pool.close()
                store.close()

        asyncio.run(run())


def test_http_child_tool_grants_and_per_agent_read_versions_guard_real_edits(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "config.txt"
    initial = "owner=original\nversion=1\nkeep=this line\n"
    path.write_text(initial)
    state = ScriptState()

    def reread_after_stale_rejection(messages):
        assert any("file version changed" in m.content for m in messages)
        return action("read_file", args={"path": "config.txt"})

    def finish_from_real_read(messages):
        evidence = tool_result(messages, "read_file")
        assert "owner=root" in evidence["content"]
        assert "version=2" in evidence["content"]
        assert "keep=this line" in evidence["content"]
        assert evidence["read_receipt"].startswith("read-")
        return action(
            "finish",
            result="Child edit and parent edit verified; stale parent read was rejected before refresh.",
        )

    scripts = {
        "root": [
            action("read_file", args={"path": "config.txt"}),
            action(
                "spawn",
                task="Change version=1 to version=2 after reading config.txt.",
                tools=["read_file", "replace_text"],
            ),
            action("wait", agents=["agent-0001"]),
            action(
                "replace_text",
                args={
                    "path": "config.txt",
                    "old_text": "owner=original",
                    "new_text": "owner=root",
                },
            ),
            reread_after_stale_rejection,
            action(
                "replace_text",
                args={
                    "path": "config.txt",
                    "old_text": "owner=original",
                    "new_text": "owner=root",
                },
            ),
            action("read_file", args={"path": "config.txt"}),
            finish_from_real_read,
        ],
        "agent-0001": [
            action("read_file", args={"path": "config.txt"}),
            action(
                "replace_text",
                args={
                    "path": "config.txt",
                    "old_text": "version=1",
                    "new_text": "version=2",
                },
            ),
            action(
                "finish", result="Applied version=2 under explicit replace_text grant."
            ),
        ],
    }
    replica = ScriptedReplica("write-version-control", scripts, state)
    with serve(replica) as (url, _):

        async def run():
            pool = WorkerPool([RemoteWorkerBackend(url, BEARER, poll_interval=0.002)])
            tools = WorkspaceTools(workspace, tmp_path / "artifacts", allow_write=True)
            store = AgentStore(tmp_path / "versions.sqlite")
            runtime = AgentRuntime(
                pool,
                store,
                tools.mapping(),
                tool_descriptions=tools.descriptions,
                replay_safe_tools=tools.replay_safe_tools,
            )
            try:
                result = await runtime.run(
                    "Delegate a narrow change; check file versions before parent edits.",
                    run_id="version-wire",
                )
                assert result.status == "completed"
                assert set(result.agents[1].tools) == {
                    "journal_read",
                    "read_file",
                    "replace_text",
                }
                assert path.read_text() == "owner=root\nversion=2\nkeep=this line\n"
                reads = [
                    event
                    for event in store.events("version-wire")
                    if event["kind"] == "tool_result"
                    and event["payload"]["name"] == "read_file"
                ]
                receipts = [
                    json.loads(event["payload"]["result"])["read_receipt"]
                    for event in reads
                ]
                assert len(receipts) == len(set(receipts)) == 4
                failures = [
                    event
                    for event in store.events("version-wire")
                    if event["kind"] == "tool_result"
                    and "file version changed" in event["payload"]["result"]
                ]
                assert len(failures) == 1 and failures[0]["agent_id"] == "root"
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_dispatched_provider_value_error_is_uncertain_and_not_preflight_rejection(
    tmp_path,
):
    state = ScriptState()

    def fail_after_dispatch(messages):
        raise ValueError("private native projection shape detail")

    replica = ScriptedReplica(
        "provider-failure", {"root": [fail_after_dispatch]}, state
    )
    with serve(replica) as (url, service):

        async def run():
            backend = RemoteWorkerBackend(url, BEARER, poll_interval=0.002)
            pool = WorkerPool([backend])
            store = AgentStore(tmp_path / "failure.sqlite")
            try:
                runtime = AgentRuntime(pool, store)
                result = await runtime.run(
                    "Test dispatched provider failure accounting.",
                    run_id="provider-error",
                )
                assert result.status == "failed"
                assert "ValueError" in result.error
                assert "private native" not in result.error
                assert len(state.calls) == 1
                assert result.usage.uncertain_tokens > 0
                assert (
                    result.usage.input_tokens
                    == result.usage.output_tokens
                    == result.usage.reserved_tokens
                    == 0
                )
                invalid = {
                    "request_id": "bad-admission",
                    "messages": [],
                    "max_tokens": 4,
                }
                with pytest.raises(RequestValidationError) as error:
                    await asyncio.to_thread(backend._http, "POST", "/v1/jobs", invalid)
                assert error.value.work_started is False
                assert len(state.calls) == 1 and service.health()["active"] == 0
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_transient_poll_failure_retries_same_replica_without_cancelling_job(tmp_path):
    state = ScriptState()
    scripts = {"root": [action("finish", result="Recovered the original worker job.")]}
    first = ScriptedReplica("recover-a", scripts, state, delays={"root": 0.03})
    second = ScriptedReplica("recover-b", scripts, state)
    with serve(first) as (url_a, _), serve(second) as (url_b, _):

        async def run():
            remote = RemoteWorkerBackend(url_a, BEARER, poll_interval=0.002)
            normal = remote._http
            failed_poll = False
            deletes = []

            def transient_poll(method, path, body=None, timeout=None):
                nonlocal failed_poll
                if method == "GET" and path.startswith("/v1/jobs/") and not failed_poll:
                    failed_poll = True
                    raise WorkerUnavailableError("one lost polling connection")
                if method == "DELETE":
                    deletes.append(path)
                return normal(method, path, body, timeout)

            remote._http = transient_poll
            pool = WorkerPool({"a": remote, "b": RemoteWorkerBackend(url_b, BEARER)})
            store = AgentStore(tmp_path / "recovery.sqlite")
            try:
                result = await AgentRuntime(pool, store).run(
                    "Recover a transient transport failure.", run_id="poll-recovery"
                )
                assert result.status == "completed"
                assert result.output == "Recovered the original worker job."
                assert failed_poll and deletes == []
                assert first.calls == ["poll-recovery.root.1"] and second.calls == []
                assert state.cancelled == []
                events = store.events("poll-recovery")
                errors = [e for e in events if e["kind"] == "backend_error"]
                generations = [e for e in events if e["kind"] == "generation"]
                assert len(errors) == len(generations) == 1
                assert (
                    errors[0]["payload"]["request_id"]
                    == generations[0]["payload"]["request_id"]
                )
                assert generations[0]["payload"]["attempt"] == 2
                assert (
                    result.usage.uncertain_tokens > 0
                    and result.usage.reserved_tokens == 0
                )
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


def test_final_retry_abandonment_cancels_retained_route_after_client_generation_ends(
    tmp_path,
):
    state = ScriptState()
    replica = ScriptedReplica(
        "abandonment-control",
        {"root": [action("finish", result="Should be cancelled before completion.")]},
        state,
        delays={"root": 5.0},
    )
    with serve(replica) as (url, service):

        async def run():
            remote = RemoteWorkerBackend(url, BEARER, poll_interval=0.002)
            normal = remote._http
            deletes = []

            def lose_poll(method, path, body=None, timeout=None):
                if method == "GET" and path.startswith("/v1/jobs/"):
                    raise WorkerUnavailableError(
                        "polling transport remains disconnected"
                    )
                if method == "DELETE":
                    deletes.append(path)
                return normal(method, path, body, timeout)

            remote._http = lose_poll
            pool = WorkerPool({"replica": remote})
            store = AgentStore(tmp_path / "abandonment.sqlite")
            try:
                result = await AgentRuntime(
                    pool, store, config=RuntimeConfig(max_backend_retries=0)
                ).run(
                    "Cancel an abandoned remote generation.", run_id="final-abandonment"
                )
                assert result.status == "failed"
                # generate() already returned its transport error; cleanup must
                # use sticky route history rather than currently active calls.
                assert not remote._active and not pool._requests
                assert pool.loads == {"replica": 0}
                await eventually(
                    lambda: "final-abandonment.root.1" in state.cancelled,
                    timeout=1.0,
                )
                assert deletes == ["/v1/jobs/final-abandonment.root.1"]
                assert service.health()["active"] == 0
                assert result.usage.uncertain_tokens > 0
                assert result.usage.reserved_tokens == 0
            finally:
                await pool.close()
                store.close()

        asyncio.run(run())


class FakeRemote:
    instances: ClassVar[list] = []
    max_concurrency = 1

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    async def health(self):
        return {"ready": True}

    async def close(self):
        self.closed = True


def test_cli_remote_config_translates_plaintext_opt_in_and_environment_auth(
    monkeypatch,
):
    FakeRemote.instances = []
    monkeypatch.setenv("TEST_WORKER_AUTH", "operator-secret")
    monkeypatch.setattr(cli, "RemoteWorkerBackend", FakeRemote)
    config = {
        "workers": [
            {
                "type": "remote",
                "name": "lan",
                "url": "http://192.0.2.1:8090",
                "token_env": "TEST_WORKER_AUTH",
                "allow_insecure_http": True,
                "max_concurrency": 2,
            }
        ]
    }

    async def run():
        pool = await cli.create_pool(config)
        try:
            options = FakeRemote.instances[0].kwargs
            assert options["token"] == "operator-secret"
            assert options["allow_insecure"] is True
            assert "token_env" not in options and "allow_insecure_http" not in options
            assert config["workers"][0]["token_env"] == "TEST_WORKER_AUTH"
        finally:
            await pool.close()
        assert FakeRemote.instances[0].closed

    asyncio.run(run())


@pytest.mark.parametrize(
    "workers",
    [
        [
            {
                "type": "remote",
                "url": "https://worker.example",
                "token_env": "MISSING_WORKER_TOKEN",
            }
        ],
        [
            {
                "type": "remote",
                "url": "https://worker.example",
                "token_env": "TEST_WORKER_AUTH",
                "invented": 1,
            }
        ],
        [{"type": "unknown"}],
        [
            {
                "type": "remote",
                "name": "same",
                "url": "https://worker.example",
                "token_env": "TEST_WORKER_AUTH",
            }
        ]
        * 2,
    ],
)
def test_cli_worker_config_rejects_unknown_options_missing_tokens_and_duplicate_names(
    monkeypatch, workers
):
    FakeRemote.instances = []
    monkeypatch.delenv("MISSING_WORKER_TOKEN", raising=False)
    monkeypatch.setenv("TEST_WORKER_AUTH", "operator-secret")
    monkeypatch.setattr(cli, "RemoteWorkerBackend", FakeRemote)
    with pytest.raises(ValueError):
        asyncio.run(cli.create_pool({"workers": workers}))
    assert all(worker.closed for worker in FakeRemote.instances)


@pytest.mark.parametrize(
    "configuration",
    [
        {"workers": [], "runtime": {}},
        {"workers": ["bad"]},
        {"workers": [{}], "unknown": True},
        {"workers": [{}], "runtime": {"max_agents": 0}},
        {"workers": [{}], "runtime": []},
    ],
)
def test_cli_load_config_rejects_invalid_top_level_shape(tmp_path, configuration):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration))
    with pytest.raises((ValueError, TypeError)):
        cli.load_config(path)


def test_cli_config_size_limit_applies_before_json_parse(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b"x" * 128_001)
    with pytest.raises(ValueError, match="128 KB"):
        cli.load_config(path)


@pytest.mark.parametrize("kind", ["local", "remote"])
def test_cli_failed_backend_readiness_closes_new_backend(monkeypatch, kind):
    created = []

    class FailedReadiness(FakeRemote):
        def __init__(self, *args, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

        async def start(self):
            raise WorkerUnavailableError("injected local startup failure")

        async def health(self):
            raise WorkerUnavailableError("injected remote readiness failure")

    monkeypatch.setenv("TEST_WORKER_AUTH", "operator-secret")
    monkeypatch.setattr(cli, "LocalForgeBackend", FailedReadiness)
    monkeypatch.setattr(cli, "RemoteWorkerBackend", FailedReadiness)
    worker = (
        {"type": "local", "model": "fixture.engine", "tokenizer": "fixture"}
        if kind == "local"
        else {
            "type": "remote",
            "url": "https://worker.example",
            "token_env": "TEST_WORKER_AUTH",
        }
    )
    with pytest.raises(WorkerUnavailableError):
        asyncio.run(cli.create_pool({"workers": [worker]}))
    assert len(created) == 1 and created[0].closed


class FakeStream:
    def __init__(self, data, *, blocks=False):
        self.data, self.blocks = data, blocks

    async def readline(self):
        if self.blocks:
            await asyncio.Event().wait()
        return self.data

    async def read(self, size):
        data, self.data = self.data, b""
        return data


class FakeProcess:
    def __init__(self, readiness, *, blocks=False):
        self.stdout = FakeStream(readiness, blocks=blocks)
        self.stderr = FakeStream(b"fixture startup diagnostic")
        self.returncode = None
        self.terminated = 0

    def terminate(self):
        self.terminated += 1
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    async def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@pytest.mark.parametrize("failure", ["invalid_json", "eof", "timeout", "health"])
def test_process_startup_failure_cleans_owned_process_and_preserves_parent_gpu_env(
    monkeypatch, failure
):
    readiness = b'{"event":"ready","port":32109}\n'
    if failure == "invalid_json":
        readiness = b"not JSON\n"
    elif failure == "eof":
        readiness = b""
    process = FakeProcess(readiness, blocks=failure == "timeout")
    environment = {}

    async def spawn(*command, **options):
        environment.update(options["env"])
        assert command[1:3] == ("-m", "forge_llm.agents.worker")
        return process

    async def bad_health(self):
        raise WorkerUnavailableError("injected readiness health failure")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-gpu")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if failure == "health":
        monkeypatch.setattr(RemoteWorkerBackend, "health", bad_health)
    config = WorkerConfig("fixture.engine", "fixture", cuda_visible_devices="child-gpu")
    with pytest.raises((ValueError, WorkerUnavailableError, asyncio.TimeoutError)):
        asyncio.run(LocalProcessBackend.start(config, startup_timeout=0.01))
    assert process.returncode is not None
    assert process.terminated == (0 if failure == "eof" else 1)
    assert environment["CUDA_VISIBLE_DEVICES"] == "child-gpu"
    assert len(environment["FORGE_WORKER_TOKEN"]) >= 32
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-gpu"


class ManifestBackend:
    def __init__(self, identity):
        self.identity = identity
        self.calls = []
        self.closed = False

    def health(self):
        return {"ready": True, **self.identity}

    async def generate(self, messages, maximum, request_id):
        self.calls.append(request_id)
        return Generation(action("finish", result="Fixture answer."), 20, 10)

    async def close(self):
        self.closed = True


def cli_args(tmp_path, config):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    path = tmp_path / "workers.json"
    path.write_text(json.dumps(config))
    return SimpleNamespace(
        command="run",
        state_dir=str(tmp_path / "state"),
        config=str(path),
        workspace=str(workspace),
        run_id="bound-identity",
        task="Fixture task.",
        task_file=None,
        allow_write=False,
        allow_tests=False,
        quiet=True,
    )


def identity():
    return {
        "model": "unchanged-model-path.engine",
        "tokenizer": "fixture-tokenizer",
        "backend": "scripted-control",
        "max_model_length": 2048,
        "model_data_sha256": "a" * 64,
        "model_config_sha256": "b" * 64,
        "tokenizer_signature": "c" * 64,
        "draft_model_data_sha256": None,
        "speculative": None,
        "native_tool_calls": False,
    }


def test_cli_resume_binds_health_fingerprints_and_closes_pool_on_drift(
    tmp_path, monkeypatch, capsys
):
    current = identity()
    backends = []

    async def factory(config):
        backend = ManifestBackend(dict(current))
        backends.append(backend)
        return WorkerPool({"replica": backend})

    monkeypatch.setattr(cli, "create_pool", factory)
    args = cli_args(
        tmp_path,
        {
            "workers": [
                {
                    "model": "unchanged-model-path.engine",
                    "tokenizer": "fixture-tokenizer",
                }
            ]
        },
    )
    assert asyncio.run(cli.execute(args)) == 0
    assert backends[0].closed and backends[0].calls == ["bound-identity.root.1"]
    manifest_path = tmp_path / "state" / "bound-identity.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["worker_identities"]["replica"] == current
    args.command = "resume"
    assert asyncio.run(cli.execute(args)) == 0
    assert backends[-1].closed and not backends[-1].calls
    for field in (
        "model_data_sha256",
        "model_config_sha256",
        "tokenizer_signature",
        "draft_model_data_sha256",
        "native_tool_calls",
    ):
        old = current[field]
        current[field] = True if field == "native_tool_calls" else "d" * 64
        with pytest.raises(ValueError, match="changed"):
            asyncio.run(cli.execute(args))
        assert backends[-1].closed and not backends[-1].calls
        current[field] = old
    assert json.loads(manifest_path.read_text()) == manifest
    capsys.readouterr()


def test_cli_failed_startup_leaves_no_manifest_and_same_id_can_retry(
    tmp_path, monkeypatch, capsys
):
    async def failed(config):
        raise WorkerUnavailableError("worker did not become ready")

    args = cli_args(
        tmp_path, {"workers": [{"model": "fixture.engine", "tokenizer": "fixture"}]}
    )
    monkeypatch.setattr(cli, "create_pool", failed)
    with pytest.raises(WorkerUnavailableError):
        asyncio.run(cli.execute(args))
    assert not (tmp_path / "state" / "bound-identity.manifest.json").exists()
    backend = ManifestBackend(identity())

    async def recovered(config):
        return WorkerPool({"replica": backend})

    monkeypatch.setattr(cli, "create_pool", recovered)
    assert asyncio.run(cli.execute(args)) == 0
    assert backend.closed
    capsys.readouterr()


def test_cli_missing_worker_identity_closes_pool_without_creating_manifest(
    tmp_path, monkeypatch
):
    metadata = identity()
    metadata["tokenizer_signature"] = None
    backend = ManifestBackend(metadata)

    async def factory(config):
        return WorkerPool({"replica": backend})

    args = cli_args(
        tmp_path, {"workers": [{"model": "fixture.engine", "tokenizer": "fixture"}]}
    )
    monkeypatch.setattr(cli, "create_pool", factory)
    with pytest.raises(ValueError, match="fingerprints"):
        asyncio.run(cli.execute(args))
    assert backend.closed
    assert not (tmp_path / "state" / "bound-identity.manifest.json").exists()
