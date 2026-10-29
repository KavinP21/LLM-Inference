from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from instinct.agents.native_tools import build_tool_specs
from instinct.agents.protocol import Generation
from instinct.agents.runtime import AgentRuntime
from instinct.harness.extensions import (
    MCPError,
    MCPServerConfig,
    MCPStdioClient,
    SkillRegistry,
)


def _skill(
    root: Path, name="review", description="Review code.", body="Do useful work."
):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8"
    )
    return directory


def test_skills_discover_metadata_then_load_and_read_references(tmp_path):
    directory = _skill(tmp_path, body="Use references/checklist.md.")
    (directory / "references").mkdir()
    (directory / "references" / "checklist.md").write_text("Check ownership.")
    registry = SkillRegistry([tmp_path])
    catalog = registry.catalog()
    assert catalog["skills"][0]["name"] == "review"
    assert "content" not in catalog["skills"][0]
    loaded = json.loads(asyncio.run(registry.load_skill({"name": "review"})))
    assert "Use references/checklist.md." in loaded["content"]
    assert len(loaded["sha256"]) == 64
    reference = json.loads(
        asyncio.run(
            registry.read_resource(
                {"name": "review", "path": "references/checklist.md"}
            )
        )
    )
    assert reference["content"] == "Check ownership."
    assert set(registry.mapping()) == {"skill_list", "skill_load", "skill_read"}
    build_tool_specs(
        registry.schemas, allowed_tools=list(registry.mapping()), controls=[]
    )


def test_discovery_does_not_load_an_oversized_body_and_supports_folded_metadata(
    tmp_path,
):
    directory = _skill(
        tmp_path, description=">\n  Review code\n  with care.", body="x" * 2000
    )
    registry = SkillRegistry([directory], max_file_bytes=256, max_metadata_bytes=128)
    assert registry.catalog()["skills"][0]["description"] == "Review code with care."
    with pytest.raises(ValueError, match="byte limit"):
        asyncio.run(registry.load_skill({"name": "review"}))


@pytest.mark.parametrize(
    "relative",
    [
        "../outside",
        "/etc/passwd",
        "references/../../outside",
        ".env",
        "secrets/key",
        "a\\b",
        "foo/./bar",
    ],
)
def test_skill_resource_boundaries(tmp_path, relative):
    _skill(tmp_path)
    registry = SkillRegistry([tmp_path])
    with pytest.raises(ValueError, match="relative path"):
        asyncio.run(registry.read_resource({"name": "review", "path": relative}))


def test_skill_symlink_and_metadata_changes_are_rejected(tmp_path):
    directory = _skill(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    (directory / "link").symlink_to(outside)
    registry = SkillRegistry([tmp_path])
    with pytest.raises(ValueError, match="symlink"):
        asyncio.run(registry.read_resource({"name": "review", "path": "link"}))
    (directory / "SKILL.md").write_text(
        "---\nname: review\ndescription: Changed.\n---\nchanged"
    )
    with pytest.raises(ValueError, match="metadata changed"):
        asyncio.run(registry.load_skill({"name": "review"}))
    registry.refresh()
    assert registry.catalog()["skills"][0]["description"] == "Changed."


def test_skill_invalid_front_matter_duplicate_names_and_bounded_scan(tmp_path):
    directory = _skill(tmp_path)
    (tmp_path / "invalid").mkdir()
    (tmp_path / "invalid" / "SKILL.md").write_text("no metadata")
    registry = SkillRegistry([tmp_path])
    assert registry.warnings and len(registry.records) == 1
    with pytest.raises(ValueError, match="duplicate skill"):
        SkillRegistry([tmp_path, directory])
    _skill(tmp_path, name="another")
    with pytest.raises(ValueError, match="skill limit"):
        SkillRegistry([tmp_path], max_skills=1)


def test_cataloged_directory_cannot_be_replaced_with_a_symlink(tmp_path):
    directory = _skill(tmp_path)
    registry = SkillRegistry([tmp_path])
    moved = tmp_path / "moved"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        asyncio.run(registry.load_skill({"name": "review"}))


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO boundary")
def test_skill_fifo_does_not_block(tmp_path):
    directory = _skill(tmp_path)
    os.mkfifo(directory / "pipe")
    registry = SkillRegistry([tmp_path])
    with pytest.raises(ValueError, match="regular"):
        asyncio.run(registry.read_resource({"name": "review", "path": "pipe"}))


@pytest.fixture
def server_script(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        """import json, os, sys, time
mode = sys.argv[1]
if mode == 'descendant':
    import signal, subprocess
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])
    with open('child.pid', 'w') as file:
        file.write(str(child.pid))
initialized = False
for line in sys.stdin:
    req = json.loads(line)
    method = req['method']
    if 'id' not in req:
        if method == 'notifications/initialized':
            initialized = True
        continue
    def reply(result=None, error=None):
        value = {'jsonrpc': '2.0', 'id': req['id']}
        value['error' if error else 'result'] = error or result
        print(json.dumps(value), flush=True)
    if method == 'server/discover':
        if mode in {'modern', 'input', 'version'}:
            assert req['params']['_meta']['io.modelcontextprotocol/protocolVersion'] == '2026-07-28'
            reply({'resultType':'complete','supportedVersions':['1900-01-01' if mode == 'version' else '2026-07-28'],'capabilities':{'tools':{}}})
        elif mode == 'quiet-probe':
            continue
        else:
            reply(error={'code':-32602,'message':'legacy only'})
    elif method == 'initialize':
        reply({'protocolVersion':'2025-11-25','capabilities':{'tools':{}}})
    elif method == 'tools/list':
        if mode in {'modern', 'input'}:
            assert req['params']['_meta']['io.modelcontextprotocol/clientCapabilities'] == {}
        else:
            assert initialized
        schema = {'type':'object','properties':{'text':{'type':'string'}},'required':['text']}
        if mode == 'schema':
            schema['$ref'] = 'unsupported'
        reply({'resultType':'complete','tools':[{'name':'echo.text','description':'Echo text.','inputSchema':schema,'annotations':{'readOnlyHint':True}}]})
    elif method == 'tools/call':
        if mode in {'slow', 'descendant'}:
            time.sleep(30)
        elif mode == 'bad-json':
            print('not JSON', flush=True)
        elif mode == 'oversized':
            print('x'*5000, flush=True)
        elif mode == 'rpc-error':
            reply(error={'code':-32602,'message':'SENSITIVE ERROR NOT FOR LOGS'})
        elif mode == 'input':
            reply({'resultType':'input_required','inputRequests':{}})
        else:
            result = {'text': req['params']['arguments']['text'], 'secret_inherited': 'INSTINCT_TEST_SECRET' in os.environ}
            reply({'resultType':'complete','content':[{'type':'text','text':json.dumps(result)}],'isError':False})
"""
    )
    return script


def _config(server_script, mode, **kwargs):
    return MCPServerConfig(
        "demo",
        (sys.executable, str(server_script), mode),
        server_script.parent,
        ("echo.text",),
        **kwargs,
    )


@pytest.mark.parametrize(
    "mode,version",
    [("modern", "2026-07-28"), ("legacy", "2025-11-25"), ("quiet-probe", "2025-11-25")],
)
def test_mcp_current_and_legacy_protocols_expose_native_schema(
    server_script, monkeypatch, mode, version
):
    monkeypatch.setenv("INSTINCT_TEST_SECRET", "private")

    async def run():
        async with MCPStdioClient(
            _config(server_script, mode, probe_timeout_seconds=0.3)
        ) as client:
            assert client.protocol_version == version
            alias = next(iter(client.mapping()))
            assert alias.startswith("mcp_demo_echo_text_")
            assert client.schemas[alias]["additionalProperties"] is False
            build_tool_specs(client.schemas, allowed_tools=[alias], controls=[])
            result = json.loads(await client.mapping()[alias]({"text": "hello"}))
            content = json.loads(result["content"][0]["text"])
            assert content == {"text": "hello", "secret_inherited": False}
            assert alias in client.mutating_tools and not client.replay_safe_tools
        assert client.process.returncode is not None

    asyncio.run(run())


def test_mcp_selects_only_operator_granted_tools_and_rejects_invalid_arguments(
    server_script,
):
    async def run():
        config = MCPServerConfig(
            "demo", (sys.executable, str(server_script), "legacy"), server_script.parent
        )
        async with MCPStdioClient(config) as client:
            assert client.mapping() == {}
            with pytest.raises(ValueError, match="allowlist"):
                await client.call_tool("echo.text", {})
        async with MCPStdioClient(_config(server_script, "legacy")) as client:
            alias = next(iter(client.mapping()))
            with pytest.raises(ValueError):
                await client.call_tool(alias, {"text": "ok", "command": "not allowed"})

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["schema", "version"])
def test_mcp_negotiation_and_schema_failures_clean_up(server_script, mode):
    async def run():
        client = MCPStdioClient(_config(server_script, mode))
        with pytest.raises((ValueError, RuntimeError)):
            await client.start()
        assert client.process.returncode is not None

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode,error",
    [("slow", TimeoutError), ("bad-json", RuntimeError), ("oversized", RuntimeError)],
)
def test_mcp_failed_and_timed_out_calls_terminate_owned_process(
    server_script, mode, error
):
    async def run():
        client = MCPStdioClient(
            _config(server_script, mode, timeout_seconds=0.1, max_message_bytes=1024)
        )
        await client.start()
        alias = next(iter(client.mapping()))
        with pytest.raises(error):
            await client.call_tool(alias, {"text": "hello"})
        assert client.process.returncode is not None

    asyncio.run(run())


def test_mcp_outer_cancellation_closes_server(server_script):
    async def run():
        client = MCPStdioClient(_config(server_script, "slow"))
        await client.start()
        alias = next(iter(client.mapping()))
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.call_tool(alias, {"text": "hello"}), 0.1)
        assert client.process.returncode is not None

    asyncio.run(run())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_mcp_timeout_kills_sigterm_resistant_descendants(server_script):
    async def run():
        client = MCPStdioClient(
            _config(server_script, "descendant", timeout_seconds=0.1)
        )
        await client.start()
        child_pid = int((server_script.parent / "child.pid").read_text())
        assert os.getpgid(child_pid) == client.process.pid
        with pytest.raises(TimeoutError):
            await client.call_tool(next(iter(client.mapping())), {"text": "hello"})
        assert client.process.returncode is not None
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("owned descendant survived MCP timeout")

    asyncio.run(run())


def test_rpc_error_does_not_log_server_error_text(server_script):
    async def run():
        async with MCPStdioClient(_config(server_script, "rpc-error")) as client:
            alias = next(iter(client.mapping()))
            with pytest.raises(MCPError) as error:
                await client.call_tool(alias, {"text": "hello"})
            assert str(error.value) == "MCP request rejected (code -32602)"

    asyncio.run(run())


def test_mcp_input_required_result_is_reported_without_requesting_secrets(
    server_script,
):
    async def run():
        async with MCPStdioClient(_config(server_script, "input")) as client:
            alias = next(iter(client.mapping()))
            with pytest.raises(RuntimeError, match="operator interaction"):
                await client.call_tool(alias, {"text": "hello"})

    asyncio.run(run())


@pytest.mark.parametrize(
    "updates",
    [
        {"command": "python server.py"},
        {"allowed_tools": ["echo", "echo"]},
        {"timeout_seconds": float("inf")},
        {"max_tools": True},
    ],
)
def test_mcp_config_rejects_ambiguous_or_unbounded_values(tmp_path, updates):
    values = {"name": "demo", "command": (sys.executable,), "cwd": tmp_path}
    values.update(updates)
    with pytest.raises(ValueError):
        MCPServerConfig(**values)


@pytest.mark.parametrize("mode", ["modern", "legacy"])
def test_mcp_tool_results_flow_through_existing_durable_runtime(server_script, mode):
    class Backend:
        def __init__(self, alias):
            self.responses = iter(
                [
                    {
                        "action": "tool",
                        "name": alias,
                        "args": {"text": "runtime evidence"},
                    },
                    {"action": "finish", "result": "Received runtime evidence."},
                ]
            )

        async def generate(self, messages, max_tokens, request_id):
            return Generation(
                json.dumps(next(self.responses)), input_tokens=10, output_tokens=10
            )

    async def run():
        async with MCPStdioClient(_config(server_script, mode)) as client:
            alias = next(iter(client.mapping()))
            runtime = AgentRuntime(
                Backend(alias),
                tools=client.mapping(),
                tool_descriptions=client.tool_descriptions(),
                tool_schemas=client.schemas,
                replay_safe_tools=client.replay_safe_tools,
                mutating_tools=client.mutating_tools,
            )
            try:
                result = await runtime.run(
                    "Use the granted echo tool.", run_id="extension-test"
                )
                assert result.status == "completed"
                events = runtime.store.events(result.run_id)
                tool_result = next(
                    event for event in events if event["kind"] == "tool_result"
                )
                assert "runtime evidence" in tool_result["payload"]["result"]
            finally:
                runtime.store.close()

    asyncio.run(run())
