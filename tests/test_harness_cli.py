from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest
from instinct.harness import cli


def _args(workspace, **updates):
    values = {
        "workspace": str(workspace),
        "state_dir": ".instinct/studio",
        "skills_dir": [],
        "mcp_config": None,
        "config": None,
        "demo": True,
        "port": 0,
        "no_open": True,
        "app": False,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _definition(**updates):
    values = {
        "name": "demo",
        "command": [sys.executable, "server.py"],
        "allowed_tools": ["echo.text"],
    }
    values.update(updates)
    return values


def _write_config(tmp_path, value):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_operator_mcp_config_uses_workspace_relative_cwd_and_explicit_arrays(tmp_path):
    directory = tmp_path / "servers"
    directory.mkdir()
    path = _write_config(tmp_path, [_definition(cwd="servers")])
    selected = cli.load_mcp_config(path, tmp_path)[0]
    assert selected.cwd == directory.resolve()
    assert selected.command == (sys.executable, "server.py")
    assert selected.allowed_tools == ("echo.text",)
    path = _write_config(tmp_path, [_definition()])
    assert cli.load_mcp_config(path, tmp_path)[0].cwd == tmp_path.resolve()


@pytest.mark.parametrize(
    "definition",
    [
        {},
        None,
        "server",
        [],
        _definition(command="python server.py"),
        _definition(allowed_tools="echo.text"),
        _definition(cwd=None),
        _definition(name=1),
        _definition(shell=True),
        _definition(max_tools=True),
        _definition(timeout_seconds=float("inf")),
    ],
)
def test_invalid_mcp_server_records_fail_before_process_construction(
    tmp_path, definition
):
    path = _write_config(tmp_path, [definition])
    with pytest.raises(ValueError):
        cli.load_mcp_config(path, tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        {},
        [_definition()] * 2,
        [_definition(name=f"server{index}") for index in range(9)],
    ],
)
def test_mcp_config_rejects_duplicates_wrong_shape_and_server_count(tmp_path, value):
    path = _write_config(tmp_path, value)
    with pytest.raises(ValueError):
        cli.load_mcp_config(path, tmp_path)


@pytest.mark.parametrize(
    "data",
    [
        b'[{"name":"one","name":"two","command":["python"]}]',
        b"[NaN]",
        b"\xff",
        b"not JSON",
    ],
)
def test_mcp_config_requires_strict_utf8_json(tmp_path, data):
    path = tmp_path / "invalid.json"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="invalid JSON"):
        cli.load_mcp_config(path, tmp_path)


def test_mcp_config_is_bounded_and_requires_regular_file(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * 128_001)
    with pytest.raises(ValueError, match="128 KB"):
        cli.load_mcp_config(path, tmp_path)
    with pytest.raises(ValueError, match="regular file"):
        cli.load_mcp_config(tmp_path, tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO config")
def test_mcp_fifo_configuration_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "config.pipe"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        cli.load_mcp_config(path, tmp_path)


def test_demo_rejects_mcp_before_loading_operator_configuration(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("demo tried to construct an MCP client")

    monkeypatch.setattr(cli, "MCPStdioClient", unexpected)
    with pytest.raises(ValueError, match="--demo"):
        asyncio.run(cli.serve(_args(tmp_path, mcp_config="unreadable-file.json")))


def test_main_reports_demo_mcp_combination_as_a_cli_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["instinct", "--demo", "--mcp-config", "mcp.json"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--demo cannot be combined" in capsys.readouterr().err


@pytest.fixture
def lifecycle_fakes(monkeypatch):
    events = []
    state = SimpleNamespace(events=events, kwargs=None, server=None, failure=None)

    class Service:
        def __init__(self, workspace, state_dir, **kwargs):
            state.workspace, state.state_dir, state.kwargs = (
                workspace,
                state_dir,
                kwargs,
            )
            events.append("service-created")

        async def close(self):
            events.append("service-closed")

        async def start_backend(self):
            events.append("backend-started")
            return {"status": "loading"}

    class Server:
        origin = "http://127.0.0.1:12345"
        bootstrap_token = "TEST-BOOTSTRAP-NOT-PRINTED"

        def __init__(self, address, service, loop, token=None):
            assert address[0] == "127.0.0.1"
            if state.failure == "bind":
                raise OSError("address unavailable")
            events.append("server-created")
            state.server = self

        def serve_forever(self):
            pass

        def shutdown(self):
            events.append("server-shutdown")
            if state.failure == "shutdown":
                raise RuntimeError("shutdown failed")

        def server_close(self):
            events.append("server-closed")

    class Thread:
        def __init__(self, **kwargs):
            events.append("thread-created")

        def start(self):
            if state.failure == "thread":
                raise RuntimeError("thread unavailable")
            events.append("thread-started")

        def join(self, timeout):
            events.append("thread-joined")

    def open_browser(url):
        assert url.endswith("/#token=TEST-BOOTSTRAP-NOT-PRINTED")
        events.append("browser-opened")
        if state.failure == "browser":
            raise RuntimeError("browser unavailable")

    monkeypatch.setattr(cli, "StudioService", Service)
    monkeypatch.setattr(cli, "StudioHTTPServer", Server)
    monkeypatch.setattr(cli.threading, "Thread", Thread)
    monkeypatch.setattr(cli.webbrowser, "open", open_browser)

    # asyncio.to_thread uses threading too; keep these deterministic tests inline.
    async def to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(cli.asyncio, "to_thread", to_thread)
    return state


@pytest.mark.parametrize("failure", ["bind", "thread", "browser"])
def test_service_is_closed_after_any_launcher_startup_failure(
    tmp_path, lifecycle_fakes, failure
):
    state = lifecycle_fakes
    state.failure = failure
    with pytest.raises((OSError, RuntimeError)):
        asyncio.run(cli.serve(_args(tmp_path, no_open=False)))
    assert state.events[-1] == "service-closed"
    if failure != "bind":
        assert "server-closed" in state.events
    if failure == "thread":
        assert "server-shutdown" not in state.events
    if failure == "browser":
        assert "server-shutdown" in state.events


def test_launcher_cancellation_shuts_down_and_joins_server_without_printing_token(
    tmp_path, lifecycle_fakes, capsys
):
    async def run():
        task = asyncio.create_task(cli.serve(_args(tmp_path)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert lifecycle_fakes.events[-4:] == [
        "server-shutdown",
        "server-closed",
        "thread-joined",
        "service-closed",
    ]
    assert lifecycle_fakes.state_dir == tmp_path.resolve() / ".instinct/studio"
    assert "TEST-BOOTSTRAP-NOT-PRINTED" not in capsys.readouterr().out


def test_service_still_closes_when_server_shutdown_fails(tmp_path, lifecycle_fakes):
    lifecycle_fakes.failure = "shutdown"

    async def run():
        task = asyncio.create_task(cli.serve(_args(tmp_path)))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(RuntimeError, match="shutdown failed"):
            await task

    asyncio.run(run())
    assert lifecycle_fakes.events[-2:] == ["server-closed", "service-closed"]


def test_live_cli_prewarms_backend_and_constructs_owned_mcp_clients(
    tmp_path, lifecycle_fakes, monkeypatch
):
    path = _write_config(tmp_path, [_definition()])
    monkeypatch.setattr(
        cli, "load_config", lambda path: {"workers": ["operator-selected"]}
    )

    async def run():
        task = asyncio.create_task(
            cli.serve(
                _args(tmp_path, demo=False, config="agents.json", mcp_config=str(path))
            )
        )
        await asyncio.sleep(0)
        client = lifecycle_fakes.kwargs["mcp_clients"][0]
        assert client.process is None
        assert "backend-started" in lifecycle_fakes.events
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
