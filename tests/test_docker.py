"""Instinct in Docker: the Mac's models served to the container, the app's
server inside a container, and the container counting as the sandbox."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from instinct import container
from instinct.agents import cli as agents_cli
from instinct.harness import cli as harness_cli
from instinct.harness import commands, sandbox
from instinct.harness.server import AppServer

ROOT = Path(__file__).resolve().parents[1]


def _mac_profile(tmp_path):
    config = json.loads((ROOT / "examples/instinct.json").read_text())
    for worker in config["workers"]:
        folder = tmp_path / Path(worker["model"]).name
        folder.mkdir(exist_ok=True)
        worker["model"] = worker["tokenizer"] = str(folder)
    return config


def test_the_container_gets_the_mac_profiles_main_and_sub_agent_models(tmp_path):
    config = _mac_profile(tmp_path)
    workers = container.model_workers(config)
    assert [w["name"] for w in workers] == ["gemma-4", "qwen"]  # Not abliterated.
    definitions = [container.worker_definition(w) for w in workers]
    assert definitions[0]["worker_id"] == "gemma-4"
    assert "name" not in definitions[0] and "lazy" not in definitions[1]
    assert definitions[1]["engine_options"]["kv_bits"] == 8
    assert definitions[0]["engine_options"]["memory_limit_bytes"] > 0
    made = container.container_config(config, workers, [8791, 8792])
    gemma, qwen = made["workers"]
    assert gemma["url"] == "http://host.docker.internal:8791"
    assert gemma["token_env"] == container.TOKEN_ENV and gemma["allow_insecure_http"]
    assert gemma["label"] == "gemma-4-26b-a4b-mlx" and qwen["group"] == "subagent"
    assert made["runtime"] == config["runtime"]
    assert "memory" not in made  # Nothing loads in the container.


def test_the_containers_config_starts_remote_workers(tmp_path, monkeypatch):
    config = _mac_profile(tmp_path)
    workers = container.model_workers(config)
    made = container.container_config(config, workers, [8791, 8792])
    monkeypatch.setenv(container.TOKEN_ENV, "t" * 40)
    started = []

    class Remote:
        def __init__(self, url, token, **options):
            started.append((url, options))

        async def health(self):
            return {"ready": True}

        async def close(self):
            pass

    monkeypatch.setattr(agents_cli, "RemoteWorkerBackend", Remote)
    path = tmp_path / "host-models.json"
    path.write_text(json.dumps(made))

    async def build():
        pool = await agents_cli.create_pool(agents_cli.load_config(path))
        await pool.close()

    asyncio.run(build())
    assert [url for url, _ in started] == [
        "http://host.docker.internal:8791",
        "http://host.docker.internal:8792",
    ]
    assert all(options["allow_insecure"] for _, options in started)
    assert all("label" not in options for _, options in started)


def test_serve_models_needs_a_token(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(container.TOKEN_ENV, raising=False)
    code = container.main(["serve-models", "--output", str(tmp_path / "out.json")])
    assert code == 2 and container.TOKEN_ENV in capsys.readouterr().err


def test_the_server_listens_in_a_container_but_still_wants_127_0_0_1():
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(ValueError, match="127.0.0.1"):
            AppServer(("0.0.0.0", 0), object(), loop)
        server = AppServer(("0.0.0.0", 0), object(), loop, container=True)
        try:
            assert server.origin == f"http://127.0.0.1:{server.server_port}"
        finally:
            server.server_close()
        with pytest.raises(ValueError):
            AppServer(("192.168.1.5", 0), object(), loop, container=True)
    finally:
        loop.close()


def test_container_mode_is_refused_outside_a_container(monkeypatch):
    monkeypatch.setattr(sandbox, "in_container", lambda: False)
    monkeypatch.setattr(sys, "argv", ["instinct", "--demo", "--container"])
    with pytest.raises(SystemExit) as exited:
        harness_cli.main()
    assert exited.value.code == 2


def test_the_container_counts_as_the_sandbox(monkeypatch):
    monkeypatch.setattr(sandbox, "available", lambda: False)
    monkeypatch.setattr(sandbox, "in_container", lambda: True)
    assert sandbox.kind() == "container" and sandbox.confined()
    tool = commands.CommandTools.__new__(commands.CommandTools)
    tool.network = True
    assert "Docker container" in tool.description()
    monkeypatch.setattr(sandbox, "in_container", lambda: False)
    assert sandbox.kind() is None and not sandbox.confined()
    assert "own permissions" in tool.description()


def test_the_image_and_launcher_are_wired_up():
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    for target in ("AS app", "AS nvidia"):
        assert target in dockerfile
    assert "USER instinct" in dockerfile
    ignored = (ROOT / ".dockerignore").read_text()
    for kept in (
        "!python/",
        "!examples/",
        "!docker/",
        "!tools/nvidia/requirements.txt",
    ):
        assert kept in ignored
    script = ROOT / "tools/docker/instinct.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "docker/entrypoint.sh")], check=True)
    usage = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert usage.returncode == 2 and "start" in usage.stdout
    text = script.read_text()
    for flag in (
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        '"127.0.0.1:$port:$port"',
    ):
        assert flag in text
