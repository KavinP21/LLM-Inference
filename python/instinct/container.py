"""Serving a Mac's models to Instinct running in Docker.

Docker on a Mac can't reach the Apple GPU, so ``tools/docker/instinct.sh``
splits the app. The models run natively here at full speed, one model worker
per model, listening on this Mac's 127.0.0.1 with a bearer token. The app,
the agents, their commands and their files run in the container, which
reaches the workers through Docker's ``host.docker.internal``.

``python -m instinct.container serve-models`` starts the workers the main
agent and sub-agents use (from ``examples/instinct.json`` unless told
otherwise). It writes the container's worker configuration into the folder
the container mounts as its state, then runs until it's stopped. The token
comes from ``$INSTINCT_WORKER_TOKEN``, which the launcher makes fresh each
time and hands to the container the same way.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .agents.backends import memory
from .agents.backends.config import WorkerConfig
from .agents.cli import load_config

TOKEN_ENV = "INSTINCT_WORKER_TOKEN"
HOST_ALIAS = "host.docker.internal"


def model_workers(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The local workers serving the runtime's main and sub-agent groups."""
    runtime = config.get("runtime", {})
    groups = {runtime.get("driver_group"), runtime.get("subagent_group")} - {None}
    picked = [
        worker
        for worker in config["workers"]
        if worker.get("type", "process") in {"process", "local"}
        and (worker.get("group") in groups or not groups)
    ]
    if not picked:
        raise ValueError("the configuration has no local model workers to serve")
    return picked


def worker_definition(worker: dict[str, Any]) -> dict[str, Any]:
    """A worker server's ``--config-json``: the WorkerConfig fields only."""
    fields = {field.name for field in dataclasses.fields(WorkerConfig)}
    definition = {key: value for key, value in worker.items() if key in fields}
    definition.setdefault("worker_id", worker.get("name", "worker"))
    options = dict(definition.get("engine_options", {}))
    options.setdefault("memory_limit_bytes", memory.mlx_limit(worker))
    definition["engine_options"] = options
    WorkerConfig(**definition)  # Fail here, not in a child process.
    return definition


def container_config(
    config: dict[str, Any], workers: list[dict[str, Any]], ports: list[int]
) -> dict[str, Any]:
    """What the app in the container runs: remote workers at this Mac."""
    remote = []
    for worker, port in zip(workers, ports, strict=True):
        remote.append(
            {
                "type": "remote",
                "name": worker["name"],
                "group": worker["group"],
                "label": Path(worker["model"]).name,
                "url": f"http://{HOST_ALIAS}:{port}",
                "token_env": TOKEN_ENV,
                # Docker's host alias never leaves this machine.
                "allow_insecure_http": True,
                "max_concurrency": int(worker.get("max_pending", 8)),
                "request_timeout": float(worker.get("request_timeout", 600)) + 60,
                "connect_timeout": 15,
            }
        )
    return {"workers": remote, "runtime": dict(config.get("runtime", {}))}


def serve_models(config_path: Path, output: Path, port: int) -> int:
    if len(os.environ.get(TOKEN_ENV, "")) < 32:
        print(f"set a token of at least 32 characters in ${TOKEN_ENV}", file=sys.stderr)
        return 2
    config = load_config(config_path)
    workers = model_workers(config)
    definitions = [worker_definition(worker) for worker in workers]
    processes: list[subprocess.Popen] = []
    ports = []
    stopped = threading.Event()

    def stop(*_: Any) -> None:
        stopped.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    output.unlink(missing_ok=True)
    try:
        for index, (worker, definition) in enumerate(zip(workers, definitions)):
            print(
                f"Loading {worker['name']} ({Path(worker['model']).name})...",
                flush=True,
            )
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "instinct.agents.worker",
                    "--config-json",
                    json.dumps(definition),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port + index),
                    "--token-env",
                    TOKEN_ENV,
                    "--exit-with-parent",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
            ready = None
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "ready":
                    ready = event
                    break
            if ready is None or stopped.is_set():
                print(f"{worker['name']} didn't start", file=sys.stderr)
                return 1
            ports.append(int(ready["port"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(container_config(config, workers, ports), indent=2)
        )
        os.replace(temporary, output)
        print(
            f"Models ready on 127.0.0.1 ports {', '.join(map(str, ports))}.", flush=True
        )
        while not stopped.is_set():
            for process in processes:
                if process.poll() is not None:
                    print("A model worker stopped.", file=sys.stderr)
                    return 1
            stopped.wait(1.0)
        return 0
    finally:
        stop()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        output.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m instinct.container")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve-models", help="serve this Mac's models to Docker"
    )
    serve.add_argument("--config", type=Path, default=Path("examples/instinct.json"))
    serve.add_argument("--output", type=Path, required=True)
    serve.add_argument("--port", type=int, default=8791)
    args = parser.parse_args(argv)
    return serve_models(args.config, args.output, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
