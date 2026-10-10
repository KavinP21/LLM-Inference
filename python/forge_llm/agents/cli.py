"""Run, resume, inspect or cancel local journaled agent tasks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from .backends import (
    LocalForgeBackend,
    LocalProcessBackend,
    RemoteWorkerBackend,
    WorkerConfig,
    WorkerPool,
)
from .profiles import CompletionProfile
from .runtime import AgentRuntime, RuntimeConfig
from .store import AgentStore
from .tools import WorkspaceTools


def load_config(path: str | Path) -> dict:
    data = Path(path).read_bytes()
    if len(data) > 128_000:
        raise ValueError("configuration exceeds 128 KB")
    config = json.loads(data)
    if not isinstance(config, dict) or config.keys() - {
        "workers",
        "runtime",
        "max_pending",
        "acceptance",
    }:
        raise ValueError(
            "configuration accepts only workers, runtime, max_pending and acceptance"
        )
    workers = config.get("workers")
    if not isinstance(workers, list) or not 1 <= len(workers) <= 32:
        raise ValueError("configuration requires 1 to 32 workers")
    if any(not isinstance(w, dict) for w in workers):
        raise ValueError("worker configurations must be objects")
    if not isinstance(config.get("runtime", {}), dict):
        raise TypeError("runtime must be an object")
    RuntimeConfig(**config.get("runtime", {}))
    return config


async def create_pool(config: dict) -> WorkerPool:
    replicas = {}
    try:
        for i, raw in enumerate(config["workers"]):
            worker = dict(raw)
            kind = worker.pop("type", "process")
            name = worker.pop("name", f"worker-{i}")
            if not isinstance(name, str) or not name or name in replicas:
                raise ValueError("worker names must be nonempty and unique")
            if kind in {"process", "local"}:
                worker.setdefault("worker_id", name)
                definition = WorkerConfig(**worker)
                if kind == "process":
                    backend = await LocalProcessBackend.start(definition)
                else:
                    backend = LocalForgeBackend(definition)
                    replicas[name] = backend
                    await backend.start()
            elif kind == "remote":
                permitted = {
                    "url",
                    "token_env",
                    "max_concurrency",
                    "request_timeout",
                    "connect_timeout",
                    "poll_interval",
                    "allow_insecure_http",
                }
                if (
                    worker.keys() - permitted
                    or not {"url", "token_env"} <= worker.keys()
                ):
                    raise ValueError(
                        "remote worker requires url and token_env; unknown options are rejected"
                    )
                variable = worker.pop("token_env")
                if (
                    not isinstance(variable, str)
                    or not variable
                    or not os.environ.get(variable)
                ):
                    raise ValueError(
                        "remote worker token_env must name a populated environment variable"
                    )
                worker["allow_insecure"] = worker.pop("allow_insecure_http", False)
                backend = RemoteWorkerBackend(token=os.environ[variable], **worker)
                replicas[name] = backend
                await backend.health()
            else:
                raise ValueError("worker type must be process, local or remote")
            replicas[name] = backend
        return WorkerPool(replicas, max_pending=config.get("max_pending", 32))
    except BaseException:
        await asyncio.gather(
            *(b.close() for b in replicas.values()), return_exceptions=True
        )
        raise


def _save_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


async def worker_identities(pool: WorkerPool) -> dict:
    keys = (
        "model",
        "tokenizer",
        "backend",
        "max_model_length",
        "model_data_sha256",
        "model_config_sha256",
        "tokenizer_signature",
        "draft_model_data_sha256",
        "speculative",
        "native_tool_calls",
        "runner",
        "runner_version",
        "mlx_version",
        "checkpoint_format",
    )
    result = {}
    for name, backend in pool.backends.items():
        health = backend.health()
        if inspect.isawaitable(health):
            health = await health
        if not isinstance(health, dict):
            raise TypeError("worker health must be an object")
        for fingerprint in (
            "model_data_sha256",
            "model_config_sha256",
            "tokenizer_signature",
        ):
            value = health.get(fingerprint)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(
                    "worker lacks the artifact/tokenizer fingerprints required for resumable runs"
                )
        result[name] = {key: health.get(key) for key in keys}
    return result


async def execute(args) -> int:
    state = Path(args.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(state, 0o700)
    with AgentStore(state / "journal.sqlite") as store:
        os.chmod(state / "journal.sqlite", 0o600)
        if args.command == "inspect":
            result = {
                "run": store.run(args.run_id),
                "agents": [asdict(a) for a in store.agents(args.run_id)],
                "events": store.events(args.run_id),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "cancel":
            store.request_cancel(args.run_id)
            print(json.dumps({"run_id": args.run_id, "cancel_requested": True}))
            return 0
        config = load_config(args.config)
        run_id = (
            args.run_id if args.command == "resume" else args.run_id or uuid.uuid4().hex
        )
        # Names are also used for local artifact paths, so constrain them before I/O.
        if (
            not run_id
            or len(run_id) > 96
            or any(not (c.isascii() and (c.isalnum() or c in "_-")) for c in run_id)
        ):
            raise ValueError(
                "run_id must contain only ASCII letters, numbers, underscores or hyphens"
            )
        tools = WorkspaceTools(
            args.workspace,
            state / "artifacts" / run_id,
            allow_write=args.allow_write,
            allow_tests=args.allow_tests,
        )
        completion = (
            CompletionProfile(config["acceptance"], tools, store)
            if "acceptance" in config
            else None
        )
        manifest = {
            "workspace": str(tools.workspace),
            "tools": sorted(tools.mapping()),
            "acceptance_test_sha256": completion.test_fingerprints
            if completion
            else {},
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
        }
        manifest_path = state / f"{run_id}.manifest.json"
        if args.command == "resume":
            previous_manifest = json.loads(manifest_path.read_text())
            if any(
                previous_manifest.get(key) != value for key, value in manifest.items()
            ):
                raise ValueError(
                    "resume requires the original workspace, tool permissions and worker configuration"
                )
        elif manifest_path.exists():
            raise ValueError(
                "run_id already has a manifest; select a fresh ID or resume it"
            )
        pool = await create_pool(config)
        try:
            manifest["worker_identities"] = await worker_identities(pool)
            manifest["implementation_sha256"] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob("*.py")
            }
            if args.command == "resume":
                if previous_manifest != manifest:
                    raise ValueError(
                        "worker model/tokenizer or agent implementation changed since this run"
                    )
            else:
                _save_new(manifest_path, manifest)
        except BaseException:
            await pool.close()
            raise
        runtime = AgentRuntime(
            pool,
            store,
            tools.mapping(),
            RuntimeConfig(**config.get("runtime", {})),
            tool_descriptions=tools.tool_descriptions(),
            replay_safe_tools=tools.replay_safe_tools,
            completion_validator=completion,
        )
        done = asyncio.Event()

        async def progress():
            cursor = 0
            while not done.is_set():
                events = store.events(run_id, after_id=cursor)
                for event in events:
                    cursor = event["id"]
                    if not args.quiet:
                        print(
                            json.dumps(
                                {
                                    "event": event["kind"],
                                    "run_id": run_id,
                                    "agent_id": event["agent_id"],
                                }
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                try:
                    await asyncio.wait_for(done.wait(), 0.25)
                except TimeoutError:
                    pass

        watcher = asyncio.create_task(progress())
        try:
            if args.command == "resume":
                result = await runtime.resume(run_id)
            else:
                task = Path(args.task_file).read_text() if args.task_file else args.task
                result = await runtime.run(task, run_id=run_id)
            document = asdict(result)
            output = state / f"{run_id}.result.json"
            if not output.exists():
                _save_new(output, document)
            print(json.dumps(document, indent=2, ensure_ascii=False))
            return 0 if result.status == "completed" else 2
        finally:
            done.set()
            await watcher
            await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        default=".forge",
        help="private local journal and artifacts directory",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--workspace", default=".")
        command.add_argument("--run-id", required=name == "resume")
        command.add_argument("--allow-write", action="store_true")
        command.add_argument("--allow-tests", action="store_true")
        command.add_argument("--quiet", action="store_true")
        if name == "run":
            task = command.add_mutually_exclusive_group(required=True)
            task.add_argument("--task")
            task.add_argument("--task-file")
    for name in ("inspect", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(execute(args)))
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"forge-agents: {exc}\n")


if __name__ == "__main__":
    main()
