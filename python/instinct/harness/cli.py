"""Launch Instinct without changing the inference backend or worker protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import stat
import threading
from pathlib import Path

from ..agents.cli import load_config
from ..agents.protocol import _unique_object
from .extensions import MCPServerConfig, MCPStdioClient, SkillRegistry
from .server import AppServer
from .service import AppService


def load_mcp_config(path: str | Path, workspace: Path) -> list[MCPServerConfig]:
    """Read operator configuration; relative server cwd is workspace-relative."""
    workspace = workspace.resolve(strict=True)
    selected_path = Path(path).resolve(strict=True)
    descriptor = os.open(
        selected_path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("MCP configuration must be a regular file")
        if info.st_size > 128_000:
            raise ValueError("MCP configuration exceeds 128 KB")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(128_001)
        if len(data) > 128_000:
            raise ValueError("MCP configuration exceeds 128 KB")
    finally:
        os.close(descriptor)
    try:
        definitions = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite JSON")
            ),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("MCP configuration contains invalid JSON") from exc
    if not isinstance(definitions, list) or len(definitions) > 8:
        raise ValueError("MCP configuration must be a list of up to eight servers")
    fields = set(MCPServerConfig.__dataclass_fields__)
    configs = []
    names = set()
    for raw in definitions:
        if not isinstance(raw, dict) or raw.keys() - fields:
            raise ValueError("MCP server entries must be objects with supported fields")
        if not {"name", "command"} <= raw.keys():
            raise ValueError("MCP server entries require name and command")
        if not isinstance(raw["name"], str):
            raise ValueError("MCP server name must be a string identifier")
        if not isinstance(raw["command"], list) or not isinstance(
            raw.get("allowed_tools", []), list
        ):
            raise ValueError("MCP command and allowed_tools must be JSON arrays")
        definition = dict(raw)
        cwd = definition.get("cwd")
        if "cwd" in definition and (not isinstance(cwd, str) or not cwd):
            raise ValueError("MCP cwd must be a nonempty directory path")
        directory = Path(cwd) if cwd is not None else workspace
        definition["cwd"] = (
            directory if directory.is_absolute() else workspace / directory
        )
        selected = MCPServerConfig(**definition)
        if selected.name in names:
            raise ValueError("MCP server names must be unique")
        names.add(selected.name)
        configs.append(selected)
    return configs


APP_TOKEN = "INSTINCT_APP_TOKEN"


async def serve(args):
    # The deterministic demo must never launch operator external programs.
    if args.demo and args.mcp_config:
        raise ValueError("--demo cannot be combined with --mcp-config")
    workspace = Path(args.workspace).resolve(strict=True)
    state_dir = Path(args.state_dir)
    if not state_dir.is_absolute():
        state_dir = workspace / state_dir
    skills = (
        SkillRegistry(
            [
                path if path.is_absolute() else workspace / path
                for path in map(Path, args.skills_dir)
            ]
        )
        if args.skills_dir
        else None
    )
    clients = (
        [
            MCPStdioClient(config)
            for config in load_mcp_config(args.mcp_config, workspace)
        ]
        if args.mcp_config
        else []
    )
    config = load_config(args.config) if args.config else {}
    service = AppService(
        workspace,
        state_dir,
        config=config,
        demo=args.demo,
        skills=skills,
        mcp_clients=clients,
    )
    server = None
    thread = None
    thread_started = False
    try:
        server = AppServer(
            ("127.0.0.1", args.port),
            service,
            asyncio.get_running_loop(),
            # The Mac app makes the token itself, so it is never printed.
            token=os.environ.get(APP_TOKEN),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        thread_started = True
        if not args.demo and config.get("workers"):
            # Warm the model in the background while the operator starts typing.
            # A submitted chat is durably queued if loading is still in progress.
            await service.start_backend()
        loop = asyncio.get_running_loop()
        # SIGTERM gets the same cleanup as Ctrl-C, so model workers stop too.
        stopped = asyncio.Event()
        loop.add_signal_handler(signal.SIGTERM, stopped.set)
        if args.app:
            print(
                json.dumps({"event": "ready", "port": server.server_port}), flush=True
            )

            def watch_app() -> None:
                # The app holds our stdin open; EOF means it quit or crashed.
                while os.read(0, 4096):
                    pass
                loop.call_soon_threadsafe(stopped.set)

            threading.Thread(target=watch_app, daemon=True).start()
        else:
            print(f"Instinct is serving at {server.origin}.", flush=True)
        await stopped.wait()
    finally:
        try:
            if server is not None:
                try:
                    if thread_started:
                        await asyncio.to_thread(server.shutdown)
                finally:
                    server.server_close()
            if thread_started:
                await asyncio.to_thread(thread.join, 1.0)
        finally:
            await service.close()


def main():
    parser = argparse.ArgumentParser(description="Instinct local agent workspace")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--state-dir", default=".instinct")
    parser.add_argument("--config", help="Existing Instinct agent worker configuration")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Deterministic harness exercise; no model inference",
    )
    parser.add_argument("--port", type=int, default=8765)
    # Used by the Mac app: token from $INSTINCT_APP_TOKEN, a JSON ready line
    # with the port, and exit when the app closes our stdin.
    parser.add_argument("--app", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--skills-dir",
        action="append",
        default=[],
        help="Operator-selected skill directory; repeatable",
    )
    parser.add_argument(
        "--mcp-config",
        help="Local JSON list of explicit MCP server commands/tool grants",
    )
    args = parser.parse_args()
    if args.demo and args.config:
        parser.error("Choose either --demo or a live --config")
    if args.demo and args.mcp_config:
        parser.error("--demo cannot be combined with --mcp-config")
    if not args.demo and not args.config:
        parser.error("--config is required unless --demo is given")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between zero and 65535")
    if args.app and len(os.environ.get(APP_TOKEN, "")) < 32:
        parser.error(f"--app needs a token of at least 32 characters in ${APP_TOKEN}")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
