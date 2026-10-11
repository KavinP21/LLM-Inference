"""Launch Forge Studio without changing the inference backend or worker protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote

from ..agents.cli import load_config
from ..agents.protocol import _unique_object
from .extensions import MCPServerConfig, MCPStdioClient, SkillRegistry
from .server import StudioHTTPServer
from .service import StudioService


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
    service = StudioService(
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
        server = StudioHTTPServer(
            ("127.0.0.1", args.port), service, asyncio.get_running_loop()
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        thread_started = True
        print(
            f"Forge Studio is serving at {server.origin}. Press Ctrl-C to stop.",
            flush=True,
        )
        if not args.no_open:
            webbrowser.open(server.origin + "/#token=" + quote(server.bootstrap_token))
        await asyncio.Event().wait()
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
    parser = argparse.ArgumentParser(description="Forge Studio local agent workspace")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--state-dir", default=".forge/studio")
    parser.add_argument("--config", help="Existing Forge agent worker configuration")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Deterministic harness exercise; no model inference",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open browser; existing authorized cookies remain valid only for this server process",
    )
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
        parser.error(
            "Live Studio requires --config; use --demo to exercise the harness"
        )
    if not 0 <= args.port <= 65535:
        parser.error("port must be between zero and 65535")
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
