"""Launch Forge Studio without changing the inference backend or worker protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote

from ..agents.cli import load_config
from .server import StudioHTTPServer
from .service import StudioService


async def serve(args):
    skills = None
    if args.skills_dir:
        from .extensions import SkillRegistry

        skills = SkillRegistry([Path(p).resolve(strict=True) for p in args.skills_dir])
    clients = []
    if args.mcp_config:
        from .extensions import MCPServerConfig, MCPStdioClient

        path = Path(args.mcp_config).resolve(strict=True)
        if path.stat().st_size > 128_000:
            raise ValueError("MCP configuration exceeds 128 KB")
        definitions = json.loads(path.read_text())
        if not isinstance(definitions, list) or len(definitions) > 8:
            raise ValueError("MCP configuration must be a list of up to eight servers")
        names = set()
        for raw in definitions:
            definition = dict(raw)
            definition["command"] = tuple(definition["command"])
            definition["allowed_tools"] = tuple(definition.get("allowed_tools", []))
            definition["cwd"] = Path(definition.get("cwd", args.workspace)).resolve(
                strict=True
            )
            selected = MCPServerConfig(**definition)
            if selected.name in names:
                raise ValueError("MCP server names must be unique")
            names.add(selected.name)
            clients.append(MCPStdioClient(selected))
    config = load_config(args.config) if args.config else {}
    service = StudioService(
        Path(args.workspace),
        Path(args.state_dir),
        config=config,
        demo=args.demo,
        skills=skills,
        mcp_clients=clients,
    )
    server = StudioHTTPServer(
        ("127.0.0.1", args.port), service, asyncio.get_running_loop()
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"Forge Studio is serving at {server.origin}. Press Ctrl-C to stop.", flush=True
    )
    if not args.no_open:
        webbrowser.open(server.origin + "/#token=" + quote(server.bootstrap_token))
    try:
        await asyncio.Event().wait()
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
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
