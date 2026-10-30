"""Launch Instinct without changing the inference backend or worker protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import ssl
import stat
import threading
from pathlib import Path

from ..agents.cli import load_config
from ..common import strict_json
from .api_keys import ApiKeys
from .extensions import MCPServerConfig, MCPStdioClient, SkillRegistry
from .projects import Projects, state_folder
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
        definitions = strict_json(data)
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


def remote_settings(args) -> dict | None:
    """Where and how to serve other devices, from flags or the app's settings."""
    saved = {}
    network_file = getattr(args, "network_file", None)
    if network_file is not None and network_file.is_file():
        saved = json.loads(network_file.read_text())
    host = getattr(args, "remote_host", None) or (
        "0.0.0.0" if saved.get("enabled") else None
    )
    if not host:
        return None
    cert = getattr(args, "tls_cert", None) or (
        Path(saved["tls_cert"]) if saved.get("tls_cert") else None
    )
    key = getattr(args, "tls_key", None) or (
        Path(saved["tls_key"]) if saved.get("tls_key") else None
    )
    if bool(cert) != bool(key):
        raise ValueError("TLS needs both a certificate and its key")
    if cert is None and not (
        getattr(args, "allow_http", False) or saved.get("allow_http")
    ):
        raise ValueError(
            "serving other devices needs --tls-cert and --tls-key, or --allow-http on a network that already encrypts (Tailscale)"
        )
    context = None
    if cert is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
    port = getattr(args, "remote_port", None) or saved.get("port") or 8765
    return {"host": host, "port": int(port), "tls": context}


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
    # The app keeps keys, network settings, the project list and each
    # project's state under one folder; the command line keeps them in the
    # workspace's state folder.
    home = getattr(args, "home", None)
    keys_file = args.keys_file or (home / "api-keys.json" if home else None)
    network_file = getattr(args, "network_file", None) or (
        home / "network.json" if home else None
    )
    projects = Projects((home or state_dir) / "projects.json")

    def build(folder: Path, folder_state: Path) -> AppService:
        return AppService(
            folder,
            folder_state,
            config=config,
            demo=args.demo,
            skills=skills,
            mcp_clients=clients,
            keys_file=keys_file,
            network_file=network_file,
            projects=projects,
        )

    current = {"service": build(workspace, state_dir)}
    servers: list[AppServer] = []

    async def switch(folder: Path) -> None:
        """Open another project in place: same models, same servers."""
        previous = current["service"]
        fresh = build(
            folder, state_folder(home, folder) if home else folder / ".instinct"
        )
        fresh.adopt(previous)
        fresh.switch_project = switch
        fresh.remote_url = previous.remote_url
        for running in servers:
            running.service = fresh
        current["service"] = fresh
        await previous.close()

    service = current["service"]
    service.switch_project = switch
    server = None
    thread = None
    thread_started = False
    remote_server = None
    try:
        server = AppServer(
            ("127.0.0.1", args.port),
            service,
            asyncio.get_running_loop(),
            # The Mac app makes the token itself, so it is never printed.
            token=os.environ.get(APP_TOKEN),
        )
        servers.append(server)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        thread_started = True
        remote = remote_settings(args)
        if remote is not None:
            remote_server = AppServer(
                (remote["host"], remote["port"]),
                service,
                asyncio.get_running_loop(),
                tls=remote["tls"],
                remote=True,
            )
            servers.append(remote_server)
            service.remote_url = (
                f"{remote_server.scheme}://{{host}}:{remote_server.server_port}"
            )
            threading.Thread(target=remote_server.serve_forever, daemon=True).start()
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
        if remote_server is not None:
            await asyncio.to_thread(remote_server.shutdown)
            remote_server.server_close()
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
            await current["service"].close()


def main():
    parser = argparse.ArgumentParser(description="Instinct local agent workspace")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--state-dir", default=".instinct")
    parser.add_argument(
        "--keys-file",
        type=Path,
        help="Where API keys for /v1 are kept (default: in the state folder)",
    )
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
    remote = parser.add_argument_group("other devices")
    remote.add_argument(
        "--remote-host", help="Also serve other devices on this address, e.g. 0.0.0.0"
    )
    remote.add_argument(
        "--remote-port", type=int, help="Port for other devices (default 8765)"
    )
    remote.add_argument(
        "--tls-cert",
        type=Path,
        help="Certificate for other devices (e.g. from tailscale cert)",
    )
    remote.add_argument("--tls-key", type=Path, help="Private key for --tls-cert")
    remote.add_argument(
        "--allow-http",
        action="store_true",
        help="Serve other devices without TLS, on a network that already encrypts (Tailscale)",
    )
    remote.add_argument("--network-file", type=Path, help=argparse.SUPPRESS)
    # The Mac app's Application Support folder.
    parser.add_argument("--home", type=Path, help=argparse.SUPPRESS)
    remote.add_argument(
        "--create-key", metavar="NAME", help="Make a key, print it once and exit"
    )
    remote.add_argument(
        "--app-key", action="store_true", help="The new key also opens the app"
    )
    args = parser.parse_args()
    if args.create_key:
        state_dir = Path(args.state_dir)
        if not state_dir.is_absolute():
            state_dir = Path(args.workspace).resolve() / state_dir
        keys = ApiKeys(args.keys_file or state_dir / "api-keys.json")
        print(keys.create(args.create_key, "app" if args.app_key else "api")["key"])
        return
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
