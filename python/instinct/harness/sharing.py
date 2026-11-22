"""Turning guest sharing on and off: the guest listener plus Tailscale Funnel.

Sharing is off until the owner turns it on in Settings. Then the guest
gateway listens on 127.0.0.1:<port> (8766 by default), and Tailscale Funnel
publishes it at ``https://<this-mac>.<tailnet>.ts.net`` with a real
certificate. Nothing else on this Mac becomes reachable: Funnel forwards only
that one local port, and the owner's app server stays on its own loopback
port.

Funnel commands (Tailscale 1.52 or later):
``tailscale funnel --bg --https=443 localhost:<port>`` to publish and
``tailscale funnel --https=443 off`` to stop. The first time, Tailscale asks
in the browser to enable HTTPS certificates and Funnel for the tailnet.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from .gateway import Gateway, GatewayServer
from .guests import Invites

DEFAULT_PORT = 8766
TAILSCALE_PATHS = (
    "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
    "/usr/local/bin/tailscale",
    "/opt/homebrew/bin/tailscale",
)


def tailscale_binary() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    return next((path for path in TAILSCALE_PATHS if os.access(path, os.X_OK)), None)


def _run(argv: list[str], timeout: float = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
    )


class Sharing:
    """Owns the guest listener and the Funnel that publishes it."""

    def __init__(
        self,
        home: Path,
        service: Callable[[], Any],
        loop: asyncio.AbstractEventLoop,
        *,
        funnel: bool = True,
    ) -> None:
        self.settings_path = home / "sharing.json"
        self.invites = Invites(home / "guest-invites.json")
        self.gateway = Gateway(self.invites, service, loop)
        self.funnel = funnel  # Off in tests and for loopback-only use.
        self.server: GatewayServer | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    # --- settings -----------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        try:
            saved = json.loads(self.settings_path.read_text())
        except (OSError, ValueError):
            saved = {}
        port = saved.get("port", DEFAULT_PORT)
        return {
            "enabled": saved.get("enabled") is True,
            "port": port
            if type(port) is int and 1024 <= port <= 65535
            else DEFAULT_PORT,
        }

    def _save(self, settings: dict[str, Any]) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(settings, indent=2))
        os.chmod(self.settings_path, 0o600)

    # --- on and off ---------------------------------------------------------

    def start(self) -> None:
        """Listen on loopback and publish through Funnel, if sharing is on."""
        settings = self.settings()
        if not settings["enabled"]:
            return
        with self._lock:
            if self.server is None:
                self.server = GatewayServer(settings["port"], self.gateway)
                threading.Thread(
                    target=self.server.serve_forever,
                    name="instinct-guests",
                    daemon=True,
                ).start()
        self.error = None
        if self.funnel:
            self.error = self._funnel(f"localhost:{self.server.server_port}")

    def stop(self) -> None:
        with self._lock:
            server, self.server = self.server, None
        if self.funnel and server is not None:
            self._funnel(None)
        if server is not None:
            server.shutdown()
            server.server_close()
        self.gateway.close()

    def update(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or payload.keys() - {"enabled", "port"}:
            raise ValueError("sharing takes enabled and port")
        settings = {**self.settings(), **payload}
        if type(settings["enabled"]) is not bool:
            raise ValueError("enabled is true or false")
        if type(settings["port"]) is not int or not 1024 <= settings["port"] <= 65535:
            raise ValueError("the port must be between 1024 and 65535")
        self._save(settings)
        self.stop()
        self.start()
        return self.status()

    def _funnel(self, target: str | None) -> str | None:
        binary = tailscale_binary()
        if binary is None:
            return (
                "Tailscale isn't installed. Install it from tailscale.com/download, "
                "sign in, then turn sharing on again."
            )
        argv = [binary, "funnel", "--https=443"]
        argv += ["off"] if target is None else ["--bg", "--yes", target]
        try:
            result = _run(argv)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"Tailscale didn't respond ({type(exc).__name__})"
        if result.returncode != 0:
            # Usually a link to enable HTTPS certificates or Funnel.
            text = (result.stderr or result.stdout).strip()
            return text[-600:] or f"tailscale funnel exited with {result.returncode}"
        return None

    # --- what the owner sees ------------------------------------------------

    def url(self) -> str | None:
        binary = tailscale_binary()
        if binary is None or not self.funnel:
            return None
        try:
            result = _run([binary, "status", "--json"], timeout=5)
            name = json.loads(result.stdout)["Self"]["DNSName"].rstrip(".")
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
            return None
        return f"https://{name}" if name else None

    def status(self) -> dict[str, Any]:
        settings = self.settings()
        return {
            **settings,
            "listening": self.server is not None,
            "local_url": (
                f"http://127.0.0.1:{self.server.server_port}" if self.server else None
            ),
            "url": self.url() if self.server is not None else None,
            "tailscale": tailscale_binary() is not None,
            "error": self.error,
            "activity": self.gateway.status(),
            "invites": self.invites.list(),
        }

    def dispatch(self, method: str, parts: list[str], payload: dict) -> dict:
        """``/api/guests`` routes for the owner's app (loopback only)."""
        if method == "GET" and parts == ["api", "guests"]:
            return self.status()
        if method == "POST" and parts == ["api", "guests", "sharing"]:
            return self.update(payload)
        if method == "POST" and parts == ["api", "guests", "invites"]:
            if not isinstance(payload, dict) or payload.keys() - {
                "name",
                "hours",
                "limits",
            }:
                raise ValueError("an invite takes name, hours and limits")
            made = self.invites.create(
                payload.get("name"),
                hours=payload.get("hours", 24),
                limits=payload.get("limits"),
            )
            url = self.url() or (
                f"http://127.0.0.1:{self.server.server_port}" if self.server else None
            )
            # The link carries the invite in its fragment, which browsers
            # never send to a server, so it stays out of any proxy log.
            made["link"] = f"{url}/#invite={made['token']}" if url else None
            return made
        if (
            method == "POST"
            and len(parts) == 5
            and parts[:3] == ["api", "guests", "invites"]
            and parts[4] in {"revoke", "delete"}
        ):
            invite_id = parts[3]
            if parts[4] == "delete":
                self.invites.delete(invite_id)
            else:
                self.invites.revoke(invite_id)
            self.gateway.revoke(invite_id)
            return self.status()
        raise ValueError("unknown guests route")
