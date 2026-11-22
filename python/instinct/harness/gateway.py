"""The guest listener: the only part of Instinct meant to face the internet.

It binds to 127.0.0.1, and ``sharing.py`` publishes it through Tailscale
Funnel, which terminates HTTPS. It shares nothing with the owner's app
server: no cookies, no routes, no files. What it serves:

* ``/``: the guest web app (``web/guest.*``), which holds no secrets.
* ``/guest/session``: trades an invite for a browser session cookie.
* ``/guest/runs/...``: browser agent runs (see ``guests.BrowserRuns``).
* ``/guest/models/<driver|subagent>/...``: the worker job protocol, with the
  invite as a bearer token, for guests running their own Instinct app.

Nothing a guest sends is logged or written to disk here; invite usage
counts are the only state that changes.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
import urllib.parse
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from ..agents.media import MAX_MEDIA_REQUEST_BYTES
from ..agents.worker import WorkerService, _error
from ..common import strict_json
from .guests import (
    RELAYED_TOOLS,
    BrowserRuns,
    GroupModels,
    GuestModels,
    Invites,
)

WEB_ROOT = Path(__file__).with_name("web")
MAX_BODY = 1 << 20
SESSION_SECONDS = 12 * 3600
COOKIE = "instinct_guest"
FILES = {
    "/": ("guest.html", "text/html; charset=utf-8"),
    "/guest.js": ("guest.js", "text/javascript; charset=utf-8"),
    "/guest.css": ("guest.css", "text/css; charset=utf-8"),
}
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: "
    "blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
    "form-action 'none'"
)


class Gateway:
    """State shared by the guest listener's request threads."""

    max_guest_runs = 24

    def __init__(
        self,
        invites: Invites,
        service: Callable[[], Any],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.invites = invites
        self._service = service
        self.loop = loop
        self.runs = BrowserRuns(self)
        self._sessions: dict[str, tuple[str, float]] = {}
        self._models: dict[str, GuestModels] = {}
        self._jobs: dict[tuple[str, str], WorkerService] = {}
        self._lock = threading.RLock()
        self._failures: list[float] = []

    # --- the owner's side ----------------------------------------------------

    def pool(self) -> Any:
        service = self._service()
        if service is None or service.backend_status != "ready" or service.demo:
            return None
        return service.backend

    def config(self) -> dict:
        return self._service().config

    def groups(self) -> dict[str, str]:
        """Model choices guests have: ``driver`` and ``subagent``, mapped to
        the pool groups the owner's runtime uses for them."""
        runtime = self.config().get("runtime", {})
        present = {worker.get("group") for worker in self.config().get("workers", [])}
        choices = {}
        for alias, key in (("driver", "driver_group"), ("subagent", "subagent_group")):
            group = runtime.get(key)
            if group and group in present:
                choices[alias] = group
        return choices

    def model_labels(self) -> list[dict[str, str]]:
        from .service import model_label

        labels = []
        workers = self.config().get("workers", [])
        for alias, group in self.groups().items():
            worker = next((w for w in workers if w.get("group") == group), {})
            labels.append({"id": alias, "label": model_label(worker.get("model", ""))})
        return labels

    def models(self, invite_id: str) -> GuestModels:
        with self._lock:
            if invite_id not in self._models:
                self._models[invite_id] = GuestModels(self, invite_id)
            return self._models[invite_id]

    def jobs(self, invite: dict, alias: str) -> WorkerService:
        groups = self.groups()
        if alias not in groups:
            raise KeyError("no such model")
        key = (invite["id"], alias)
        with self._lock:
            service = self._jobs.get(key)
            if service is None:
                service = WorkerService(
                    GroupModels(self.models(invite["id"]), groups[alias]),
                    max_pending=invite["limits"]["concurrent"],
                    max_timeout=900.0,
                    max_history=64,
                    worker_id=f"guest-{alias}",
                )
                service.start()
                self._jobs[key] = service
            return service

    def revoke(self, invite_id: str) -> None:
        """Stop everything an invite has running, and sign its browsers out."""
        self.runs.cancel_invite(invite_id)
        with self._lock:
            for key in [k for k in self._jobs if k[0] == invite_id]:
                self._jobs.pop(key).close()
            for token in [t for t, (i, _) in self._sessions.items() if i == invite_id]:
                del self._sessions[token]

    def status(self) -> dict[str, Any]:
        with self._lock:
            sessions = len(self._sessions)
            jobs = sum(
                service.health().get("active", 0) for service in self._jobs.values()
            )
        return {"browser_sessions": sessions, "runs": self.runs.active(), "jobs": jobs}

    def close(self) -> None:
        with self._lock:
            services, self._jobs = list(self._jobs.values()), {}
        for service in services:
            service.close()
        self.invites.flush()

    # --- sign-in -------------------------------------------------------------

    def throttled(self) -> bool:
        """Back off after repeated bad invites (they are 256-bit random, so
        this only keeps noise down)."""
        now = time.monotonic()
        with self._lock:
            self._failures = [t for t in self._failures if now - t < 60]
            return len(self._failures) >= 30

    def failed(self) -> None:
        with self._lock:
            self._failures.append(time.monotonic())

    def open_session(self, invite: dict) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            now = time.time()
            self._sessions = {
                t: (i, at)
                for t, (i, at) in self._sessions.items()
                if now - at < SESSION_SECONDS
            }
            self._sessions[token] = (invite["id"], now)
        return token

    def session(self, token: str | None) -> dict | None:
        if not token:
            return None
        with self._lock:
            entry = self._sessions.get(token)
            if entry is None:
                return None
            invite_id, opened = entry
            if time.time() - opened > SESSION_SECONDS:
                del self._sessions[token]
                return None
        return self.invites.get(invite_id)

    def end_session(self, token: str | None) -> None:
        with self._lock:
            self._sessions.pop(token or "", None)


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, gateway: Gateway) -> None:
        self.gateway = gateway
        super().__init__(("127.0.0.1", port), GatewayHandler)


class GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        return None  # Guests' requests never enter a log.

    # --- plumbing ------------------------------------------------------------

    @property
    def gateway(self) -> Gateway:
        return self.server.gateway

    def _send(self, status: int, body: bytes, mime: str, headers=()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any, headers=()) -> None:
        self._send(
            status,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
            "application/json",
            headers,
        )

    def _fail(self, exc: BaseException) -> None:
        if isinstance(exc, PermissionError):
            return self._json(
                401, {"error": {"code": "authentication", "message": str(exc)}}
            )
        if isinstance(exc, KeyError):
            return self._json(
                404, {"error": {"code": "not_found", "message": "not found"}}
            )
        status, result = _error(exc)
        self._json(status, result)

    def _body(self, limit: int = MAX_BODY) -> dict:
        if (
            self.headers.get("Content-Type", "").split(";")[0].strip()
            != "application/json"
        ):
            raise ValueError("requests must be application/json")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked requests are not supported")
        size = int(self.headers.get("Content-Length", "0"))
        if not 1 <= size <= limit:
            self.close_connection = True
            raise ValueError(f"request body must contain 1 to {limit} bytes")
        self.connection.settimeout(30)
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("incomplete request body")
        value = strict_json(data)
        if not isinstance(value, dict):
            raise ValueError("request must be a JSON object")
        return value

    def _cookie(self) -> str | None:
        try:
            cookies = SimpleCookie()
            cookies.load(self.headers.get("Cookie", ""))
            morsel = cookies.get(COOKIE)
            return morsel.value if morsel else None
        except Exception:  # noqa: BLE001 - a malformed cookie is no session.
            return None

    def _same_origin(self) -> bool:
        """State-changing browser requests must come from this page."""
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"https://{host}", f"http://{host}"}:
            return False
        return self.headers.get("Sec-Fetch-Site", "same-origin") in {
            "same-origin",
            "none",
        }

    def _browser_invite(self) -> dict:
        invite = self.gateway.session(self._cookie())
        if invite is None:
            raise PermissionError("sign in with your invite link")
        return invite

    def _bearer_invite(self) -> dict:
        header = self.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else None
        if self.gateway.throttled():
            raise PermissionError("too many failed sign-ins; wait a minute")
        invite = self.gateway.invites.check(token)
        if invite is None:
            self.gateway.failed()
            raise PermissionError("invalid, expired or revoked invite")
        return invite

    # --- routes --------------------------------------------------------------

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        path = url.path
        try:
            if path in FILES:
                name, mime = FILES[path]
                try:
                    return self._send(200, (WEB_ROOT / name).read_bytes(), mime)
                except OSError:
                    return self._json(
                        503,
                        {
                            "error": {
                                "code": "unavailable",
                                "message": "the guest app isn't installed",
                            }
                        },
                    )
            parts = path.strip("/").split("/")
            if path == "/guest/me":
                invite = self._browser_invite()
                return self._json(200, self._me(invite))
            if (
                len(parts) == 4
                and parts[:2] == ["guest", "runs"]
                and parts[3] == "events"
            ):
                invite = self._browser_invite()
                query = urllib.parse.parse_qs(url.query)
                result = self.gateway.runs.events(
                    invite["id"],
                    parts[2],
                    after=max(0, int(query.get("after", ["0"])[0])),
                    wait=float(query.get("wait", ["20"])[0]),
                )
                return self._json(200, result)
            if len(parts) >= 4 and parts[:2] == ["guest", "models"]:
                return self._jobs_route("GET", parts[2], parts[3:], url.query)
            self._json(404, {"error": {"code": "not_found", "message": "not found"}})
        except Exception as exc:  # noqa: BLE001 - one request's failure.
            self._fail(exc)

    def do_POST(self) -> None:
        path = urllib.parse.urlsplit(self.path).path
        parts = path.strip("/").split("/")
        try:
            if len(parts) >= 4 and parts[:2] == ["guest", "models"]:
                return self._jobs_route("POST", parts[2], parts[3:], "")
            if not self._same_origin():
                return self._json(
                    403, {"error": {"code": "origin", "message": "unexpected origin"}}
                )
            if path == "/guest/session":
                return self._sign_in()
            if path == "/guest/session/end":
                self.gateway.end_session(self._cookie())
                return self._json(
                    200,
                    {"signed_out": True},
                    [
                        (
                            "Set-Cookie",
                            f"{COOKIE}=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/",
                        )
                    ],
                )
            invite = self._browser_invite()
            if path == "/guest/runs":
                body = self._body(MAX_MEDIA_REQUEST_BYTES)
                return self._json(
                    200,
                    asyncio.run_coroutine_threadsafe(
                        self.gateway.runs.start(invite, body), self.gateway.loop
                    ).result(30),
                )
            if (
                len(parts) == 5
                and parts[:2] == ["guest", "runs"]
                and parts[3] == "tools"
            ):
                self.gateway.runs.answer(
                    invite["id"], parts[2], parts[4], self._body(MAX_BODY)
                )
                return self._json(200, {"ok": True})
            if (
                len(parts) == 4
                and parts[:2] == ["guest", "runs"]
                and parts[3] == "cancel"
            ):
                self.gateway.runs.cancel(invite["id"], parts[2])
                return self._json(200, {"ok": True})
            self._json(404, {"error": {"code": "not_found", "message": "not found"}})
        except Exception as exc:  # noqa: BLE001 - one request's failure.
            self._fail(exc)

    def do_DELETE(self) -> None:
        parts = urllib.parse.urlsplit(self.path).path.strip("/").split("/")
        try:
            if len(parts) >= 4 and parts[:2] == ["guest", "models"]:
                return self._jobs_route("DELETE", parts[2], parts[3:], "")
            self._json(404, {"error": {"code": "not_found", "message": "not found"}})
        except Exception as exc:  # noqa: BLE001 - one request's failure.
            self._fail(exc)

    def _sign_in(self) -> None:
        if self.gateway.throttled():
            raise PermissionError("too many failed sign-ins; wait a minute")
        body = self._body(4096)
        invite = self.gateway.invites.check(body.get("invite"))
        if set(body) != {"invite"} or invite is None:
            self.gateway.failed()
            raise PermissionError("that invite is invalid, expired or revoked")
        token = self.gateway.open_session(invite)
        secure = (
            ""
            if self.headers.get("Host", "").startswith(("127.0.0.1", "localhost"))
            else "; Secure"
        )
        self._json(
            200,
            self._me(invite),
            [
                (
                    "Set-Cookie",
                    f"{COOKIE}={token}; HttpOnly; SameSite=Strict; Path=/; "
                    f"Max-Age={SESSION_SECONDS}{secure}",
                )
            ],
        )

    def _me(self, invite: dict) -> dict:
        return {
            "name": invite["name"],
            "expires": invite["expires"],
            "limits": invite["limits"],
            "tokens_today": invite["tokens_today"],
            "models": self.gateway.model_labels(),
            "ready": self.gateway.pool() is not None,
            "tools": list(RELAYED_TOOLS),
        }

    def _jobs_route(self, method: str, alias: str, rest: list[str], query: str) -> None:
        """The worker job protocol for one model, for guests' own apps."""
        invite = self._bearer_invite()
        service = self.gateway.jobs(invite, alias)
        if method == "GET" and rest == ["health"]:
            result = service.health()
        elif method == "POST" and rest == ["v1", "jobs"]:
            result = service.submit(self._body(MAX_MEDIA_REQUEST_BYTES))
        elif len(rest) == 3 and rest[:2] == ["v1", "jobs"]:
            request_id = urllib.parse.unquote(rest[2])
            if method == "GET":
                options = urllib.parse.parse_qs(query)
                result = service.get(
                    request_id,
                    wait=float(options.get("wait", ["0"])[0]),
                    since=int(options.get("since", ["0"])[0]),
                )
            elif method == "DELETE":
                result = service.cancel(request_id)
            else:
                raise KeyError(method)
        else:
            raise KeyError(method)
        self._json(200, result)
