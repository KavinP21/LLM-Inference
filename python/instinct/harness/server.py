"""Loopback-only app server, with exact-origin and cookie authentication."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hmac
import json
import secrets
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ..agents.media import MAX_MEDIA_REQUEST_BYTES

WEB_ROOT = Path(__file__).with_name("web")
MAX_BODY = 128_000


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service, loop, *, token: str | None = None):
        if address[0] != "127.0.0.1":
            raise ValueError("the server only binds to 127.0.0.1")
        self.service, self.loop = service, loop
        self.bootstrap_token = token or secrets.token_urlsafe(32)
        self.session_token = secrets.token_urlsafe(32)
        super().__init__(address, AppHandler)
        self.origin = f"http://127.0.0.1:{self.server_port}"


class AppHandler(BaseHTTPRequestHandler):
    server: AppServer

    def log_message(self, *args):
        # Browser credentials and operator briefs never enter access logs.
        return None

    def _send(self, status: int, body: bytes, mime="application/json", headers=()):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        for key, value in headers:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value, headers=()):
        self._send(
            status,
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
            headers=headers,
        )

    def _origin_ok(self):
        return (
            self.headers.get("Host") == urlsplit(self.server.origin).netloc
            and self.headers.get("Origin", self.server.origin) == self.server.origin
            and self.headers.get("Sec-Fetch-Site", "same-origin")
            in {"same-origin", "none"}
        )

    def _authenticated(self):
        try:
            cookies = SimpleCookie()
            cookies.load(self.headers.get("Cookie", ""))
            value = cookies.get("instinct_session")
            return value is not None and hmac.compare_digest(
                value.value, self.server.session_token
            )
        except Exception:
            return False

    def _body(self, *, limit=MAX_BODY):
        if (
            self.headers.get("Content-Type", "").split(";")[0].strip()
            != "application/json"
        ):
            raise ValueError("requests must be application/json")
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked requests are not supported")
        size = int(self.headers.get("Content-Length", "0"))
        if not 1 <= size <= limit:
            if limit < size <= MAX_MEDIA_REQUEST_BYTES:
                # Read what an image send could carry, so the client gets the
                # 400 rather than a reset from closing mid-upload.
                self.connection.settimeout(1)
                try:
                    self.rfile.read(size)
                except OSError:
                    pass
            raise ValueError(f"request body must contain 1 to {limit} bytes")
        self.connection.settimeout(5)
        data = self.rfile.read(size)
        if len(data) != size:
            raise ValueError("incomplete request body")

        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON field")
                value[key] = item
            return value

        value = json.loads(
            data,
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite JSON")
            ),
        )
        json.dumps(value, allow_nan=False)
        if not isinstance(value, dict):
            raise ValueError("request must be a JSON object")
        return value

    def do_GET(self):
        if not self._origin_ok():
            return self._json(403, {"error": "unexpected origin or host"})
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            if not self._authenticated():
                return self._json(
                    401,
                    {"error": "Open this window from the Instinct app"},
                )
            if path == "/api/events":
                return self._events()
            parts = path.strip("/").split("/")
            if (
                len(parts) == 4
                and parts[:2] == ["api", "terminals"]
                and parts[3] == "stream"
            ):
                return self._terminal_stream(parts[2])
            return self._dispatch("GET", path, {})
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.css": ("app.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/vendor/xterm.js": ("vendor/xterm.js", "text/javascript; charset=utf-8"),
            "/vendor/addon-fit.js": (
                "vendor/addon-fit.js",
                "text/javascript; charset=utf-8",
            ),
            "/vendor/xterm.css": ("vendor/xterm.css", "text/css; charset=utf-8"),
        }
        if path not in files:
            return self._json(404, {"error": "not found"})
        filename, mime = files[path]
        try:
            self._send(200, (WEB_ROOT / filename).read_bytes(), mime)
        except OSError:
            self._json(503, {"error": "the app's UI files are missing"})

    def do_POST(self):
        if not self._origin_ok():
            return self._json(403, {"error": "unexpected origin or host"})
        path = urlsplit(self.path).path
        # Authorize before reading potentially large image uploads. Auth itself
        # stays on the small-body limit and uses its existing bootstrap check.
        if path != "/api/auth" and not self._authenticated():
            self.close_connection = True
            return self._json(401, {"error": "not signed in"})
        parts = path.strip("/").split("/")
        image_send = path == "/api/chats" or (
            len(parts) == 4 and parts[:2] == ["api", "chats"] and parts[3] == "message"
        )
        try:
            body = self._body(limit=MAX_MEDIA_REQUEST_BYTES if image_send else MAX_BODY)
        except (ValueError, TypeError, OSError, RecursionError) as exc:
            return self._json(400, {"error": str(exc)})
        if path == "/api/auth":
            token = body.get("token")
            if (
                set(body) != {"token"}
                or not isinstance(token, str)
                or not token.isascii()
                or not hmac.compare_digest(token, self.server.bootstrap_token)
            ):
                return self._json(401, {"error": "invalid session token"})
            return self._json(
                200,
                {"authenticated": True},
                [
                    (
                        "Set-Cookie",
                        f"instinct_session={self.server.session_token}; HttpOnly; SameSite=Strict; Path=/",
                    )
                ],
            )
        self._dispatch("POST", path, body)

    def _events(self):
        """Server-sent events: push the live snapshot whenever it changes."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        async def snapshot():
            return self.server.service.live()

        last, quiet = None, 0.0
        try:
            while not self.server.service.closing:
                future = asyncio.run_coroutine_threadsafe(snapshot(), self.server.loop)
                data = json.dumps(future.result(timeout=5), ensure_ascii=False)
                if data != last:
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    last, quiet = data, 0.0
                elif quiet >= 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    quiet = 0.0
                time.sleep(0.1)
                quiet += 0.1
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return
        except concurrent.futures.TimeoutError:
            return

    def _terminal_stream(self, terminal_id):
        """Server-sent events with a terminal's output, base64 encoded."""
        try:
            terminal = self.server.service.terminals.get(terminal_id)
        except ValueError as exc:
            return self._json(404, {"error": str(exc)})
        query = parse_qs(urlsplit(self.path).query)
        try:
            offset = max(0, int(query.get("offset", ["0"])[0]))
        except ValueError:
            return self._json(400, {"error": "offset must be an integer"})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not self.server.service.closing:
                offset, data, closed = terminal.read(offset, 15)
                if data:
                    payload = json.dumps(
                        {"offset": offset, "data": base64.b64encode(data).decode()}
                    )
                    self.wfile.write(f"data: {payload}\n\n".encode())
                elif closed:
                    self.wfile.write(b"event: exit\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                else:
                    self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _dispatch(self, method, path, body):
        future = asyncio.run_coroutine_threadsafe(
            self.server.service.dispatch(method, path, body), self.server.loop
        )
        try:
            result = future.result(timeout=10)
            headers = (
                [("Content-Disposition", 'attachment; filename="instinct-trace.json"')]
                if path.endswith("/export")
                else []
            )
            self._json(200, result, headers)
        except concurrent.futures.TimeoutError:
            # Do not imply rollback of a mutation already submitted to the loop.
            self._json(
                503,
                {
                    "error": "Controller response timed out; refresh state before retrying"
                },
            )
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception:
            self._json(
                500, {"error": "Unexpected controller failure; inspect the local trace"}
            )
