"""Authenticated bounded HTTP model-replica worker.

Run one process per replica/GPU. The job protocol acknowledges submission before
inference, allowing polling, cancellation, and safe retries with the same ID.
No model-generated code or tools execute inside this model-only service.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import signal
import socket
import ssl
import threading
import time
import urllib.parse
from collections import OrderedDict
from concurrent.futures import CancelledError, Future
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .backends import (
    AuthenticationError,
    BackendError,
    CapacityError,
    ContextLengthError,
    GenerationCancelledError,
    GenerationTimeoutError,
    LocalForgeBackend,
    RequestValidationError,
    WorkerConfig,
    WorkerUnavailableError,
    is_loopback,
    validate_request,
)
from .protocol import ChatMessage, Generation, ModelBackend


@dataclass
class _Job:
    request_id: str
    fingerprint: str
    created: float
    future: Future[Any] | None = None
    completed: float | None = None
    result: dict[str, Any] | None = None
    result_bytes: int = 0


def _error(exc: BaseException) -> tuple[int, dict[str, Any]]:
    if isinstance(exc, AuthenticationError):
        code, status = "authentication", 401
    elif isinstance(exc, CapacityError):
        code, status = "capacity", 429
    elif isinstance(exc, ContextLengthError):
        code, status = "context_length", 422
    elif isinstance(exc, (GenerationTimeoutError, asyncio.TimeoutError)):
        code, status = "timeout", 408
    elif isinstance(
        exc, (GenerationCancelledError, CancelledError, asyncio.CancelledError)
    ):
        code, status = "cancelled", 409
    elif isinstance(exc, WorkerUnavailableError):
        code, status = "unavailable", 503
    elif isinstance(exc, (ValueError, TypeError)):
        code, status = "invalid_request", 400
    else:
        code, status = "generation_failed", 500
    message = (
        str(exc)
        if code != "generation_failed"
        else f"model generation failed ({type(exc).__name__})"
    )
    return status, {"error": {"code": code, "message": message[:2048] or code}}


class WorkerService:
    """Thread-safe jobs around an async backend, with bounded retained history."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        max_pending: int = 8,
        max_history: int = 1024,
        history_ttl: float = 600.0,
        max_timeout: float = 120.0,
        max_result_bytes: int = 2 << 20,
        max_history_bytes: int = 32 << 20,
        worker_id: str = "worker",
    ) -> None:
        for name, value in (
            ("max_pending", max_pending),
            ("max_history", max_history),
            ("max_result_bytes", max_result_bytes),
            ("max_history_bytes", max_history_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in (history_ttl, max_timeout)):
            raise ValueError(
                "worker history and deadline limits must be positive and finite"
            )
        self.backend = backend
        self.max_pending = max_pending
        self.max_history = max_history
        self.history_ttl = history_ttl
        self.max_timeout = max_timeout
        self.max_result_bytes = max_result_bytes
        if max_history_bytes < max_result_bytes:
            raise ValueError("max_history_bytes must cover one maximum result")
        self.max_history_bytes = max_history_bytes
        self.worker_id = worker_id
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, _Job] = OrderedDict()
        self._closed = False
        self._ready = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"forge-jobs-{worker_id}", daemon=True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        self._loop.close()

    def start(self, timeout: float = 120.0) -> None:
        async def start_backend() -> None:
            start = getattr(self.backend, "start", None)
            if start:
                await start()

        asyncio.run_coroutine_threadsafe(start_backend(), self._loop).result(timeout)
        self._ready = True

    def _prune(self) -> None:
        now = time.monotonic()
        for key, job in tuple(self._jobs.items()):
            if job.completed is not None and now - job.completed > self.history_ttl:
                del self._jobs[key]
        completed = [
            key for key, job in self._jobs.items() if job.completed is not None
        ]
        for key in completed[: -self.max_history]:
            del self._jobs[key]
        retained_bytes = sum(job.result_bytes for job in self._jobs.values())
        for key, job in tuple(self._jobs.items()):
            if retained_bytes <= self.max_history_bytes:
                break
            if job.completed is not None:
                retained_bytes -= job.result_bytes
                del self._jobs[key]

    async def _generate(
        self,
        messages: tuple[ChatMessage, ...],
        max_tokens: int,
        request_id: str,
        timeout: float,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> Generation:
        if tool_specs is not None:
            return await asyncio.wait_for(
                self.backend.generate_action(
                    messages, max_tokens, request_id, tool_specs=tool_specs
                ),
                timeout=timeout,
            )
        return await asyncio.wait_for(
            self.backend.generate(messages, max_tokens, request_id), timeout=timeout
        )

    def _done(self, job: _Job, future: Future[Any]) -> None:
        try:
            generation = future.result()
            if not isinstance(generation, Generation):
                raise BackendError("backend returned an unsupported generation")
            result = {
                "status": "completed",
                "request_id": job.request_id,
                "generation": asdict(generation),
            }
            if (
                len(
                    json.dumps(result, ensure_ascii=False, allow_nan=False).encode(
                        "utf-8"
                    )
                )
                > self.max_result_bytes
            ):
                raise BackendError("generation exceeds the worker response limit")
        except BaseException as exc:  # noqa: BLE001 - isolate model-provider failures and cancellation.
            _, result = _error(exc)
            if isinstance(exc, (ValueError, TypeError)) and not isinstance(
                exc, (ContextLengthError, RequestValidationError)
            ):
                # This exception came from a dispatched model backend. A native
                # forward pass may already have run; a Python ValueError is not
                # evidence that admission rejected the request before inference.
                result = {
                    "error": {
                        "code": "generation_failed",
                        "message": f"model generation failed ({type(exc).__name__})",
                    }
                }
            result["status"] = "failed"
            result["request_id"] = job.request_id
        with self._lock:
            # Explicit cancellation wins over a late completion callback.
            if job.completed is None:
                job.completed = time.monotonic()
                job.result = result
                job.result_bytes = len(
                    json.dumps(result, ensure_ascii=False).encode("utf-8")
                )
            self._prune()

    def submit(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict) or set(payload) - {
            "request_id",
            "messages",
            "max_tokens",
            "timeout",
            "tools",
        }:
            raise ValueError("invalid job submission fields")
        raw_messages = payload.get("messages")
        if not isinstance(raw_messages, list):
            raise TypeError("messages must be a list")
        messages = []
        for raw in raw_messages:
            if not isinstance(raw, dict) or set(raw) != {"role", "content"}:
                raise ValueError("messages require exactly role and content")
            messages.append(ChatMessage(**raw))
        request_id = payload.get("request_id")
        max_tokens = payload.get("max_tokens")
        messages = validate_request(
            messages, max_tokens, request_id, allow_tool_messages="tools" in payload
        )
        tool_specs = None
        if "tools" in payload:
            if getattr(
                self.backend, "supports_native_tools", False
            ) is not True or not callable(
                getattr(self.backend, "generate_action", None)
            ):
                raise RequestValidationError(
                    "worker does not support native tool calls"
                )
            from .native_tools import validate_tool_specs

            tool_specs = validate_tool_specs(payload["tools"])
        timeout = payload.get("timeout", self.max_timeout)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        timeout = min(float(timeout), self.max_timeout)
        canonical = {
            "messages": [asdict(m) for m in messages],
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if tool_specs is not None:
            canonical["tools"] = tool_specs
        fingerprint = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        with self._lock:
            if self._closed or not self._ready:
                raise WorkerUnavailableError("worker is not ready")
            self._prune()
            existing = self._jobs.get(request_id)
            if existing:
                if existing.fingerprint != fingerprint:
                    raise ValueError("request_id was already used for a different job")
                return existing.result or {
                    "status": "pending",
                    "request_id": request_id,
                }
            active = sum(job.completed is None for job in self._jobs.values())
            if active >= self.max_pending:
                raise CapacityError("worker queue is full")
            job = _Job(request_id, fingerprint, time.monotonic())
            self._jobs[request_id] = job
            job.future = asyncio.run_coroutine_threadsafe(
                self._generate(messages, max_tokens, request_id, timeout, tool_specs),
                self._loop,
            )
            job.future.add_done_callback(lambda future: self._done(job, future))
            return {"status": "pending", "request_id": request_id}

    def get(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            self._prune()
            job = self._jobs.get(request_id)
            if not job:
                raise ValueError("unknown request_id")
            return job.result or {"status": "pending", "request_id": request_id}

    def cancel(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(request_id)
            if not job:
                raise ValueError("unknown request_id")
            if job.completed is None:
                job.completed = time.monotonic()
                _, job.result = _error(GenerationCancelledError("generation cancelled"))
                job.result.update(status="failed", request_id=request_id)
                job.result_bytes = len(json.dumps(job.result).encode("utf-8"))
                if job.future:
                    job.future.cancel()
            return job.result or {"status": "pending", "request_id": request_id}

    def health(self) -> dict[str, Any]:
        # Health must never inspect the engine from an HTTP handler thread.
        method = getattr(self.backend, "health", None)
        identity = (
            method() if method and not asyncio.iscoroutinefunction(method) else {}
        )
        with self._lock:
            return {
                **identity,
                "schema": "forge_worker_v1",
                "worker_id": identity.get("worker_id", self.worker_id),
                "ready": self._ready and not self._closed,
                "max_pending": self.max_pending,
                "active": sum(job.completed is None for job in self._jobs.values()),
                "history_ttl": self.history_ttl,
                "max_history": self.max_history,
                "max_history_bytes": self.max_history_bytes,
                "max_timeout": self.max_timeout,
                "pid": os.getpid(),
                "native_tool_calls": getattr(
                    self.backend, "supports_native_tools", False
                )
                is True,
            }

    def close(self, timeout: float = 15.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = [
                job.future
                for job in self._jobs.values()
                if job.completed is None and job.future
            ]
        for future in futures:
            future.cancel()

        async def close_backend() -> None:
            close = getattr(self.backend, "close", None)
            if close:
                await close()

        try:
            asyncio.run_coroutine_threadsafe(close_backend(), self._loop).result(
                timeout
            )
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=timeout)


class WorkerHTTPServer(ThreadingHTTPServer):
    """HTTP/1.0 server with bounded handler threads and bounded request bodies."""

    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 32

    def __init__(
        self,
        address: tuple[str, int],
        service: WorkerService,
        token: str,
        *,
        max_handler_threads: int = 32,
        max_body_bytes: int = 1 << 20,
        socket_timeout: float = 5.0,
    ) -> None:
        if not token or not token.isascii() or any(c.isspace() for c in token):
            raise ValueError("worker requires an ASCII bearer token without whitespace")
        if (
            type(max_handler_threads) is not int
            or max_handler_threads <= 0
            or type(max_body_bytes) is not int
            or max_body_bytes <= 0
        ):
            raise ValueError("HTTP worker resource limits must be positive integers")
        if not math.isfinite(socket_timeout) or socket_timeout <= 0:
            raise ValueError("socket_timeout must be positive and finite")
        self.service = service
        self._token = token
        self.max_body_bytes = max_body_bytes
        self.socket_timeout = socket_timeout
        self._handler_slots = threading.BoundedSemaphore(max_handler_threads)
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler)

    def get_request(self) -> tuple[Any, Any]:
        connection, address = super().get_request()
        connection.settimeout(self.socket_timeout)
        return connection, address

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._handler_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()


class _Handler(BaseHTTPRequestHandler):
    server: WorkerHTTPServer
    server_version = "ForgeWorker/1"

    def log_message(self, format: str, *args: Any) -> None:
        # Prompts, tokens, and bearer credentials never enter HTTP logs.
        return

    def _respond(self, status: int, result: dict[str, Any]) -> None:
        raw = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)
        self.close_connection = True

    def _authorize(self) -> None:
        value = self.headers.get("Authorization", "")
        expected = "Bearer " + self.server._token
        if not value.isascii() or not hmac.compare_digest(value, expected):
            raise AuthenticationError("invalid worker bearer token")

    def _request_id(self) -> str:
        path = urllib.parse.urlsplit(self.path)
        if path.query or path.fragment or not path.path.startswith("/v1/jobs/"):
            raise ValueError("unknown worker route")
        request_id = urllib.parse.unquote(path.path[len("/v1/jobs/") :])
        validate_request([ChatMessage("user", "")], 1, request_id)
        return request_id

    def _body(self) -> Any:
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("chunked request bodies are unsupported")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("request Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if len(self.headers.get_all("Content-Length", [])) != 1:
            raise ValueError("request requires exactly one Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError as exc:
            raise ValueError("request requires a valid Content-Length") from exc
        if not 0 < length <= self.server.max_body_bytes:
            raise ValueError("request body exceeds the worker size limit")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("truncated request body")

        def reject_constant(_: str) -> None:
            raise ValueError("JSON non-finite values are unsupported")

        def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON object key")
                result[key] = value
            return result

        try:
            return json.loads(
                raw, parse_constant=reject_constant, object_pairs_hook=unique_object
            )
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("request body must be valid UTF-8 JSON") from exc

    def _route(self, method: str) -> None:
        try:
            self._authorize()
            if method == "GET" and self.path == "/health":
                result = self.server.service.health()
            elif method == "POST" and self.path == "/v1/jobs":
                result = self.server.service.submit(self._body())
            elif method == "GET":
                result = self.server.service.get(self._request_id())
            elif method == "DELETE":
                result = self.server.service.cancel(self._request_id())
            else:
                raise ValueError("unknown worker route")
            self._respond(200, result)
        except Exception as exc:  # noqa: BLE001 - convert one HTTP request's failure into a bounded response.
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError)):
                return
            status, result = _error(exc)
            try:
                self._respond(status, result)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def do_DELETE(self) -> None:
        self._route("DELETE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve one authenticated Forge model replica"
    )
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--backend", choices=("auto", "mlx", "cuda"), default="auto")
    parser.add_argument("--max-model-length", type=int, default=2048)
    parser.add_argument("--kv-cache-bytes", type=int, default=256 << 20)
    parser.add_argument("--max-pending", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--worker-id")
    parser.add_argument(
        "--engine-options", default="{}", help="JSON engine-specific options"
    )
    parser.add_argument(
        "--allow-download", action="store_true", help="allow fetching tokenizer files"
    )
    parser.add_argument(
        "--native-tool-calls",
        action="store_true",
        help="enable native tokenizer function catalogs",
    )
    parser.add_argument("--config-json", help=argparse.SUPPRESS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--token-env", default="FORGE_WORKER_TOKEN")
    parser.add_argument(
        "--allow-insecure-lan",
        action="store_true",
        help="explicitly permit authenticated plaintext HTTP on a remote interface",
    )
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if not token:
        parser.error(f"set a bearer credential in {args.token_env}")
    if bool(args.tls_cert) != bool(args.tls_key):
        parser.error("--tls-cert and --tls-key must be supplied together")
    if not is_loopback(args.host) and not args.tls_cert and not args.allow_insecure_lan:
        parser.error("a remote bind requires TLS or --allow-insecure-lan")
    if not 0 <= args.port <= 65535:
        parser.error("port must be between zero and 65535")
    try:
        if args.config_json:
            config = WorkerConfig(**json.loads(args.config_json))
        else:
            values: dict[str, Any] = {
                "model": args.model,
                "tokenizer": args.tokenizer,
                "backend": args.backend,
                "max_model_length": args.max_model_length,
                "kv_cache_bytes": args.kv_cache_bytes,
                "max_pending": args.max_pending,
                "request_timeout": args.request_timeout,
                "engine_options": json.loads(args.engine_options),
                "local_files_only": not args.allow_download,
                "native_tool_calls": args.native_tool_calls,
            }
            if args.worker_id:
                values["worker_id"] = args.worker_id
            config = WorkerConfig(**values)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    backend = LocalForgeBackend(config)
    service = WorkerService(
        backend,
        max_pending=config.max_pending,
        max_timeout=config.request_timeout,
        worker_id=config.worker_id,
    )
    server: WorkerHTTPServer | None = None
    try:
        service.start()
        server = WorkerHTTPServer((args.host, args.port), service, token)
        if args.tls_cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(args.tls_cert, args.tls_key)
            server.socket = context.wrap_socket(server.socket, server_side=True)

        def stop(signum: int, frame: Any) -> None:
            assert server is not None
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(
            json.dumps(
                {
                    "event": "ready",
                    "host": args.host,
                    "port": server.server_port,
                    "worker_id": config.worker_id,
                    "pid": os.getpid(),
                }
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.1)
        return 0
    finally:
        if server:
            server.server_close()
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())
