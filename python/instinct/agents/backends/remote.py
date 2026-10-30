"""Workers reached over HTTP, including ones this process launches."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from ..media import MAX_MEDIA_REQUEST_BYTES
from ..protocol import ChatMessage, Generation
from .config import (
    DIALECTS,
    AuthenticationError,
    BackendError,
    CapacityError,
    ContextLengthError,
    GenerationCancelledError,
    GenerationTimeoutError,
    RequestValidationError,
    WorkerConfig,
    WorkerUnavailableError,
    _positive,
    validate_context_limit,
    validate_options,
    validate_request,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        # Never forward a bearer credential to a different endpoint.
        return None


def is_loopback(host: str | None) -> bool:
    if host and host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


class RemoteWorkerBackend:
    """Authenticated HTTP job client with bounded admission and idempotent retry.

    An ambiguous submit is retried at the same worker with the same request ID.
    Completed jobs are retained by the worker for its advertised retention time.
    The client never silently reroutes an ambiguously submitted request.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        max_concurrency: int = 1,
        request_timeout: float = 120.0,
        connect_timeout: float = 5.0,
        poll_interval: float = 0.05,
        allow_insecure: bool = False,
        max_response_bytes: int = 2 << 20,
        max_request_bytes: int = 1 << 20,
    ) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("worker URL must be HTTP or HTTPS")
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/")
        ):
            raise ValueError("worker URL must contain only scheme, host, and port")
        if (
            parsed.scheme == "http"
            and not is_loopback(parsed.hostname)
            and not allow_insecure
        ):
            raise ValueError("remote HTTP requires allow_insecure=True; prefer HTTPS")
        if (
            not isinstance(token, str)
            or not token
            or not token.isascii()
            or any(c.isspace() for c in token)
        ):
            raise ValueError(
                "worker bearer token must be nonempty ASCII without whitespace"
            )
        if type(max_concurrency) is not int or max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.url = url.rstrip("/")
        self._token = token
        self.max_concurrency = max_concurrency
        self.request_timeout = _positive(request_timeout, "request_timeout")
        self.connect_timeout = _positive(connect_timeout, "connect_timeout")
        self.poll_interval = _positive(poll_interval, "poll_interval")
        # Latest decoded text per in-flight request, for live progress.
        self.partial: dict[str, str] = {}
        self.max_response_bytes = max_response_bytes
        if (
            type(max_request_bytes) is not int
            or not 1 <= max_request_bytes <= MAX_MEDIA_REQUEST_BYTES
        ):
            raise ValueError(
                "max_request_bytes exceeds the bounded media transport limit"
            )
        self.max_request_bytes = max_request_bytes
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        self._active: set[str] = set()
        self._closed = False
        self.identity: dict[str, Any] | None = None
        self.supports_native_tools = False
        self.supports_context_limits = False

    def _http(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        payload = (
            None
            if body is None
            else json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        if payload is not None and len(payload) > self.max_request_bytes:
            raise ValueError("request body exceeds the configured worker size limit")
        request = urllib.request.Request(
            self.url + path,
            data=payload,
            method=method,
            headers={
                "Authorization": "Bearer " + self._token,
                "Content-Type": "application/json",
            },
        )
        status = 200
        try:
            response = self._opener.open(
                request, timeout=timeout or self.connect_timeout
            )
        except urllib.error.HTTPError as exc:
            response = exc
            status = exc.code
        except (OSError, urllib.error.URLError) as exc:
            raise WorkerUnavailableError(
                f"worker connection failed ({type(exc).__name__})"
            ) from exc
        try:
            with response:
                raw = response.read(self.max_response_bytes + 1)
        except (OSError, urllib.error.URLError) as exc:
            raise WorkerUnavailableError(
                f"worker response failed ({type(exc).__name__})"
            ) from exc
        if len(raw) > self.max_response_bytes:
            raise BackendError("worker response exceeds the configured size limit")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise BackendError("worker returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise BackendError("worker returned an invalid response object")
        if status >= 400 or "error" in result:
            error = result.get("error", {})
            code = (
                error.get("code", "worker_error")
                if isinstance(error, dict)
                else "worker_error"
            )
            message = (
                error.get("message", "worker failed")
                if isinstance(error, dict)
                else "worker failed"
            )
            error_type = {
                "authentication": AuthenticationError,
                "capacity": CapacityError,
                "context_length": ContextLengthError,
                "timeout": GenerationTimeoutError,
                "cancelled": GenerationCancelledError,
                "unavailable": WorkerUnavailableError,
                "invalid_request": RequestValidationError,
            }.get(code, BackendError)
            raise error_type(str(message)[:2048])
        return result

    async def health(self) -> dict[str, Any]:
        result = await asyncio.to_thread(self._http, "GET", "/health")
        if result.get("schema") != "instinct_worker_v1" or not result.get("ready"):
            raise WorkerUnavailableError(
                "worker is not ready or uses an unsupported protocol"
            )
        self.identity = result
        dialect = result.get("native_call_dialect", "json")
        if dialect not in DIALECTS:
            raise BackendError("worker returned an unsupported native tool dialect")
        self.native_call_dialect = dialect
        native = result.get("native_tool_calls", False)
        if type(native) is not bool:
            raise BackendError("worker returned an invalid native tool capability")
        self.supports_native_tools = native
        context_limits = result.get("supports_context_limits", False)
        if type(context_limits) is not bool:
            raise BackendError("worker returned an invalid context-limit capability")
        self.supports_context_limits = context_limits
        return result

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        *,
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generation:
        return await self._dispatch(
            messages,
            max_tokens,
            request_id,
            max_context_tokens=max_context_tokens,
            options=options,
        )

    async def generate_action(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: Sequence[dict[str, Any]],
        *,
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generation:
        from ..native_tools import validate_tool_specs

        specs = validate_tool_specs(tool_specs)
        if not self.supports_native_tools:
            await self.health()
        if not self.supports_native_tools:
            raise RequestValidationError(
                "remote worker does not support native tool calls"
            )
        return await self._dispatch(
            messages,
            max_tokens,
            request_id,
            specs,
            max_context_tokens=max_context_tokens,
            options=options,
        )

    async def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: list[dict[str, Any]] | None = None,
        *,
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Generation:
        messages = validate_request(
            messages, max_tokens, request_id, allow_tool_messages=tool_specs is not None
        )
        options = validate_options(options)
        if self._closed:
            raise WorkerUnavailableError("worker client is closed")
        max_context_tokens = validate_context_limit(max_context_tokens)
        if max_context_tokens is not None:
            if not self.supports_context_limits:
                await self.health()
            if not self.supports_context_limits:
                raise RequestValidationError(
                    "remote worker does not enforce request context limits"
                )
        if request_id in self._active:
            raise ValueError("request_id is already active on this worker client")
        if len(self._active) >= self.max_concurrency:
            raise CapacityError("worker client concurrency limit reached")
        self._active.add(request_id)
        deadline = time.monotonic() + self.request_timeout
        path = "/v1/jobs/" + urllib.parse.quote(request_id, safe="")
        body = {
            "request_id": request_id,
            "messages": [asdict(m) for m in messages],
            "max_tokens": max_tokens,
            "timeout": self.request_timeout,
        }
        if tool_specs is not None:
            body["tools"] = tool_specs
        if max_context_tokens is not None:
            body["max_context_tokens"] = max_context_tokens
        if options:
            body["options"] = options
        submitted = False
        try:
            # One bounded retry only for a transport failure, with idempotency.
            for attempt in range(2):
                try:
                    result = await asyncio.to_thread(
                        self._http,
                        "POST",
                        "/v1/jobs",
                        body,
                        min(
                            self.connect_timeout,
                            max(0.001, deadline - time.monotonic()),
                        ),
                    )
                    submitted = True
                    break
                except WorkerUnavailableError:
                    if attempt or time.monotonic() >= deadline:
                        raise
            while result.get("status") in {"pending", "running"}:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GenerationTimeoutError("remote generation deadline exceeded")
                if isinstance(result.get("partial"), str):
                    self.partial[request_id] = result["partial"]
                # The worker holds this request until new text arrives or the
                # job finishes, so completion is seen immediately.
                since = len(self.partial.get(request_id, ""))
                wait = round(min(0.5, max(0.0, remaining - 0.05)), 3)
                result = await asyncio.to_thread(
                    self._http,
                    "GET",
                    f"{path}?wait={wait}&since={since}",
                    None,
                    self.connect_timeout + wait,
                )
            if result.get("status") != "completed" or not isinstance(
                result.get("generation"), dict
            ):
                raise BackendError("worker returned an invalid generation state")
            data = result["generation"]
            required = {
                "text",
                "input_tokens",
                "output_tokens",
                "finish_reason",
                "model",
            }
            if (
                not required <= set(data)
                or set(data) - required - {"dialect", "reasoning", "metrics"}
                or data.get("dialect") not in DIALECTS | {None}
                or not isinstance(data.get("reasoning", ""), str)
                or not isinstance(data.get("metrics") or {}, dict)
                or not isinstance(data.get("text"), str)
                or any(
                    type(data.get(name)) is not int or data[name] < 0
                    for name in ("input_tokens", "output_tokens")
                )
            ):
                raise BackendError("worker returned invalid generation fields")
            if data["model"] is not None and not isinstance(data["model"], str):
                raise BackendError("worker returned an invalid model identity")
            if data["output_tokens"] > max_tokens or data.get("finish_reason") not in {
                "stop",
                "length",
            }:
                raise BackendError(
                    "worker violated the generation budget or finish contract"
                )
            return Generation(**data)
        except (asyncio.CancelledError, BackendError) as exc:
            # A retryable transport error leaves the same-ID job alive for
            # recovery at this replica. Its server-side deadline still applies.
            # Aborts, expired deadlines, and permanent contract failures cancel
            # work even when submission may have lost its acknowledgement.
            should_cancel = (
                isinstance(exc, asyncio.CancelledError)
                or (
                    not getattr(exc, "retryable", False)
                    and getattr(exc, "work_started", None) is not False
                )
                or time.monotonic() >= deadline
            )
            if should_cancel and (
                submitted or time.monotonic() < deadline + self.connect_timeout
            ):
                try:
                    await asyncio.shield(self.cancel(request_id))
                except BackendError:
                    pass
            raise
        finally:
            self._active.discard(request_id)
            self.partial.pop(request_id, None)

    @property
    def max_model_length(self):
        return self.identity.get("max_model_length") if self.identity else None

    async def cancel(self, request_id: str) -> None:
        validate_request([ChatMessage("user", "")], 1, request_id)
        try:
            await asyncio.to_thread(
                self._http,
                "DELETE",
                "/v1/jobs/" + urllib.parse.quote(request_id, safe=""),
            )
        except BackendError as exc:
            # Cancelling an unsubmitted or expired ID is harmless.
            if not isinstance(
                exc, GenerationCancelledError
            ) and "unknown request" not in str(exc):
                raise

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(
            *(self.cancel(r) for r in tuple(self._active)), return_exceptions=True
        )


class LocalProcessBackend(RemoteWorkerBackend):
    """Supervised independent Python process owning one full Instinct replica."""

    @classmethod
    async def start(
        cls,
        config: WorkerConfig,
        *,
        startup_timeout: float = 120.0,
        defer_load: bool = False,
    ) -> LocalProcessBackend:
        _positive(startup_timeout, "startup_timeout")
        token = secrets.token_urlsafe(32)
        env = dict(os.environ)
        env["INSTINCT_WORKER_TOKEN"] = token
        if config.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        command = [
            sys.executable,
            "-m",
            "instinct.agents.worker",
            "--config-json",
            json.dumps(asdict(config)),
            "--port",
            "0",
            "--exit-with-parent",
            *(["--defer-load"] if defer_load else []),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_lines: list[str] = []

        async def drain_stderr() -> None:
            assert process.stderr is not None
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return
                stderr_lines.append(chunk.decode(errors="replace"))
                del stderr_lines[:-16]

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            assert process.stdout is not None
            line = await asyncio.wait_for(
                process.stdout.readline(), timeout=startup_timeout
            )
            if not line:
                await process.wait()
                raise WorkerUnavailableError(
                    f"local worker exited during startup (exit {process.returncode}): {''.join(stderr_lines)[-2048:]}"
                )
            readiness = json.loads(line)
            if readiness.get("event") != "ready" or not isinstance(
                readiness.get("port"), int
            ):
                raise WorkerUnavailableError(
                    "local worker returned an invalid readiness message"
                )
            backend = cls(
                f"http://127.0.0.1:{readiness['port']}",
                token,
                max_concurrency=config.max_pending if config.chat_runner else 1,
                request_timeout=config.request_timeout,
                max_request_bytes=MAX_MEDIA_REQUEST_BYTES
                if config.takes_media
                else 1 << 20,
            )
            backend.process = process
            backend._stderr_task = stderr_task
            backend._stderr_lines = stderr_lines
            await backend.health()
            return backend
        except BaseException:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 5.0)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            await stderr_task
            raise

    async def close(self) -> None:
        await super().close()
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 10.0)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        await self._stderr_task
