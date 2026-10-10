"""Model replicas and bounded routing for the agent runtime.

Each local backend owns one engine on a dedicated thread. Separate worker
processes own separate complete model replicas, including their KV memory.
This is request distribution, not model or tensor parallelism.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from .protocol import ChatMessage, Generation, ModelBackend


class BackendError(RuntimeError):
    """A model worker rejected or failed a generation."""

    retryable = False
    work_started = True  # Conservatively account for unknown partial inference.


class RequestValidationError(BackendError, ValueError):
    """A worker rejected a request before inference."""

    work_started = False


class CapacityError(BackendError):
    """The bounded worker queue is full."""

    retryable = True
    work_started = False


class AuthenticationError(BackendError):
    """A remote worker rejected authentication."""

    work_started = False


class ContextLengthError(BackendError, ValueError):
    """The prompt and generation do not fit the model context."""

    work_started = False


class GenerationTimeoutError(BackendError, TimeoutError):
    """The generation exhausted its deadline."""


class GenerationCancelledError(BackendError):
    """The generation was explicitly cancelled."""


class WorkerUnavailableError(BackendError):
    """A worker could not be reached or is shutting down."""

    retryable = True


def validate_request(
    messages: Sequence[ChatMessage],
    max_tokens: int,
    request_id: str,
    *,
    allow_tool_messages: bool = False,
) -> tuple[ChatMessage, ...]:
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError("request_id must contain 1 to 128 characters")
    if not all(c.isascii() and (c.isalnum() or c in "_.:-") for c in request_id):
        raise ValueError("request_id contains unsupported characters")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
        raise ValueError("max_tokens must be an integer between 1 and 65536")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of chat messages")
    if not 1 <= len(messages) <= 256:
        raise ValueError("messages must contain 1 to 256 entries")
    result = []
    total = 0
    for message in messages:
        if not isinstance(message, ChatMessage):
            raise TypeError("each message must be a ChatMessage")
        allowed_roles = (
            {"system", "user", "assistant", "tool"}
            if allow_tool_messages
            else {"system", "user", "assistant"}
        )
        if message.role not in allowed_roles:
            raise ValueError("unsupported message role")
        if not isinstance(message.content, str):
            raise TypeError("message content must be text")
        total += len(message.content.encode("utf-8"))
        result.append(message)
    if total > 1 << 20:
        raise ValueError("chat messages exceed the 1 MiB limit")
    return tuple(result)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class WorkerConfig:
    model: str
    tokenizer: str
    backend: str = "auto"
    max_model_length: int = 2048
    kv_cache_bytes: int = 256 << 20
    max_pending: int = 8
    request_timeout: float = 120.0
    worker_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    engine_options: dict[str, Any] = field(default_factory=dict)
    speculative: dict[str, Any] | None = None
    eos_token_ids: list[int] | None = None
    local_files_only: bool = True
    native_tool_calls: bool = False
    # Set only in independently launched processes; never mutate the parent's GPU.
    cuda_visible_devices: str | None = None

    def __post_init__(self) -> None:
        if not self.model or not self.tokenizer:
            raise ValueError("model and tokenizer are required")
        if self.backend not in {"auto", "mlx", "cuda"}:
            raise ValueError("backend must be auto, mlx, or cuda")
        for name in ("max_model_length", "kv_cache_bytes", "max_pending"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.request_timeout, "request_timeout")
        if type(self.native_tool_calls) is not bool:
            raise TypeError("native_tool_calls must be a boolean")
        if not isinstance(self.engine_options, dict):
            raise TypeError("engine_options must be a dictionary")
        if {
            "max_model_length",
            "kv_cache_bytes",
            "max_num_sequences",
        } & self.engine_options.keys():
            raise ValueError("engine_options cannot override worker resource limits")
        if self.speculative is not None:
            if not isinstance(self.speculative, dict) or self.backend == "cuda":
                raise ValueError(
                    "speculative workers currently require an MLX configuration"
                )
            if set(self.speculative) - {
                "draft_model_path",
                "draft_tokenizer",
                "draft_tokens",
                "verification_mode",
                "adaptive",
                "draft_kv_cache_bytes",
            }:
                raise ValueError("unknown speculative worker options")
            if set(self.engine_options) - {
                "prefill_chunk_size",
                "custom_metal",
                "metal_paged_attention",
                "attention_tile_size",
            }:
                raise ValueError(
                    "speculative engine_options accept only prefill_chunk_size, custom_metal, metal_paged_attention, and attention_tile_size"
                )
        if self.eos_token_ids is not None and (
            not isinstance(self.eos_token_ids, list)
            or any(type(token) is not int or token < 0 for token in self.eos_token_ids)
        ):
            raise ValueError("eos_token_ids must be a list of nonnegative integers")
        validate_request([ChatMessage("user", "")], 1, self.worker_id)


class LocalForgeBackend:
    """Serial actor for one Forge engine with cancellation between engine steps.

    Loading, tokenization, inference, cancellation, and closing all execute on
    the actor thread. No caller may concurrently access its non-thread-safe
    engine. Queue time counts against the generation deadline.
    """

    def __init__(
        self,
        config: WorkerConfig,
        *,
        engine_factory: Callable[[], Any] | None = None,
        tokenizer_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.supports_native_tools = config.native_tool_calls
        if (
            config.cuda_visible_devices is not None
            and os.environ.get("CUDA_VISIBLE_DEVICES") != config.cuda_visible_devices
        ):
            raise ValueError(
                "GPU placement must be configured when launching a LocalProcessBackend"
            )
        self._engine_factory = engine_factory
        self._tokenizer_factory = tokenizer_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"forge-{config.worker_id[:12]}"
        )
        self._lock = threading.Lock()
        self._requests: dict[str, threading.Event] = {}
        self._closed = False
        self._engine: Any = None
        self._tokenizer: Any = None
        self._vocab_size: int | None = None
        self._context_limit = config.max_model_length
        self._eos: list[int] = []
        self._actual_backend = config.backend
        self._model_type = ""
        self._completed = 0
        self._last_speculative_stats: dict[str, Any] | None = None
        self._artifact_identity: dict[str, str | None] = {
            "model_data_sha256": None,
            "model_config_sha256": None,
            "tokenizer_signature": None,
            "draft_model_data_sha256": None,
        }

    def _load(self) -> None:
        if self._engine is not None:
            return
        if self._tokenizer_factory:
            tokenizer = self._tokenizer_factory()
        else:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise BackendError(
                    "install forge-llm[agents] to load a chat tokenizer"
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("the worker tokenizer requires an explicit chat template")
        if self.supports_native_tools and not self._tokenizer_factory:
            probe_name = "forge_native_capability_probe"
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Check available function definitions."}],
                tokenize=False,
                add_generation_prompt=True,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": probe_name,
                            "description": "Capability probe; never executed.",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            )
            if not isinstance(rendered, str) or probe_name not in rendered:
                raise RequestValidationError(
                    "tokenizer chat template does not render native tools; disable native_tool_calls or use a tool-aware tokenizer"
                )
        if self._engine_factory:
            engine = self._engine_factory()
            model_config = getattr(getattr(engine, "model", None), "config", None)
        else:
            from forge_llm import create_engine
            from forge_llm.model_file import ModelFile

            with ModelFile(self.config.model) as artifact:
                model_config = artifact.config
                self._artifact_identity["model_data_sha256"] = artifact.data_sha256
                self._artifact_identity["model_config_sha256"] = hashlib.sha256(
                    json.dumps(asdict(model_config), sort_keys=True).encode()
                ).hexdigest()
            # Compare token IDs, rather than vocabulary length (some artifacts
            # pad the output embedding vocabulary).
            vocabulary = tokenizer.get_vocab()
            if not vocabulary or any(
                type(t) is not int or not 0 <= t < model_config.vocab_size
                for t in vocabulary.values()
            ):
                raise ValueError("tokenizer token IDs do not fit the model vocabulary")
            self._artifact_identity["tokenizer_signature"] = hashlib.sha256(
                json.dumps(
                    {
                        "vocab": vocabulary,
                        "special_ids": tokenizer.all_special_ids,
                        "chat_template": tokenizer.chat_template,
                        # Fast-tokenizer merges and normalization also affect the
                        # model input even when the token ID vocabulary is identical.
                        "backend": tokenizer.backend_tokenizer.to_str()
                        if hasattr(tokenizer, "backend_tokenizer")
                        else None,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if self.config.speculative is not None:
                from forge_llm.speculative import SpeculativeEngine

                spec_options = dict(self.config.speculative)
                draft_tokenizer_name = spec_options.pop("draft_tokenizer", None)
                if draft_tokenizer_name:
                    from transformers import AutoTokenizer

                    draft_tokenizer = AutoTokenizer.from_pretrained(
                        draft_tokenizer_name,
                        local_files_only=self.config.local_files_only,
                        trust_remote_code=False,
                    )

                    def fingerprint(value: Any) -> str:
                        payload = {
                            "vocab": value.get_vocab(),
                            "special_ids": value.all_special_ids,
                        }
                        return hashlib.sha256(
                            json.dumps(payload, sort_keys=True).encode()
                        ).hexdigest()

                    spec_options["target_tokenizer_fingerprint"] = fingerprint(
                        tokenizer
                    )
                    spec_options["draft_tokenizer_fingerprint"] = fingerprint(
                        draft_tokenizer
                    )
                engine = SpeculativeEngine(
                    self.config.model,
                    max_model_length=self.config.max_model_length,
                    kv_cache_bytes=self.config.kv_cache_bytes,
                    **self.config.engine_options,
                    **spec_options,
                )
            else:
                engine = create_engine(
                    self.config.model,
                    backend=self.config.backend,
                    max_num_sequences=1,
                    max_model_length=self.config.max_model_length,
                    kv_cache_bytes=self.config.kv_cache_bytes,
                    **self.config.engine_options,
                )
            loaded_file = getattr(getattr(engine, "model", None), "file", None)
            if loaded_file is not None:
                self._artifact_identity["model_data_sha256"] = loaded_file.data_sha256
                self._artifact_identity["model_config_sha256"] = hashlib.sha256(
                    json.dumps(asdict(loaded_file.config), sort_keys=True).encode()
                ).hexdigest()
            draft_file = getattr(getattr(engine, "draft_model", None), "file", None)
            if draft_file is not None:
                self._artifact_identity["draft_model_data_sha256"] = (
                    draft_file.data_sha256
                )
        if self.config.speculative is None and not callable(
            getattr(engine, "forget", None)
        ):
            close = getattr(engine, "close", None)
            if close:
                close()
            raise BackendError(
                "worker requires terminal request cleanup; rebuild the Forge engine with forget() support"
            )
        self._tokenizer = tokenizer
        self._engine = engine
        self._actual_backend = (
            "mlx"
            if self.config.speculative is not None
            else getattr(
                engine,
                "backend",
                "cuda" if self.config.backend == "auto" else self.config.backend,
            )
        )
        self._context_limit = int(
            getattr(engine, "max_model_length", self.config.max_model_length)
        )
        self._vocab_size = getattr(model_config, "vocab_size", None)
        self._model_type = getattr(model_config, "model_type", "")
        configured_eos = getattr(model_config, "eos_token_id", None)
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        self._eos = sorted(
            {int(t) for t in (configured_eos, tokenizer_eos) if t is not None}
        )
        if self.config.eos_token_ids is not None:
            self._eos = sorted(set(self.config.eos_token_ids))
        elif not self._tokenizer_factory:
            # Gemma and other chat models may terminate a turn with an additional
            # token absent from the binary artifact's single EOS field.
            try:
                from transformers import GenerationConfig

                generation_config = GenerationConfig.from_pretrained(
                    self.config.tokenizer,
                    local_files_only=True,
                )
                ids = generation_config.eos_token_id
                if ids is not None:
                    self._eos = sorted(set(ids if isinstance(ids, list) else [ids]))
            except OSError:
                pass
        if any(
            type(t) is not int
            or t < 0
            or (self._vocab_size is not None and t >= self._vocab_size)
            for t in self._eos
        ):
            self._engine.close()
            self._engine = None
            raise ValueError("EOS token IDs do not fit the model vocabulary")

    async def start(self) -> LocalForgeBackend:
        with self._lock:
            if self._closed:
                raise WorkerUnavailableError("worker is closed")
        await asyncio.wrap_future(self._executor.submit(self._load))
        return self

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = len(self._requests)
            closed = self._closed
        return {
            "schema": "forge_worker_v1",
            "worker_id": self.config.worker_id,
            "ready": self._engine is not None and not closed,
            "model": self.config.model,
            "tokenizer": self.config.tokenizer,
            "backend": self._actual_backend,
            "max_model_length": self._context_limit,
            "max_pending": self.config.max_pending,
            "active": active,
            "completed": self._completed,
            "capabilities": ["greedy_chat", "cancellation", "replica"],
            "native_tool_calls": self.supports_native_tools,
            "speculative": self.config.speculative,
            "last_speculative_stats": self._last_speculative_stats,
            **self._artifact_identity,
        }

    def _forget(self, engine_request_id: int) -> None:
        self._engine.forget(engine_request_id)

    def _chat_messages(self, messages: tuple[ChatMessage, ...]) -> list[dict[str, str]]:
        if self._model_type != "gemma3_text":
            return [asdict(message) for message in messages]
        # Gemma's native template has no system role and requires alternating
        # user/assistant turns. Preserve the coordinator instructions in the
        # first user turn, and combine adjacent evidence messages of one role.
        system: list[str] = []
        turns: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                if turns:
                    raise ValueError(
                        "Gemma system messages must precede conversation turns"
                    )
                system.append(message.content)
                continue
            content = message.content
            if not turns:
                if message.role != "user":
                    raise ValueError("Gemma conversations must start with a user turn")
                content = "\n\n".join(system + [content])
            if turns and turns[-1]["role"] == message.role:
                turns[-1]["content"] += "\n\n" + content
            else:
                turns.append({"role": message.role, "content": content})
        if not turns:
            raise ValueError("Gemma conversations require a user message")
        return turns

    def _generate(
        self,
        messages: tuple[ChatMessage, ...],
        max_tokens: int,
        cancel: threading.Event,
        deadline: float,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> Generation:
        def check() -> None:
            if cancel.is_set():
                raise GenerationCancelledError("generation cancelled")
            if time.monotonic() >= deadline:
                raise GenerationTimeoutError("generation deadline exceeded")

        check()
        self._load()
        check()
        try:
            template_options = {"tools": tool_specs} if tool_specs is not None else {}
            token_ids = self._tokenizer.apply_chat_template(
                self._chat_messages(messages),
                tokenize=True,
                add_generation_prompt=True,
                **template_options,
            )
        except (ValueError, TypeError) as exc:
            raise RequestValidationError(
                "chat template could not encode the request"
            ) from exc
        if isinstance(token_ids, Mapping):
            token_ids = token_ids.get("input_ids")
        if not isinstance(token_ids, (list, tuple)) or not token_ids:
            raise RequestValidationError(
                "chat template must produce a nonempty list of token IDs"
            )
        if any(
            type(t) is not int
            or t < 0
            or (self._vocab_size is not None and t >= self._vocab_size)
            for t in token_ids
        ):
            raise RequestValidationError(
                "chat template emitted invalid model token IDs"
            )
        if len(token_ids) + max_tokens > self._context_limit:
            raise ContextLengthError(
                f"prompt has {len(token_ids)} tokens; {max_tokens} output tokens exceed the {self._context_limit}-token context"
            )
        check()
        if self.config.speculative is not None:
            result = self._engine.generate_result(
                token_ids,
                max_tokens,
                self._eos,
                cancelled=lambda: cancel.is_set() or time.monotonic() >= deadline,
            )
            check()
            if result.finish_reason == "cancelled":
                raise GenerationCancelledError("speculative generation cancelled")
            if (
                result.finish_reason not in {"eos", "length"}
                or len(result.tokens) > max_tokens
            ):
                raise BackendError(
                    "speculative engine violated its generation contract"
                )
            self._last_speculative_stats = result.stats.to_dict()
            self._completed += 1
            return Generation(
                text=self._tokenizer.decode(result.tokens, skip_special_tokens=True),
                input_tokens=len(token_ids),
                output_tokens=len(result.tokens),
                finish_reason="stop" if result.finish_reason == "eos" else "length",
                model=self.config.model,
            )
        engine_request_id = self._engine.submit(token_ids, max_tokens, self._eos)
        output: list[int] = []
        completed = False
        try:
            while not completed:
                check()
                events = self._engine.step()
                for event in events:
                    if event.request_id != engine_request_id:
                        raise BackendError(
                            "engine emitted a token for an unowned request"
                        )
                    output.append(int(event.token))
                    completed = bool(event.finished)
                if len(output) > max_tokens:
                    raise BackendError("engine exceeded the generation token budget")
            check()
            text = self._tokenizer.decode(output, skip_special_tokens=True)
            if not isinstance(text, str):
                raise BackendError("tokenizer did not decode text")
            self._completed += 1
            return Generation(
                text=text,
                input_tokens=len(token_ids),
                output_tokens=len(output),
                finish_reason="stop"
                if output and output[-1] in self._eos
                else "length",
                model=self.config.model,
            )
        finally:
            if not completed:
                self._engine.cancel(engine_request_id)
            self._forget(engine_request_id)

    async def generate(
        self, messages: Sequence[ChatMessage], max_tokens: int, request_id: str
    ) -> Generation:
        return await self._dispatch(messages, max_tokens, request_id)

    async def generate_action(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: Sequence[dict[str, Any]],
    ) -> Generation:
        if not self.supports_native_tools:
            raise RequestValidationError(
                "native tool calls are disabled on this worker"
            )
        from .native_tools import validate_tool_specs

        specs = validate_tool_specs(tool_specs)
        return await self._dispatch(messages, max_tokens, request_id, specs)

    async def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> Generation:
        messages = validate_request(
            messages, max_tokens, request_id, allow_tool_messages=tool_specs is not None
        )
        cancel = threading.Event()
        with self._lock:
            if self._closed:
                raise WorkerUnavailableError("worker is closed")
            if request_id in self._requests:
                raise ValueError("request_id is already active on this worker")
            if len(self._requests) >= self.config.max_pending:
                raise CapacityError("worker queue is full")
            self._requests[request_id] = cancel
        deadline = time.monotonic() + self.config.request_timeout
        future = self._executor.submit(
            self._generate, messages, max_tokens, cancel, deadline, tool_specs
        )

        # Keep capacity reserved until the actor actually stops. asyncio task
        # cancellation cannot interrupt an in-flight GPU kernel.
        def released(_: Any) -> None:
            with self._lock:
                self._requests.pop(request_id, None)

        future.add_done_callback(released)
        wrapped = asyncio.wrap_future(future)
        # A cancelled awaiting task leaves inference finishing on its actor.
        # Consume its eventual exception without hiding it from normal awaits.
        wrapped.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            cancel.set()
            raise

    async def cancel(self, request_id: str) -> None:
        with self._lock:
            event = self._requests.get(request_id)
        if event:
            event.set()

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for event in self._requests.values():
                event.set()

        def close_engine() -> None:
            if self._engine is not None:
                close = getattr(self._engine, "close", None)
                if close:
                    close()
                self._engine = None

        await asyncio.wrap_future(self._executor.submit(close_engine))
        self._executor.shutdown(wait=True)


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
        self.max_response_bytes = max_response_bytes
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
        self._active: set[str] = set()
        self._closed = False
        self.identity: dict[str, Any] | None = None
        self.supports_native_tools = False

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
        if payload is not None and len(payload) > 1 << 20:
            raise ValueError("request body exceeds 1 MiB")
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
        if result.get("schema") != "forge_worker_v1" or not result.get("ready"):
            raise WorkerUnavailableError(
                "worker is not ready or uses an unsupported protocol"
            )
        self.identity = result
        native = result.get("native_tool_calls", False)
        if type(native) is not bool:
            raise BackendError("worker returned an invalid native tool capability")
        self.supports_native_tools = native
        return result

    async def generate(
        self, messages: Sequence[ChatMessage], max_tokens: int, request_id: str
    ) -> Generation:
        return await self._dispatch(messages, max_tokens, request_id)

    async def generate_action(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: Sequence[dict[str, Any]],
    ) -> Generation:
        from .native_tools import validate_tool_specs

        specs = validate_tool_specs(tool_specs)
        if not self.supports_native_tools:
            await self.health()
        if not self.supports_native_tools:
            raise RequestValidationError(
                "remote worker does not support native tool calls"
            )
        return await self._dispatch(messages, max_tokens, request_id, specs)

    async def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> Generation:
        messages = validate_request(
            messages, max_tokens, request_id, allow_tool_messages=tool_specs is not None
        )
        if self._closed:
            raise WorkerUnavailableError("worker client is closed")
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
                await asyncio.sleep(min(self.poll_interval, remaining))
                result = await asyncio.to_thread(
                    self._http,
                    "GET",
                    path,
                    None,
                    min(self.connect_timeout, max(0.001, deadline - time.monotonic())),
                )
            if result.get("status") != "completed" or not isinstance(
                result.get("generation"), dict
            ):
                raise BackendError("worker returned an invalid generation state")
            data = result["generation"]
            if (
                set(data)
                != {"text", "input_tokens", "output_tokens", "finish_reason", "model"}
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


class WorkerPool:
    """Bounded least-loaded replica routing, optionally pinned by worker name.

    A generation stays on its selected replica. Retries retain that route for
    route_ttl seconds and max_route_history IDs within this pool's lifetime.
    Route history is in memory; a coordinator restart loses this affinity.
    """

    def __init__(
        self,
        backends: Mapping[str, ModelBackend] | Sequence[ModelBackend],
        *,
        max_pending: int = 32,
        per_worker_concurrency: int = 1,
        max_route_history: int = 1024,
        route_ttl: float = 600.0,
    ) -> None:
        if (
            type(max_pending) is not int
            or max_pending <= 0
            or type(per_worker_concurrency) is not int
            or per_worker_concurrency <= 0
        ):
            raise ValueError("worker pool limits must be positive integers")
        self.backends = (
            dict(backends)
            if isinstance(backends, Mapping)
            else {f"worker-{i}": backend for i, backend in enumerate(backends)}
        )
        if not self.backends or any(
            not isinstance(name, str) or not name for name in self.backends
        ):
            raise ValueError("worker pool requires named backends")
        if len({id(b) for b in self.backends.values()}) != len(self.backends):
            raise ValueError("each replica must own a distinct backend")
        self.max_pending = max_pending
        if type(max_route_history) is not int or max_route_history < max_pending:
            raise ValueError("max_route_history must be at least max_pending")
        self.max_route_history = max_route_history
        self.route_ttl = _positive(route_ttl, "route_ttl")
        self._routes: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._limits = {
            name: min(
                per_worker_concurrency,
                getattr(backend, "max_concurrency", per_worker_concurrency),
            )
            for name, backend in self.backends.items()
        }
        self._loads = dict.fromkeys(self.backends, 0)
        self._condition = asyncio.Condition()
        self._requests: dict[str, str | None] = {}
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False
        self._cursor = 0

    @property
    def loads(self) -> dict[str, int]:
        return dict(self._loads)

    @property
    def supports_native_tools(self) -> bool:
        return all(
            getattr(backend, "supports_native_tools", False) is True
            and callable(getattr(backend, "generate_action", None))
            for backend in self.backends.values()
        )

    def _prune_routes(self) -> None:
        now = time.monotonic()
        for request_id, (_, created) in tuple(self._routes.items()):
            if request_id not in self._requests and now - created > self.route_ttl:
                del self._routes[request_id]
        for request_id in tuple(self._routes):
            if len(self._routes) <= self.max_route_history:
                break
            if request_id not in self._requests:
                del self._routes[request_id]

    async def generate(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        *,
        worker: str | None = None,
    ) -> Generation:
        return await self._dispatch(messages, max_tokens, request_id, worker=worker)

    async def generate_action(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        tool_specs: Sequence[dict[str, Any]],
        *,
        worker: str | None = None,
    ) -> Generation:
        if not self.supports_native_tools:
            raise RequestValidationError(
                "every replica in a native-tool pool must support native tool calls"
            )
        from .native_tools import validate_tool_specs

        specs = validate_tool_specs(tool_specs)
        return await self._dispatch(
            messages, max_tokens, request_id, worker=worker, tool_specs=specs
        )

    async def _dispatch(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
        request_id: str,
        *,
        worker: str | None = None,
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> Generation:
        messages = validate_request(
            messages, max_tokens, request_id, allow_tool_messages=tool_specs is not None
        )
        if worker is not None and worker not in self.backends:
            raise ValueError("unknown worker route")
        selected: str | None = None
        async with self._condition:
            if self._closed:
                raise WorkerUnavailableError("worker pool is closed")
            if request_id in self._requests:
                raise ValueError("request_id is already active in this pool")
            if len(self._requests) >= self.max_pending:
                raise CapacityError("worker pool queue is full")
            self._prune_routes()
            previous = self._routes.get(request_id)
            if previous:
                if worker is not None and worker != previous[0]:
                    raise ValueError("request_id is already pinned to another worker")
                worker = previous[0]
            self._requests[request_id] = None
            task = asyncio.current_task()
            if task is not None:
                self._tasks[request_id] = task
        try:
            async with self._condition:
                while selected is None:
                    if self._closed:
                        raise WorkerUnavailableError("worker pool is closed")
                    names = [worker] if worker else list(self.backends)
                    available = [
                        name for name in names if self._loads[name] < self._limits[name]
                    ]
                    if available:
                        minimum = min(self._loads[name] for name in available)
                        ties = [
                            name for name in available if self._loads[name] == minimum
                        ]
                        selected = ties[self._cursor % len(ties)]
                        self._cursor += 1
                        self._loads[selected] += 1
                        self._requests[request_id] = selected
                        self._routes[request_id] = (selected, time.monotonic())
                        self._routes.move_to_end(request_id)
                        self._prune_routes()
                    else:
                        await self._condition.wait()
            if tool_specs is not None:
                return await self.backends[selected].generate_action(
                    messages, max_tokens, request_id, tool_specs=tool_specs
                )
            return await self.backends[selected].generate(
                messages, max_tokens, request_id
            )
        finally:
            async with self._condition:
                if selected is not None:
                    self._loads[selected] -= 1
                self._requests.pop(request_id, None)
                self._tasks.pop(request_id, None)
                self._condition.notify_all()

    async def cancel(self, request_id: str) -> None:
        async with self._condition:
            selected = self._requests.get(request_id)
            if selected is None:
                retained = self._routes.get(request_id)
                if retained is not None:
                    selected = retained[0]
            task = self._tasks.get(request_id)
        try:
            if selected:
                cancel = getattr(self.backends[selected], "cancel", None)
                if cancel:
                    await cancel(request_id)
        finally:
            if task is not None and task is not asyncio.current_task():
                task.cancel()

    async def close(self) -> None:
        async with self._condition:
            self._closed = True
            active_tasks = tuple(self._tasks.values())
            self._condition.notify_all()
        for task in active_tasks:
            if task is not asyncio.current_task():
                task.cancel()
        await asyncio.gather(
            *(task for task in active_tasks if task is not asyncio.current_task()),
            return_exceptions=True,
        )
        await asyncio.gather(
            *(
                backend.close()
                for backend in self.backends.values()
                if hasattr(backend, "close")
            ),
            return_exceptions=False,
        )


WorkerPoolBackend = WorkerPool


class LocalProcessBackend(RemoteWorkerBackend):
    """Supervised independent Python process owning one full Forge replica."""

    @classmethod
    async def start(
        cls, config: WorkerConfig, *, startup_timeout: float = 120.0
    ) -> LocalProcessBackend:
        _positive(startup_timeout, "startup_timeout")
        token = secrets.token_urlsafe(32)
        env = dict(os.environ)
        env["FORGE_WORKER_TOKEN"] = token
        if config.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        command = [
            sys.executable,
            "-m",
            "forge_llm.agents.worker",
            "--config-json",
            json.dumps(asdict(config)),
            "--port",
            "0",
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            env=env,
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
                max_concurrency=1,
                request_timeout=config.request_timeout,
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
