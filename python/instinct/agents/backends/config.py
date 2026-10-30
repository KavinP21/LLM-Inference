"""Worker settings, request validation and the errors workers report."""

from __future__ import annotations

import json
import math
import platform
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ...chat_formats import EFFORT_BUDGETS
from ..media import MAX_TOTAL_MEDIA_BYTES, validate_media
from ..protocol import ChatMessage

DIALECTS = frozenset({"json", "qwen3_coder_xml", "gemma4"})


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


SAMPLING_OPTIONS = {"temperature", "top_p", "top_k", "seed"}


def validate_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Per-request generation options: reasoning effort and sampling."""
    if options is None:
        return {}
    if not isinstance(options, Mapping) or set(options) - {"effort"} - SAMPLING_OPTIONS:
        raise RequestValidationError(
            "generation options accept effort, temperature, top_p, top_k and seed"
        )
    effort = options.get("effort", "off")
    if effort not in EFFORT_BUDGETS:
        raise RequestValidationError(f"effort must be one of {sorted(EFFORT_BUDGETS)}")
    result: dict[str, Any] = {"effort": effort}
    for name in SAMPLING_OPTIONS & set(options):
        value = options[name]
        integral = name in {"top_k", "seed"}
        if isinstance(value, bool) or not isinstance(
            value, int if integral else (int, float)
        ):
            raise RequestValidationError(f"{name} must be a number")
        result[name] = value
    return result


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


def validate_context_limit(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or not 1 <= value <= 1_000_000):
        raise RequestValidationError(
            "max_context_tokens must be an integer between 1 and 1000000"
        )
    return value


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
    media_total = 0
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
        media = validate_media(message.media)
        if media and message.role != "user":
            raise ValueError("media attachments require the user role")
        media_total += sum(
            len(item["data"]) * 3 // 4 - item["data"].count("=") for item in media
        )
        if media_total > MAX_TOTAL_MEDIA_BYTES:
            raise ValueError("conversation media exceeds the byte limit")
        result.append(ChatMessage(message.role, message.content, media))
    if total > 1 << 20:
        raise ValueError("chat messages exceed the 1 MiB limit")
    return tuple(result)


# Options an MLX worker passes to its ChatRunner.
CHAT_OPTIONS = frozenset(
    {"effort", "kv_bits", "prefill_chunk_size", "prefix_cache_bytes"}
)


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
    eos_token_ids: list[int] | None = None
    local_files_only: bool = True
    native_tool_calls: bool = False
    # "chat": ChatRunner, many requests at once (MLX checkpoints and .engine
    # files on Apple silicon). "engine": one request at a time through an
    # engine's submit/step API (CUDA). "auto" picks chat on Apple silicon.
    runner: str = "auto"
    enable_thinking: bool = False
    # Set only in independently launched processes; never mutate the parent's GPU.
    cuda_visible_devices: str | None = None

    @property
    def chat_runner(self) -> bool:
        if self.runner != "auto":
            return self.runner == "chat"
        return (
            self.backend != "cuda"
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"
        )

    @property
    def takes_media(self) -> bool:
        """Image input, which Gemma 4 checkpoints have on the chat runner."""
        config = Path(self.model) / "config.json"
        if not (self.chat_runner and config.is_file()):
            return False
        settings = json.loads(config.read_text())
        return settings.get("model_type") == "gemma4" and bool(
            settings.get("vision_config")
        )

    def __post_init__(self) -> None:
        if not self.model or not self.tokenizer:
            raise ValueError("model and tokenizer are required")
        if self.backend not in {"auto", "mlx", "cuda"}:
            raise ValueError("backend must be auto, mlx, or cuda")
        if self.runner == "vlm":
            raise ValueError(
                'the mlx-vlm runner was removed; MLX checkpoints run on the engine (runner "auto")'
            )
        if self.runner not in {"auto", "chat", "engine"}:
            raise ValueError("runner must be auto, chat or engine")
        if type(self.enable_thinking) is not bool:
            raise TypeError("enable_thinking must be a boolean")
        if self.runner == "chat" and (
            self.backend == "cuda" or self.cuda_visible_devices is not None
        ):
            raise ValueError("the chat runner needs Apple MLX execution")
        for name in ("max_model_length", "kv_cache_bytes", "max_pending"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _positive(self.request_timeout, "request_timeout")
        if type(self.native_tool_calls) is not bool:
            raise TypeError("native_tool_calls must be a boolean")
        if not isinstance(self.engine_options, dict):
            raise TypeError("engine_options must be a dictionary")
        if self.chat_runner and self.engine_options.keys() - CHAT_OPTIONS:
            raise ValueError(
                f"engine options for MLX workers are {sorted(CHAT_OPTIONS)}"
            )
        if self.engine_options.get("effort", "off") not in EFFORT_BUDGETS:
            raise ValueError(f"effort must be one of {sorted(EFFORT_BUDGETS)}")
        if self.engine_options.get("kv_bits", 16) not in {8, 16}:
            raise ValueError("kv_bits must be 8 or 16")
        if {
            "max_model_length",
            "kv_cache_bytes",
            "max_num_sequences",
        } & self.engine_options.keys():
            raise ValueError("engine_options cannot override worker resource limits")
        if self.eos_token_ids is not None and (
            not isinstance(self.eos_token_ids, list)
            or any(type(token) is not int or token < 0 for token in self.eos_token_ids)
        ):
            raise ValueError("eos_token_ids must be a list of nonnegative integers")
        validate_request([ChatMessage("user", "")], 1, self.worker_id)
