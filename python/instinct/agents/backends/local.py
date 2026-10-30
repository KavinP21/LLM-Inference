"""A worker that runs its model in this process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..protocol import ChatMessage, Generation
from .config import (
    SAMPLING_OPTIONS,
    BackendError,
    CapacityError,
    ContextLengthError,
    GenerationCancelledError,
    GenerationTimeoutError,
    RequestValidationError,
    WorkerConfig,
    WorkerUnavailableError,
    validate_context_limit,
    validate_options,
    validate_request,
)


class LocalModelBackend:
    """Serial actor for one Instinct engine with cancellation between engine steps.

    Loading, tokenization, inference, cancellation, and closing all execute on
    the actor thread. No caller may concurrently access its non-thread-safe
    engine. Queue time counts against the generation deadline.
    """

    supports_context_limits = True

    def __init__(
        self,
        config: WorkerConfig,
        *,
        engine_factory: Callable[[], Any] | None = None,
        tokenizer_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.native_call_dialect = "json"
        # Latest decoded text per in-flight request, for live progress.
        self.partial: dict[str, str] = {}
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
        # A ChatRunner batches every request it is given, so each waits on its
        # own thread; other engines run one request at a time.
        # An injected engine_factory supplies the runner itself for "chat",
        # or a submit/step engine otherwise.
        self._chat = config.chat_runner and (
            engine_factory is None or config.runner == "chat"
        )
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_pending if self._chat else 1,
            thread_name_prefix=f"instinct-{config.worker_id[:12]}",
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
        self._runner_metadata: dict[str, Any] = {
            "runner": "chat" if self._chat else "engine",
            "runner_version": "0.2.0" if self._chat else "0.1.0",
        }
        self._last_runner_stats: dict[str, Any] | None = None
        self._artifact_identity: dict[str, str | None] = {
            "model_data_sha256": None,
            "model_config_sha256": None,
            "tokenizer_signature": None,
            "draft_model_data_sha256": None,
        }
        if self._chat and Path(config.model).is_dir():
            # Everything that identifies a checkpoint worker comes from its
            # files, so it's known (and the same) before and after loading.
            from ...chat_formats import family_for
            from ...chat_runner import checkpoint_fingerprint

            config_file = Path(config.model) / "config.json"
            model_type = (
                json.loads(config_file.read_text()).get("model_type", "")
                if config_file.is_file()
                else ""
            )
            self.native_call_dialect = family_for(model_type).dialect
            self._runner_metadata.update(family=family_for(model_type).name)
            self._artifact_identity.update(checkpoint_fingerprint(config.model))

    def _load(self) -> None:
        if self._engine is not None:
            return
        if self._chat:
            self._load_runner()
            return
        if self._tokenizer_factory:
            tokenizer = self._tokenizer_factory()
        else:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise BackendError(
                    "install instinct[agents] to load a chat tokenizer"
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
            )
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("the worker tokenizer requires an explicit chat template")
        if self.supports_native_tools and not self._tokenizer_factory:
            probe_name = "instinct_native_capability_probe"
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
            from instinct import create_engine
            from instinct.model_file import ModelFile

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
        if not callable(getattr(engine, "forget", None)):
            close = getattr(engine, "close", None)
            if close:
                close()
            raise BackendError(
                "worker requires terminal request cleanup; rebuild the Instinct engine with forget() support"
            )
        self._tokenizer = tokenizer
        self._engine = engine
        self._actual_backend = getattr(
            engine,
            "backend",
            "cuda" if self.config.backend == "auto" else self.config.backend,
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

    def _load_runner(self) -> None:
        options = self.config.engine_options
        if self._engine_factory:
            runner = self._engine_factory()
        else:
            from ...chat_runner import ChatRunner

            runner = ChatRunner(
                self.config.model,
                tokenizer=self.config.tokenizer,
                max_num_sequences=self.config.max_pending,
                max_context_tokens=self.config.max_model_length,
                kv_cache_bytes=self.config.kv_cache_bytes,
                prefix_cache_bytes=options.get("prefix_cache_bytes"),
                prefill_chunk_size=options.get("prefill_chunk_size", 512),
                kv_bits=options.get("kv_bits", 16),
                eos_token_ids=self.config.eos_token_ids,
                local_files_only=self.config.local_files_only,
            )
        self._engine = runner
        self._tokenizer = runner.tokenizer
        self._actual_backend = "mlx"
        self._context_limit = runner.max_context_tokens
        self._model_type = runner.model_type

    async def start(self) -> LocalModelBackend:
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
            "schema": "instinct_worker_v1",
            "worker_id": self.config.worker_id,
            "ready": self._engine is not None and not closed,
            "loaded": self._engine is not None,
            "model": self.config.model,
            "tokenizer": self.config.tokenizer,
            "backend": self._actual_backend,
            "max_model_length": self._context_limit,
            "max_pending": self.config.max_pending,
            "active": active,
            "completed": self._completed,
            "capabilities": ["greedy_chat", "cancellation", "replica"],
            "native_tool_calls": self.supports_native_tools,
            "enable_thinking": self.config.enable_thinking,
            "supports_media": self.config.takes_media,
            "supports_context_limits": self.supports_context_limits,
            "native_call_dialect": self.native_call_dialect,
            **self._artifact_identity,
            **self._runner_metadata,
            "last_runner_stats": self._last_runner_stats,
        }

    def _forget(self, engine_request_id: int) -> None:
        self._engine.forget(engine_request_id)

    @property
    def max_model_length(self):
        return self._context_limit

    def _chat_messages(self, messages: tuple[ChatMessage, ...]) -> list[dict[str, str]]:
        if self._model_type != "gemma3_text":
            return [
                {"role": message.role, "content": message.content}
                for message in messages
            ]
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
        max_context_tokens: int | None = None,
        options: dict[str, Any] | None = None,
        request_id: str = "",
    ) -> Generation:
        def check() -> None:
            if cancel.is_set():
                raise GenerationCancelledError("generation cancelled")
            if time.monotonic() >= deadline:
                raise GenerationTimeoutError("generation deadline exceeded")

        check()
        if not self.config.takes_media and any(message.media for message in messages):
            raise RequestValidationError("this worker's model doesn't take images")
        self._load()
        check()
        if self._chat:
            return self._generate_runner(
                messages,
                max_tokens,
                tool_specs,
                cancel,
                deadline,
                max_context_tokens,
                options or {},
                request_id,
            )
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
        context_limit = min(
            self._context_limit, max_context_tokens or self._context_limit
        )
        if len(token_ids) + max_tokens > context_limit:
            raise ContextLengthError(
                f"prompt has {len(token_ids)} tokens; {max_tokens} output tokens exceed the {context_limit}-token context"
            )
        check()
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

    def _generate_runner(
        self,
        messages,
        max_tokens,
        tool_specs,
        cancel,
        deadline,
        max_context_tokens,
        options,
        request_id,
    ) -> Generation:
        from ..media import materialize_media

        context_limit = min(
            self._context_limit, max_context_tokens or self._context_limit
        )
        effort = options.get("effort") or self.config.engine_options.get(
            "effort", "medium" if self.config.enable_thinking else "off"
        )
        partial: list[str] = []

        def on_text(piece: str) -> None:
            partial.append(piece)
            self.partial[request_id] = "".join(partial)

        with tempfile.TemporaryDirectory(prefix="instinct-worker-media-") as temporary:
            turns = []
            for index, message in enumerate(messages):
                turn = {"role": message.role, "content": message.content}
                if message.media:
                    directory = Path(temporary) / str(index)
                    directory.mkdir(mode=0o700)
                    paths = materialize_media(message.media, directory)
                    turn["media"] = [
                        {
                            "type": part["type"],
                            "path": part[part["type"]],
                            "key": item["sha256"],
                        }
                        for part, item in zip(paths, message.media)
                    ]
                turns.append(turn)
            try:
                sampling = {k: options[k] for k in SAMPLING_OPTIONS & set(options)}
                result = self._engine.generate(
                    turns,
                    max_tokens=max_tokens,
                    tools=tool_specs,
                    effort=effort,
                    cancelled=lambda: cancel.is_set() or time.monotonic() >= deadline,
                    on_text=on_text,
                    max_context_tokens=context_limit,
                    **sampling,
                )
            except InterruptedError as exc:
                if time.monotonic() >= deadline:
                    raise GenerationTimeoutError(
                        "generation deadline exceeded"
                    ) from exc
                raise GenerationCancelledError("generation cancelled") from exc
            except ValueError as exc:
                if "context" in str(exc):
                    raise ContextLengthError(str(exc)) from exc
                raise
            finally:
                self.partial.pop(request_id, None)
        self._completed += 1
        self._last_runner_stats = self._engine.stats()
        return Generation(
            result.text,
            result.input_tokens,
            result.output_tokens,
            result.finish_reason,
            self.config.model,
            result.dialect,
            result.reasoning,
            result.metrics,
        )

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
        if not self.supports_native_tools:
            raise RequestValidationError(
                "native tool calls are disabled on this worker"
            )
        from ..native_tools import validate_tool_specs

        specs = validate_tool_specs(tool_specs)
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
        max_context_tokens = validate_context_limit(max_context_tokens)
        if max_context_tokens is not None and max_tokens >= max_context_tokens:
            raise ContextLengthError(
                "output budget leaves no room in the requested context"
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
            self._generate,
            messages,
            max_tokens,
            cancel,
            deadline,
            tool_specs,
            max_context_tokens,
            options,
            request_id,
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
