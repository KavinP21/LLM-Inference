"""Chat on the MLX engine, for many callers at once.

Callers on any thread call :meth:`ChatRunner.generate` with a conversation.
One engine thread owns the model and steps every active request together, so
concurrent chats and sub-agents share each forward pass. This module adds
what a chat needs on top of token generation:

* Rendering. The model's chat template, with tools and thinking on or off,
  and the per-family conversions in ``chat_formats``.
* Conversation reuse. After a reply, the exact token stream is kept. When
  the next request is the same conversation plus new messages, only the new
  messages are rendered and appended, so the prompt starts with exactly the
  tokens the engine's prefix cache holds and only the new part is prefilled.
  For other prompts the boundary before the final message is checkpointed,
  which is what a sibling sub-agent or the next chat turn will share.
* Thinking budgets, streaming text and per-request metrics.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chat_formats import (
    ChatFamily,
    family_for,
    incremental_suffix,
    split_thinking,
    thinking_budget,
    to_gemma_messages,
)
from .mlx_engine import MlxEngine
from .runtime import TERMINAL
from .sampling import SamplingParams

MIN_CHECKPOINT_TOKENS = 256


@dataclass
class RunnerResult:
    text: str
    reasoning: str
    reasoning_finished: bool
    finish_reason: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    dialect: str
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Session:
    messages: list[dict]
    render_key: str
    tokens: list[int]
    stop_text: str


@dataclass
class _Job:
    ids: list[int]
    max_tokens: int
    sampling: SamplingParams
    checkpoints: list[int]
    namespace: str
    # Thinking budget: after this many thinking tokens, force the close
    # marker. Counting starts at the open marker (or at once if the prompt
    # already opened the thought); 0 means no limit.
    budget: int | None
    open_token: int
    close: list[int]
    counting: bool = False
    # (position, content key, pixels) for each image in the prompt.
    images: list[tuple[int, str, Any]] = field(default_factory=list)
    tokens: queue.Queue = field(default_factory=queue.Queue)
    request_id: int | None = None
    error: BaseException | None = None
    reused: int = 0
    finish: str = ""
    thinking: int = 0
    closed: bool = False


_DONE = object()


def checkpoint_fingerprint(path: str | Path) -> dict[str, str]:
    """Cheap identity for resumable runs: each file's size plus a hash of its
    first and last MiB. That covers the safetensors header (shapes, dtypes,
    offsets), so a re-quantized or replaced file changes it, without reading
    gigabytes of weights at every start."""
    directory = Path(path)

    def digest(names: list[str]) -> str:
        entries = []
        for name in sorted(names):
            file = directory / name
            if not file.is_file():
                continue
            size = file.stat().st_size
            with file.open("rb") as stream:
                head = stream.read(1 << 20)
                stream.seek(max(0, size - (1 << 20)))
                tail = stream.read(1 << 20)
            entries.append([name, size, hashlib.sha256(head + tail).hexdigest()])
        return hashlib.sha256(json.dumps(entries).encode()).hexdigest()

    files = [f.name for f in directory.iterdir() if f.is_file()]
    tokenizer = [
        n
        for n in files
        if n
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "vocab.json",
        }
    ]
    return {
        "model_data_sha256": digest([n for n in files if n.endswith(".safetensors")]),
        "model_config_sha256": digest(["config.json"]),
        "tokenizer_signature": digest(tokenizer),
    }


def _identity(message: dict) -> dict:
    """The parts of a message that affect the prompt (media by content hash)."""
    return {
        "role": message["role"],
        "content": message.get("content", ""),
        "media": [
            (item["type"], item.get("key") or item["path"])
            for item in message.get("media", ())
        ],
    }


class ChatRunner:
    def __init__(
        self,
        path: str | Path,
        *,
        tokenizer: str | Path | None = None,
        max_num_sequences: int = 8,
        max_context_tokens: int | None = None,
        kv_cache_bytes: int = 4 << 30,
        prefix_cache_bytes: int | None = None,
        prefill_chunk_size: int = 512,
        kv_bits: int = 16,
        max_sessions: int = 64,
        eos_token_ids: Sequence[int] | None = None,
        local_files_only: bool = True,
    ) -> None:
        from transformers import AutoTokenizer

        started = time.perf_counter()
        self.path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer or path),
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("the tokenizer has no chat template")
        # MLX streams belong to the thread that makes them, so the engine is
        # built on the thread that runs it.
        self._lock = threading.Lock()
        self._inbox: queue.Queue = queue.Queue()
        self._wake = threading.Event()
        self._closed = False
        options = dict(
            max_num_sequences=max_num_sequences,
            max_model_length=max_context_tokens,
            kv_cache_bytes=kv_cache_bytes,
            prefix_cache_bytes=kv_cache_bytes // 2
            if prefix_cache_bytes is None
            else prefix_cache_bytes,
            prefill_chunk_size=prefill_chunk_size,
            kv_bits=kv_bits,
        )
        loaded = threading.Event()
        failure: list[BaseException] = []
        self._thread = threading.Thread(
            target=self._run,
            args=(path, options, loaded, failure),
            name="instinct-engine",
            daemon=True,
        )
        self._thread.start()
        loaded.wait()
        if failure:
            raise failure[0]
        self.mx = self.engine.mx
        self.model_type = self.engine.model.model_type
        self.family: ChatFamily = family_for(self.model_type)
        self.max_context_tokens = self.engine.max_model_length
        self.stop_ids = self._stop_ids(eos_token_ids)
        self.load_seconds = time.perf_counter() - started
        self.weight_bytes = self.mx.get_active_memory()
        self.max_sessions = max_sessions
        self._sessions: OrderedDict[int, _Session] = OrderedDict()
        self._session_key = 0
        self._image_cache: OrderedDict[str, Any] = OrderedDict()
        self.totals = {
            "requests": 0,
            "prompt_tokens": 0,
            "reused_tokens": 0,
            "generated_tokens": 0,
            "session_hits": 0,
        }
        self.last_metrics: dict[str, Any] = {}

    def _stop_ids(self, configured: Sequence[int] | None) -> frozenset[int]:
        if configured is not None:
            return frozenset(int(t) for t in configured)
        ids = set(self.engine.model.eos_token_ids or ())
        generation = self.path / "generation_config.json"
        if self.path.is_dir() and generation.is_file():
            value = json.loads(generation.read_text()).get("eos_token_id")
            ids.update(
                value
                if isinstance(value, list)
                else [value]
                if value is not None
                else []
            )
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return frozenset(int(t) for t in ids)

    # --- the engine thread ----------------------------------------------------

    def _run(self, path, options, loaded, failure) -> None:
        try:
            self.engine = MlxEngine(path, **options)
        except BaseException as exc:
            failure.append(exc)
            return
        finally:
            loaded.set()
        self._loop()

    def _loop(self) -> None:
        jobs: dict[int, _Job] = {}
        engine = self.engine
        while True:
            while True:
                try:
                    kind, job = self._inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "submit":
                    try:
                        images = [
                            (at, self._embed(key, pixels))
                            for at, key, pixels in job.images
                        ]
                        job.request_id = engine.submit(
                            job.ids,
                            job.max_tokens,
                            self.stop_ids,
                            sampling=job.sampling,
                            cache_namespace=job.namespace,
                            checkpoints=job.checkpoints,
                            images=images,
                        )
                        jobs[job.request_id] = job
                    except Exception as exc:  # Reported to the caller, not the loop.
                        job.error = exc
                        job.tokens.put(_DONE)
                elif job.request_id in jobs:
                    engine.cancel(job.request_id)
                    self._close_job(jobs.pop(job.request_id))
            if self._closed:
                for job in jobs.values():
                    engine.cancel(job.request_id)
                    job.error = RuntimeError("the model is shutting down")
                    self._close_job(job)
                engine.close()
                return
            if not engine.has_work():
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            try:
                events = engine.step()
            except Exception as exc:
                for job in jobs.values():
                    job.error = exc
                    self._close_job(job)
                jobs.clear()
                continue
            for event in events:
                job = jobs.get(event.request_id)
                if job is None:
                    continue
                job.tokens.put(event.token)
                self._budget(job, event.token)
                if event.finished:
                    self._close_job(jobs.pop(event.request_id))

    def _embed(self, key: str, pixels: Any) -> Any:
        """Image soft tokens, kept for the images of recent conversations."""
        embeddings = self._image_cache.pop(key, None)
        if embeddings is None:
            embeddings = self.engine.model.vision.encode(pixels)
            self.mx.eval(embeddings)
        self._image_cache[key] = embeddings
        while len(self._image_cache) > 32:
            self._image_cache.popitem(last=False)
        return embeddings

    def _budget(self, job: _Job, token: int) -> None:
        if not job.budget or job.closed:
            return
        if not job.counting:
            job.counting = token == job.open_token
            return
        if token == job.close[-1]:
            job.closed = True
            return
        job.thinking += 1
        if job.budget and job.thinking >= job.budget:
            job.closed = True
            self.engine.force(job.request_id, job.close)

    def _close_job(self, job: _Job) -> None:
        request = self.engine.requests.get(job.request_id) if job.request_id else None
        if request is not None and request.state in TERMINAL:
            job.reused = request.reused
            job.finish = request.reason
            self.engine.forget(job.request_id)
        job.tokens.put(_DONE)

    # --- prompts --------------------------------------------------------------

    def _template_messages(self, messages: Sequence[dict]) -> list[dict]:
        converted = []
        for m in messages:
            content = m.get("content", "")
            if m.get("media"):
                content = [{"type": item["type"]} for item in m["media"]]
                content.append({"type": "text", "text": m.get("content", "")})
            converted.append({"role": m["role"], "content": content})
        if self.family.name == "gemma4":
            return to_gemma_messages(converted)
        if self.model_type == "gemma3_text":
            return _merge_system(converted)
        return converted

    def render(
        self,
        messages: Sequence[dict],
        tools: Any,
        thinking: bool,
        generation_prompt: bool = True,
    ) -> str:
        options = {"tools": tools} if tools else {}
        return self.tokenizer.apply_chat_template(
            self._template_messages(messages),
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=thinking,
            **options,
        )

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _boundary(self, messages, tools, thinking, ids) -> int:
        """Token length of everything before the final message."""
        if len(messages) < 2:
            return 0
        probe = self.render(
            list(messages[:-1]) + [{"role": "user", "content": "\x00\x01"}],
            tools,
            thinking,
        )
        full = self.render(messages, tools, thinking)
        shared = 0
        while shared < min(len(probe), len(full)) and probe[shared] == full[shared]:
            shared += 1
        boundary = 0
        for a, b in zip(self._encode(full[:shared]), ids):
            if a != b:
                break
            boundary += 1
        # The token at the cut may merge differently next time; leave it out.
        return max(0, boundary - 1)

    def _prompt(
        self, messages: list[dict], tools: Any, thinking: bool, render_key: str
    ):
        identity = [_identity(m) for m in messages]
        with self._lock:
            candidates = list(reversed(self._sessions.items()))
        for key, session in candidates:
            n = len(session.messages)
            if (
                session.render_key != render_key
                or n >= len(identity)
                or identity[:n] != session.messages
            ):
                continue
            if any(m.get("media") for m in messages[n:]):
                break  # New images need their tokens expanded; render it all.
            previous = [dict(m) for m in messages[:n]]
            previous[-1] = {**previous[-1], "content": session.messages[-1]["content"]}
            suffix = incremental_suffix(
                lambda ms, gp: self.render(ms, tools, thinking, gp),
                previous,
                messages,
                session.stop_text,
            )
            if suffix is None:
                break
            with self._lock:
                self._sessions.pop(key, None)
            return session.tokens + self._encode(suffix), suffix, "session", []
        text = self.render(messages, tools, thinking)
        if any(m.get("media") for m in messages):
            return self._encode(self._expand_images(text, messages)), text, "cold", []
        ids = self._encode(text)
        boundary = self._boundary(messages, tools, thinking, ids)
        return (
            ids,
            text,
            "cold",
            [boundary] if boundary >= MIN_CHECKPOINT_TOKENS else [],
        )

    def _images(self, messages: Sequence[dict]) -> list[tuple[str, Any]]:
        """(key, pixels) for every image in the conversation, in order."""
        from .backends.gemma4_vision import load_image

        vision = getattr(self.engine.model, "vision", None)
        found = []
        for message in messages:
            for item in message.get("media", ()):
                if item["type"] != "image" or vision is None:
                    raise ValueError(
                        f"{self.model_type} doesn't take {item['type']} input"
                    )
                found.append(
                    (item.get("key") or item["path"], load_image(item["path"]))
                )
        return found

    def _expand_images(self, text: str, messages: Sequence[dict]) -> str:
        """Replace each image placeholder with its run of soft-token placeholders."""
        tokenizer, vision = self.tokenizer, self.engine.model.vision
        counts = iter(
            vision.soft_tokens(pixels) for _, pixels in self._images(messages)
        )
        marker = tokenizer.image_token
        pieces = text.split(marker)
        out = [pieces[0]]
        for piece in pieces[1:]:
            out.append(
                tokenizer.boi_token
                + marker * next(counts)
                + tokenizer.eoi_token
                + piece
            )
        return "".join(out)

    # --- generation -----------------------------------------------------------

    def generate(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int,
        tools: list[dict] | None = None,
        effort: str = "off",
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        cancelled: Callable[[], bool] = lambda: False,
        on_text: Callable[[str], None] | None = None,
        max_context_tokens: int | None = None,
        namespace: str = "",
    ) -> RunnerResult:
        started = time.perf_counter()
        messages = list(messages)
        budget = thinking_budget(effort)
        thinking = budget is not None
        family = self.family
        if temperature is None:
            # Greedy without thinking keeps tool calls deterministic; thinking
            # uses the model card's sampling.
            temperature = family.temperature if thinking else 0.0
        sampling = SamplingParams(
            temperature=temperature,
            top_p=family.top_p if top_p is None else top_p,
            top_k=family.top_k if top_k is None else top_k,
            seed=seed,
        )
        render_key = hashlib.sha256(
            json.dumps([tools, thinking], sort_keys=True, default=str).encode()
        ).hexdigest()
        ids, text, source, checkpoints = self._prompt(
            messages, tools, thinking, render_key
        )
        limit = min(
            max_context_tokens or self.max_context_tokens, self.max_context_tokens
        )
        if len(ids) + max_tokens > limit:
            raise ValueError(
                f"prompt has {len(ids)} tokens; with {max_tokens} output tokens it exceeds the {limit}-token context"
            )
        opens_thinking = thinking and text.rstrip().endswith(family.think_open.strip())
        images = []
        if any(m.get("media") for m in messages):
            media = self._images(messages)
            token = self.engine.model.image_token_id
            starts = [
                i
                for i, t in enumerate(ids)
                if t == token and (i == 0 or ids[i - 1] != token)
            ]
            if len(starts) != len(media):
                raise ValueError(
                    "the prompt's image placeholders don't match its images"
                )
            images = [(at, key, pixels) for at, (key, pixels) in zip(starts, media)]
            # Every image uses the same placeholder token, so cached prefixes
            # are kept apart by the images they contain.
            digest = hashlib.sha256(
                "\n".join(key for _, key, _ in images).encode()
            ).hexdigest()
            namespace = f"{namespace}:images:{digest[:16]}"
        job = _Job(
            ids,
            max_tokens,
            sampling,
            checkpoints,
            namespace,
            budget if thinking else None,
            self._encode(family.think_open)[0],
            self._encode(family.think_close),
            counting=opens_thinking,
            images=images,
        )
        if self._closed:
            raise RuntimeError("the model runner is closed")
        self._inbox.put(("submit", job))
        self._wake.set()

        from tokenizers.decoders import DecodeStream

        stream = DecodeStream(skip_special_tokens=False)
        backend = self.tokenizer.backend_tokenizer
        output: list[int] = []
        pieces: list[str] = []
        first_token_at = None
        while True:
            try:
                token = job.tokens.get(timeout=0.05)
            except queue.Empty:
                token = None
            if token is _DONE:
                break
            if cancelled():
                self._inbox.put(("cancel", job))
                self._wake.set()
                self._drain(job)
                raise InterruptedError("generation cancelled")
            if token is None:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output.append(token)
            if token in self.stop_ids:
                continue
            piece = stream.step(backend, token)
            if piece:
                pieces.append(piece)
                if on_text is not None:
                    on_text(piece)
        if job.error is not None:
            raise job.error
        finished = time.perf_counter()
        stopped = job.finish == "stop"
        raw = "".join(pieces)
        answer, reasoning, reasoning_finished = split_thinking(
            (family.think_open + raw) if opens_thinking else raw, family
        )
        if stopped:
            stop_text = self.tokenizer.decode([output[-1]])
            tokens = ids + output
        else:
            # Cut off: close the turn the way the template would.
            stop_text = family.turn_end
            tokens = ids + output + self._encode(stop_text)
        with self._lock:
            self._session_key += 1
            self._sessions[self._session_key] = _Session(
                [
                    _identity(m)
                    for m in messages + [{"role": "assistant", "content": answer}]
                ],
                render_key,
                tokens,
                stop_text,
            )
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        prefilled = len(ids) - job.reused
        ttft = (first_token_at or finished) - started
        decode_seconds = max(finished - (first_token_at or finished), 1e-9)
        metrics = {
            "cache_source": source
            if source == "session"
            else "prefix"
            if job.reused
            else "cold",
            "prompt_tokens": len(ids),
            "reused_tokens": job.reused,
            "prefilled_tokens": prefilled,
            "ttft_ms": round(ttft * 1000, 1),
            "prefill_tps": round(prefilled / ttft, 1)
            if prefilled and ttft > 0
            else None,
            "decode_tps": round(max(len(output) - 1, 0) / decode_seconds, 1),
            "output_tokens": len(output),
            "total_ms": round((finished - started) * 1000, 1),
            "peak_memory_gb": round(self.mx.get_peak_memory() / 1e9, 2),
            "effort": effort,
            "temperature": temperature,
            "batch": self.engine.stats()["running"],
        }
        with self._lock:
            self.totals["requests"] += 1
            self.totals["prompt_tokens"] += len(ids)
            self.totals["reused_tokens"] += job.reused
            self.totals["generated_tokens"] += len(output)
            self.totals["session_hits"] += source == "session"
            self.last_metrics = metrics
        return RunnerResult(
            text=answer,
            reasoning=reasoning,
            reasoning_finished=reasoning_finished,
            finish_reason="stop" if stopped else "length",
            input_tokens=len(ids),
            output_tokens=len(output),
            cached_tokens=job.reused,
            dialect=family.dialect,
            metrics=metrics,
        )

    def _drain(self, job: _Job) -> None:
        while job.tokens.get() is not _DONE:
            pass

    def stats(self) -> dict[str, Any]:
        engine = self.engine.stats()
        with self._lock:
            totals = dict(self.totals)
            last = dict(self.last_metrics)
            sessions = len(self._sessions)
        prefix = engine["prefix_cache"] or {}
        return {
            "model": str(self.path),
            "model_type": self.model_type,
            "family": self.family.name,
            "dialect": self.family.dialect,
            "draft": False,
            "draft_enabled": False,
            "load_seconds": round(self.load_seconds, 2),
            "weight_bytes": self.weight_bytes,
            "cache_entries": prefix.get("entries", 0),
            "cache_bytes": prefix.get("bytes", 0),
            "sessions": sessions,
            "active_memory_bytes": self.mx.get_active_memory(),
            "max_context_tokens": self.max_context_tokens,
            "running": engine["running"],
            "waiting": engine["waiting"],
            "kv_device_bytes": engine["kv_device_bytes"],
            **totals,
            "last": last,
        }

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wake.set()
        self._thread.join(timeout=30)

    def __enter__(self) -> ChatRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _merge_system(messages: list[dict]) -> list[dict]:
    """Gemma 3's template has no system role and wants alternating turns."""
    system: list[str] = []
    turns: list[dict] = []
    for message in messages:
        if message["role"] == "system":
            if turns:
                raise ValueError(
                    "Gemma system messages must come before the conversation"
                )
            system.append(message["content"])
            continue
        content = message["content"]
        if not turns:
            if message["role"] != "user":
                raise ValueError("Gemma conversations must start with a user turn")
            content = "\n\n".join(system + [content])
        if turns and turns[-1]["role"] == message["role"]:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": message["role"], "content": content})
    if not turns:
        raise ValueError("Gemma conversations need a user message")
    return turns
