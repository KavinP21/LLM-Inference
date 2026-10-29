"""Chat runner for MLX-VLM checkpoints (Gemma 4, Qwen3.5 and anything else
mlx-vlm loads), tuned for agent loops.

Generation itself is mlx-vlm's ``stream_generate``. What this module adds is
prompt reuse. Agent turns re-send the whole conversation, so without reuse
every step re-prefills thousands of tokens:

* Sessions. After each reply the KV cache and the exact token stream are
  kept. If the next request is the same conversation plus new messages, only
  the new messages' tokens are rendered and prefilled. Because the token
  stream is extended rather than re-rendered, this also works for caches that
  can't be rolled back (sliding windows, recurrent state).
* Snapshots. For a prompt that doesn't continue a session, the cache is
  copied at the last stable message boundary (everything before the final
  user/tool message). Later prompts that start with the same tokens, such as
  the next chat message or a sibling sub-agent with the same system prompt,
  restore the copy instead of prefilling it again.
* Adaptive drafting. With a draft model (Gemma 4's MTP assistant), drafting
  is used for greedy agent turns and switched off for a while when its
  measured acceptance drops below break-even.
"""

from __future__ import annotations

import copy
import hashlib
import json
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

# Below this many accepted tokens per round, a block-3 MTP round costs more
# than plain decoding on Gemma 4 26B A4B (measured on an M3 Max).
DRAFT_BREAK_EVEN = 2.25
DRAFT_COOLDOWN_REQUESTS = 6
MIN_SNAPSHOT_TOKENS = 256


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
    state: Any  # mlx_vlm PromptCacheState
    stop_text: str
    # Stop token the drafting loop generated but never fed back into the cache.
    pending: list[int]
    media_kwargs: dict
    nbytes: int


@dataclass
class _Snapshot:
    tokens: tuple[int, ...]
    cache: list
    nbytes: int


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


def _cache_bytes(cache: Sequence[Any]) -> int:
    total = 0
    for entry in cache:
        state = getattr(entry, "state", None)
        arrays = state if isinstance(state, (list, tuple)) else [state]
        total += sum(getattr(a, "nbytes", 0) for a in arrays if a is not None)
    return total


class VlmRunner:
    """Single-threaded; the owning worker serializes calls."""

    def __init__(
        self,
        path: str | Path,
        *,
        draft_path: str | Path | None = None,
        draft_block_size: int = 3,
        max_context_tokens: int = 65536,
        prefill_step_size: int = 1024,
        cache_budget_bytes: int = 3 << 30,
        max_entries: int = 12,
    ) -> None:
        import mlx.core as mx
        from mlx_vlm.utils import load

        if not mx.metal.is_available():
            raise RuntimeError("the MLX-VLM runner needs a Metal GPU")
        self.mx = mx
        self.path = Path(path)
        load_started = time.perf_counter()
        self.model, self.processor = load(str(self.path))
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        config = self.model.config
        self.model_type = str(getattr(config, "model_type", ""))
        self.family: ChatFamily = family_for(self.model_type)
        self.draft_model = None
        if draft_path is not None:
            from mlx_vlm.speculative.drafters import load_drafter

            self.draft_model, self.draft_kind = load_drafter(
                str(draft_path), kind="mtp"
            )
        self.draft_block_size = draft_block_size
        self.max_context_tokens = max_context_tokens
        self.prefill_step_size = prefill_step_size
        self.cache_budget_bytes = cache_budget_bytes
        self.max_entries = max_entries
        self.load_seconds = time.perf_counter() - load_started
        self.weight_bytes = mx.get_active_memory()
        self._sessions: OrderedDict[int, _Session] = OrderedDict()
        self._snapshots: OrderedDict[int, _Snapshot] = OrderedDict()
        self._next_key = 0
        self._draft_ewma = DRAFT_BREAK_EVEN + 0.25
        self._draft_cooldown = 0
        self._lock = threading.Lock()
        self.totals = {
            "requests": 0,
            "prompt_tokens": 0,
            "reused_tokens": 0,
            "generated_tokens": 0,
            "session_hits": 0,
            "snapshot_hits": 0,
        }
        self.last_metrics: dict[str, Any] = {}

    # --- prompt construction -------------------------------------------------

    def _template_messages(self, messages: Sequence[dict]) -> list[dict]:
        converted = []
        for message in messages:
            media = message.get("media", ())
            if media:
                parts = [{"type": item["type"]} for item in media]
                parts.append({"type": "text", "text": message.get("content", "")})
                converted.append({"role": message["role"], "content": parts})
            else:
                converted.append(
                    {"role": message["role"], "content": message.get("content", "")}
                )
        if self.family.name == "gemma4":
            converted = to_gemma_messages(converted)
        return converted

    def render(
        self, messages: Sequence[dict], tools, thinking: bool, generation_prompt=True
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

    # --- cache bookkeeping ---------------------------------------------------

    def _evict(self) -> None:
        def used() -> int:
            return sum(s.nbytes for s in self._sessions.values()) + sum(
                s.nbytes for s in self._snapshots.values()
            )

        while (
            len(self._sessions) + len(self._snapshots) > self.max_entries
            or used() > self.cache_budget_bytes
        ) and (self._sessions or self._snapshots):
            # Drop whichever least-recently-used entry is older.
            oldest_session = next(iter(self._sessions), None)
            oldest_snapshot = next(iter(self._snapshots), None)
            if oldest_snapshot is None or (
                oldest_session is not None and oldest_session < oldest_snapshot
            ):
                self._sessions.popitem(last=False)
            else:
                self._snapshots.popitem(last=False)

    def _key(self) -> int:
        self._next_key += 1
        return self._next_key

    def _fresh_cache(self) -> list:
        language = self.model.language_model
        if hasattr(language, "make_cache"):
            return list(language.make_cache())
        from mlx_vlm.models.cache import make_prompt_cache

        return list(make_prompt_cache(language))

    def _snapshot(self, cache: list) -> list:
        mx = self.mx
        saved = []
        for entry in cache:
            state = entry.state
            many = isinstance(state, (list, tuple))
            arrays = [
                mx.array(a) if a is not None else None
                for a in (state if many else [state])
            ]
            saved.append((type(entry), arrays, many, copy.copy(entry.__dict__)))
        mx.eval([a for _, arrays, _, _ in saved for a in arrays if a is not None])
        return saved

    def _restore(self, saved: list) -> list:
        mx = self.mx
        cache = []
        for cls, arrays, many, attributes in saved:
            entry = cls.__new__(cls)
            entry.__dict__.update(copy.copy(attributes))
            copies = [mx.array(a) if a is not None else None for a in arrays]
            entry.state = copies if many else copies[0]
            cache.append(entry)
        return cache

    def _prime_positions(self) -> None:
        """Qwen-style models keep rotary-position state between calls; reset
        it so a restored text-only prefix continues at the cache offset."""
        language = getattr(self.model, "language_model", None)
        if language is not None and hasattr(language, "_rope_deltas"):
            language._position_ids = None
            language._rope_deltas = self.mx.zeros((1, 1), dtype=self.mx.int32)

    def _prefill(self, cache: list, tokens: Sequence[int], cancelled) -> None:
        mx = self.mx
        from mlx_vlm.generate.common import generation_stream

        language = self.model.language_model
        for start in range(0, len(tokens), self.prefill_step_size):
            if cancelled():
                raise InterruptedError("generation cancelled")
            chunk = mx.array([list(tokens[start : start + self.prefill_step_size])])
            with mx.stream(generation_stream):
                language(chunk, cache=cache)
            mx.eval([entry.state for entry in cache])

    def _stable_boundary(self, messages, tools, thinking, ids) -> int:
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
        limit = min(len(probe), len(full))
        while shared < limit and probe[shared] == full[shared]:
            shared += 1
        prefix = self._encode(full[:shared])
        boundary = 0
        for a, b in zip(prefix, ids):
            if a != b:
                break
            boundary += 1
        # Leave the token at the cut out; it may merge differently next time.
        return max(0, boundary - 1)

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
        cancelled: Callable[[], bool] = lambda: False,
        on_text: Callable[[str], None] | None = None,
        max_context_tokens: int | None = None,
    ) -> RunnerResult:
        limit = min(
            max_context_tokens or self.max_context_tokens, self.max_context_tokens
        )
        with self._lock:
            return self._generate(
                list(messages),
                max_tokens=max_tokens,
                tools=tools,
                effort=effort,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                cancelled=cancelled,
                on_text=on_text,
                context_limit=limit,
            )

    def _generate(
        self,
        messages,
        *,
        max_tokens,
        tools,
        effort,
        temperature,
        top_p,
        top_k,
        cancelled,
        on_text,
        context_limit,
    ) -> RunnerResult:
        mx = self.mx
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.generate.common import PromptCacheState
        from mlx_vlm.utils import prepare_inputs

        started = time.perf_counter()
        budget = thinking_budget(effort)
        thinking = budget is not None
        if temperature is None:
            # Greedy without thinking keeps tool calls deterministic and lets the
            # drafter help; with thinking, use the model card's sampling.
            temperature = self.family.temperature if thinking else 0.0
        top_p = self.family.top_p if top_p is None else top_p
        top_k = self.family.top_k if top_k is None else top_k
        identity = [_identity(m) for m in messages]
        render_key = hashlib.sha256(
            json.dumps([tools, thinking], sort_keys=True, default=str).encode()
        ).hexdigest()
        full_text = self.render(messages, tools, thinking)

        source, session, media_kwargs = "cold", None, {}
        new_media = []
        ids: list[int] | None = None
        for key in reversed(self._sessions):
            candidate = self._sessions[key]
            if candidate.render_key != render_key or len(candidate.messages) >= len(
                identity
            ):
                continue
            if identity[: len(candidate.messages)] != candidate.messages:
                continue
            if any(m.get("media") for m in messages[len(candidate.messages) :]):
                break
            previous = [dict(m) for m in messages[: len(candidate.messages)]]
            previous[-1] = {
                **previous[-1],
                "content": candidate.messages[-1]["content"],
            }
            suffix = incremental_suffix(
                lambda ms, gp: self.render(ms, tools, thinking, gp),
                previous,
                messages,
                candidate.stop_text,
            )
            if suffix is None:
                break
            session = self._sessions.pop(key)
            ids = list(session.state.token_ids) + session.pending + self._encode(suffix)
            media_kwargs, source = session.media_kwargs, "session"
            break

        pixel_values = None
        if ids is None:
            new_media = [item for m in messages for item in m.get("media", ())]
            state = PromptCacheState()
            if new_media:
                inputs = prepare_inputs(
                    self.processor,
                    images=[i["path"] for i in new_media if i["type"] == "image"]
                    or None,
                    videos=[i["path"] for i in new_media if i["type"] == "video"]
                    or None,
                    prompts=full_text,
                    image_token_index=getattr(
                        self.model.config, "image_token_index", None
                    ),
                    add_special_tokens=False,
                )
                ids = inputs["input_ids"].flatten().tolist()
                pixel_values = inputs.get("pixel_values")
                extra = {
                    k: v
                    for k, v in inputs.items()
                    if k not in ("input_ids", "pixel_values", "attention_mask")
                }
                media_kwargs = {
                    k: v
                    for k, v in extra.items()
                    if k in ("image_grid_thw", "video_grid_thw")
                }
            else:
                ids = self._encode(full_text)
                extra = {}
        else:
            state = session.state
            extra = {}

        if len(ids) + max_tokens > context_limit:
            raise ValueError(
                f"prompt has {len(ids)} tokens; with {max_tokens} output tokens it exceeds the "
                f"{context_limit}-token context"
            )

        if source == "cold" and pixel_values is None:
            # Restore the longest snapshot that prefixes this prompt, then
            # prefill up to the stable boundary and snapshot that too.
            best = None
            for key, snapshot in self._snapshots.items():
                n = len(snapshot.tokens)
                if n < len(ids) and (best is None or n > len(best[1].tokens)):
                    if tuple(ids[:n]) == snapshot.tokens:
                        best = (key, snapshot)
            cache_tokens: list[int] = []
            restored = 0
            if best is not None:
                self._snapshots.move_to_end(best[0])
                cache = self._restore(best[1].cache)
                cache_tokens = list(best[1].tokens)
                restored = len(cache_tokens)
                source = "snapshot"
            else:
                cache = self._fresh_cache()
            boundary = self._stable_boundary(messages, tools, thinking, ids)
            if boundary - len(cache_tokens) >= MIN_SNAPSHOT_TOKENS:
                self._prime_positions()
                self._prefill(cache, ids[len(cache_tokens) : boundary], cancelled)
                cache_tokens = ids[:boundary]
                saved = self._snapshot(cache)
                self._snapshots[self._key()] = _Snapshot(
                    tuple(cache_tokens), saved, _cache_bytes(cache)
                )
                self._evict()
            if cache_tokens:
                state.cache, state.token_ids = cache, list(cache_tokens)
            reused = restored
        else:
            reused = len(state.token_ids) if state.token_ids else 0

        kwargs: dict[str, Any] = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            prefill_step_size=self.prefill_step_size,
            prompt_cache_state=state,
        )
        kwargs.update(extra)
        kwargs.update(media_kwargs)
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if thinking:
            kwargs.update(enable_thinking=True)
            if budget:
                kwargs.update(
                    thinking_budget=budget,
                    thinking_start_token=self.family.think_open.split("thought")[
                        0
                    ].strip(),
                    thinking_end_token=self.family.think_close,
                )
        # Drafting pays off on tool calls, edits and code (+5-12% on Gemma 4
        # 26B A4B) and costs ~14% on free prose, so it's only used for agent
        # turns, and switched off for a while when acceptance drops.
        use_draft = (
            self.draft_model is not None
            and bool(tools)
            and temperature == 0
            and not thinking
            and self._draft_cooldown == 0
        )
        if self._draft_cooldown:
            self._draft_cooldown -= 1
            if self._draft_cooldown == 0:
                self._draft_ewma = DRAFT_BREAK_EVEN + 0.25
        draft_before = None
        if use_draft:
            from mlx_vlm.speculative.common import speculative_stats_snapshot

            draft_before = speculative_stats_snapshot(self.draft_model)
            kwargs.update(
                draft_model=self.draft_model,
                draft_kind=self.draft_kind,
                draft_block_size=self.draft_block_size,
            )

        prompt_opens_thinking = thinking and full_text.rstrip().endswith(
            self.family.think_open.strip()
        )
        pieces: list[str] = []
        first_token_at = None
        result = None
        completed = False
        try:
            for result in stream_generate(
                self.model, self.processor, "", input_ids=mx.array([ids]), **kwargs
            ):
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                if result.text:
                    pieces.append(result.text)
                    if on_text is not None:
                        on_text(result.text)
                if cancelled():
                    raise InterruptedError("generation cancelled")
            completed = True
        finally:
            if not completed and session is None:
                # The cache now holds tokens the session record doesn't; never reuse it.
                state.cache = state.token_ids = None
        finished = time.perf_counter()
        if result is None:
            raise RuntimeError("model produced no output")

        raw = "".join(pieces)
        stopped = result.finish_reason == "stop"
        stop_text, pending = "", []
        if state.token_ids and state.cache is not None:
            offset = max(int(getattr(c, "offset", 0) or 0) for c in state.cache)
            overshoot = offset - len(state.token_ids)
            # A drafting round can write a token past the last accepted one.
            # The buffered caches it uses exist to make that tail trimmable.
            if overshoot > 0 and all(c.is_trimmable() for c in state.cache):
                for c in state.cache:
                    c.trim(overshoot)
                offset -= overshoot
            if offset == len(state.token_ids) - 1:
                # The last token was sampled but never fed back; the next turn
                # feeds it first.
                pending = [state.token_ids[-1]]
                state.token_ids = state.token_ids[:-1]
            if offset == len(state.token_ids):
                if stopped:
                    stop_text = self.tokenizer.decode(
                        [(pending or state.token_ids)[-1]]
                    )
                else:
                    # Cut off by the length limit: close the turn the way the
                    # template would before appending the next messages.
                    stop_text = self.family.turn_end
                    pending = pending + self._encode(stop_text)
        answer, reasoning, reasoning_finished = split_thinking(
            (self.family.think_open + raw) if prompt_opens_thinking else raw,
            self.family,
        )
        if stop_text and state.cache is not None:
            entry = _Session(
                identity + [{"role": "assistant", "content": answer, "media": []}],
                render_key,
                state,
                stop_text,
                pending,
                media_kwargs,
                _cache_bytes(state.cache),
            )
            self._sessions[self._key()] = entry
            self._evict()

        draft = None
        if draft_before is not None:
            from mlx_vlm.speculative.common import speculative_stats_since

            rounds, accepted, drafted = speculative_stats_since(
                self.draft_model, draft_before
            )
            if rounds:
                per_round = (accepted + rounds) / rounds
                self._draft_ewma = 0.7 * self._draft_ewma + 0.3 * per_round
                if self._draft_ewma < DRAFT_BREAK_EVEN:
                    self._draft_cooldown = DRAFT_COOLDOWN_REQUESTS
                draft = {"rounds": rounds, "tokens_per_round": round(per_round, 2)}

        prompt_tokens = len(ids)
        prefilled = prompt_tokens - reused
        ttft = (first_token_at or finished) - started
        decode_seconds = max(finished - (first_token_at or finished), 1e-9)
        self.totals["requests"] += 1
        self.totals["prompt_tokens"] += prompt_tokens
        self.totals["reused_tokens"] += reused
        self.totals["generated_tokens"] += result.generation_tokens
        if source == "session":
            self.totals["session_hits"] += 1
        elif source == "snapshot":
            self.totals["snapshot_hits"] += 1
        metrics = {
            "cache_source": source,
            "prompt_tokens": prompt_tokens,
            "reused_tokens": reused,
            "prefilled_tokens": prefilled,
            "ttft_ms": round(ttft * 1000, 1),
            "prefill_tps": round(prefilled / ttft, 1)
            if prefilled and ttft > 0
            else None,
            "decode_tps": round(
                max(result.generation_tokens - 1, 0) / decode_seconds, 1
            ),
            "output_tokens": result.generation_tokens,
            "total_ms": round((finished - started) * 1000, 1),
            "peak_memory_gb": round(result.peak_memory, 2),
            "effort": effort,
            "temperature": temperature,
            "draft": draft,
        }
        self.last_metrics = metrics
        return RunnerResult(
            text=answer,
            reasoning=reasoning,
            reasoning_finished=reasoning_finished,
            finish_reason="stop" if stopped else "length",
            input_tokens=prompt_tokens,
            output_tokens=result.generation_tokens,
            cached_tokens=reused,
            dialect=self.family.dialect,
            metrics=metrics,
        )

    def stats(self) -> dict[str, Any]:
        return {
            "model": str(self.path),
            "model_type": self.model_type,
            "family": self.family.name,
            "dialect": self.family.dialect,
            "draft": self.draft_model is not None,
            "draft_enabled": self.draft_model is not None and self._draft_cooldown == 0,
            "load_seconds": round(self.load_seconds, 2),
            "weight_bytes": self.weight_bytes,
            "cache_entries": len(self._sessions) + len(self._snapshots),
            "cache_bytes": sum(s.nbytes for s in self._sessions.values())
            + sum(s.nbytes for s in self._snapshots.values()),
            "active_memory_bytes": self.mx.get_active_memory(),
            "max_context_tokens": self.max_context_tokens,
            **self.totals,
            "last": self.last_metrics,
        }

    def clear(self) -> None:
        self._sessions.clear()
        self._snapshots.clear()
        self.mx.clear_cache()

    def close(self) -> None:
        self.clear()
        self.model = self.processor = self.draft_model = None

    def __enter__(self) -> VlmRunner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
