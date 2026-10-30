"""Continuous-batching MLX engine.

Every :meth:`MlxEngine.step` decodes one token for each running request in a
single batched forward pass, then prefills one chunk of a waiting prompt, so
a long new prompt delays the others by at most a chunk per step. Requests are
admitted when a sequence slot and enough K/V pages for their whole budget
are free, so a request that starts always finishes.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np

from .backends.base import mlx_build_info
from .backends.factory import load_model
from .batches import DecodeBatch, PrefillBatch
from .paged_kv import KVCache, KVGroup
from .prefix_cache import PrefixCache
from .runtime import TERMINAL, SequenceState, TokenEvent
from .sampling import GREEDY, SamplingParams, sample

PAGE_SIZE = 64


@dataclass
class Request:
    request_id: int
    prompt: np.ndarray
    max_new_tokens: int
    stop: frozenset[int]
    sampling: SamplingParams
    namespace: str
    checkpoints: tuple[int, ...]
    # (position, [tokens, hidden]) embeddings that replace image placeholders.
    images: tuple[tuple[int, Any], ...] = ()
    state: SequenceState = SequenceState.WAITING
    output: list[int] = field(default_factory=list)
    reason: str = ""
    # Positions whose K/V is written; the next token goes at this position.
    length: int = 0
    tables: dict[str, list[int]] = field(default_factory=dict)
    reserved: dict[str, int] = field(default_factory=dict)
    slot: int = -1
    forced: deque[int] = field(default_factory=deque)
    reused: int = 0
    submitted: float = field(default_factory=time.perf_counter)
    first_token: float | None = None

    @property
    def tokens(self) -> np.ndarray:
        return np.concatenate([self.prompt, np.asarray(self.output, dtype=np.int32)])


class MlxEngine:
    backend = "mlx"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        max_num_sequences: int = 8,
        max_model_length: int | None = None,
        kv_cache_bytes: int = 4 << 30,
        prefill_chunk_size: int = 512,
        prefill_burst_size: int = 2048,
        prefix_cache_bytes: int = 0,
        prefix_cache_entries: int = 24,
        page_size: int = PAGE_SIZE,
        kv_bits: int = 16,
        model: Any = None,
    ) -> None:
        if max_num_sequences <= 0 or kv_cache_bytes <= 0 or page_size <= 0:
            raise ValueError("sequences, cache bytes and page size must be positive")
        if not 0 < prefill_chunk_size <= prefill_burst_size:
            raise ValueError(
                "prefill chunk sizes must be positive, the burst no smaller"
            )
        if not 0 <= prefix_cache_bytes <= kv_cache_bytes:
            raise ValueError(
                "prefix_cache_bytes must be between zero and kv_cache_bytes"
            )
        self.model = model if model is not None else load_model(model_path)
        self.mx = mx = self.model.mx
        self.max_num_sequences = max_num_sequences
        self.max_model_length = min(
            self.model.max_position_embeddings,
            max_model_length or self.model.max_position_embeddings,
        )
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_burst_size = prefill_burst_size
        self.page_size = page_size
        groups: list[KVGroup] = list(self.model.kv_groups)
        self.kv_bits = kv_bits
        self.page_bytes = {
            g.name: g.bytes_per_token(kv_bits) * page_size for g in groups
        }
        state_groups = list(getattr(self.model, "state_groups", ()))
        snapshot_slots = (
            prefix_cache_entries if prefix_cache_bytes and state_groups else 0
        )
        self.cache = KVCache(
            mx,
            groups,
            page_size=page_size,
            capacities=self._capacities(groups, kv_cache_bytes),
            dtype=self.model.dtype,
            state_groups=state_groups,
            state_slots=(max_num_sequences + snapshot_slots) if state_groups else 0,
            bits=kv_bits,
        )
        self.prefix_cache = (
            PrefixCache(
                self.cache,
                budget_bytes=prefix_cache_bytes,
                max_entries=prefix_cache_entries,
                page_bytes=self.page_bytes,
                slot_bytes=sum(g.bytes_per_slot() for g in state_groups),
            )
            if prefix_cache_bytes
            else None
        )
        self.requests: dict[int, Request] = {}
        self.waiting: deque[int] = deque()
        self.running: list[int] = []
        self._next_id = 1
        self._closed = False
        self.totals = {
            "submitted": 0,
            "completed": 0,
            "cancelled": 0,
            "prompt_tokens": 0,
            "reused_tokens": 0,
            "generated_tokens": 0,
            "steps": 0,
        }

    def _capacities(self, groups: list[KVGroup], budget: int) -> dict[str, int]:
        """Split the byte budget into pages per group. A sliding-window group
        gets room for every sequence's window twice over (the second half for
        cached prefixes); full-attention groups share the rest by size."""
        capacities: dict[str, int] = {}
        remaining = budget
        for group in groups:
            if group.window:
                pages = 2 * self.max_num_sequences * self._window_pages(group)
                pages = min(pages, max(1, budget // 2 // self.page_bytes[group.name]))
                capacities[group.name] = pages
                remaining -= pages * self.page_bytes[group.name]
        full = [g for g in groups if not g.window]
        per_token = sum(g.bytes_per_token(self.kv_bits) for g in full)
        for group in full:
            share = remaining * group.bytes_per_token(self.kv_bits) / per_token
            capacities[group.name] = max(1, int(share // self.page_bytes[group.name]))
        return capacities

    def _window_pages(self, group: KVGroup) -> int:
        return math.ceil((group.window + self.prefill_burst_size) / self.page_size) + 1

    # --- requests -------------------------------------------------------------

    def submit(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
        *,
        sampling: SamplingParams | None = None,
        cache_namespace: str = "",
        checkpoints: Iterable[int] = (),
        images: Iterable[tuple[int, Any]] = (),
    ) -> int:
        """Queue a prompt. ``checkpoints`` are prompt lengths to keep in the
        prefix cache on the way, such as the end of a shared system prompt.
        ``images`` are ``(position, embeddings)`` for image soft tokens; use a
        cache namespace that identifies the images, since their placeholder
        tokens are all the same."""
        self._check_open()
        prompt = np.asarray(list(input_ids), dtype=np.int32)
        stop = frozenset(int(t) for t in eos_token_ids)
        vocab = self.model.vocab_size
        if prompt.ndim != 1 or prompt.size == 0:
            raise ValueError("the prompt must be a non-empty list of token ids")
        if (
            int(prompt.min()) < 0
            or int(prompt.max()) >= vocab
            or any(not 0 <= t < vocab for t in stop)
        ):
            raise ValueError("token id is outside the vocabulary")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if prompt.size + max_new_tokens > self.max_model_length:
            raise ValueError(
                f"{prompt.size} prompt tokens plus {max_new_tokens} new ones exceed "
                f"the {self.max_model_length}-token context"
            )
        marks = tuple(sorted({int(c) for c in checkpoints if 0 < int(c) < prompt.size}))
        request = Request(
            self._next_id,
            prompt,
            max_new_tokens,
            stop,
            sampling or GREEDY,
            cache_namespace,
            marks,
            tuple(sorted(images, key=lambda item: item[0])),
        )
        for at, embeddings in request.images:
            if at < 0 or at + int(embeddings.shape[0]) > prompt.size:
                raise ValueError("image embeddings fall outside the prompt")
        needs = self._needs(request, 0)
        if any(needs[name] > pool.capacity for name, pool in self.cache.pools.items()):
            raise ValueError("the request needs more K/V memory than the cache has")
        self._next_id += 1
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)
        self.totals["submitted"] += 1
        return request.request_id

    def force(self, request_id: int, tokens: Iterable[int]) -> None:
        """Emit ``tokens`` next instead of sampling (e.g. to end a thinking block)."""
        self.requests[int(request_id)].forced.extend(int(t) for t in tokens)

    def cancel(self, request_id: int) -> None:
        request = self.requests.get(int(request_id))
        if request is None or request.state in TERMINAL:
            return
        if request.request_id in self.waiting:
            self.waiting.remove(request.request_id)
        self._finish(request, SequenceState.CANCELLED, "cancelled", keep=False)

    def forget(self, request_id: int) -> None:
        request = self.requests.get(int(request_id))
        if request is not None and request.state not in TERMINAL:
            raise RuntimeError("cannot forget an active request")
        self.requests.pop(int(request_id), None)

    def request(self, request_id: int) -> Request:
        return self.requests[int(request_id)]

    def has_work(self) -> bool:
        return bool(self.running or self.waiting)

    # --- admission and pages --------------------------------------------------

    def _needs(self, request: Request, reused: int) -> dict[str, int]:
        total = request.prompt.size + request.max_new_tokens
        needs = {}
        for name, group in self.cache.groups.items():
            pages = math.ceil(total / self.page_size) - reused // self.page_size
            if group.window:
                pages = min(pages, self._window_pages(group))
            needs[name] = pages
        return needs

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_num_sequences:
            request = self.requests[self.waiting[0]]
            entry = None
            if self.prefix_cache:
                entry = self.prefix_cache.find(request.namespace, request.prompt)
            needs = self._needs(request, len(entry.tokens) if entry else 0)
            slots = 1 if self.cache.state_groups else 0
            if not self._reserve(needs, slots):
                if not (
                    self.prefix_cache
                    and self.prefix_cache.evict(pages=needs, slots=slots)
                ):
                    return
                if entry is not None and entry not in self.prefix_cache.entries:
                    continue  # Eviction dropped the entry this request matched.
                if not self._reserve(needs, slots):
                    return
            self.waiting.popleft()
            request.reserved = needs
            request.tables = {name: [] for name in self.cache.groups}
            if slots:
                request.slot = self.cache.take_slot()
                self.cache.clear_slot(request.slot)
            if entry is not None:
                self._attach(request, entry)
            request.state = SequenceState.PREFILLING
            self.running.append(request.request_id)
            self.totals["prompt_tokens"] += int(request.prompt.size)
            self.totals["reused_tokens"] += request.reused

    def _reserve(self, needs: dict[str, int], slots: int) -> bool:
        if slots and len(self.cache.state_slots) < slots:
            return False
        if any(self.cache.pools[name].available < n for name, n in needs.items()):
            return False
        for name, count in needs.items():
            self.cache.pools[name].reserve(count)
        return True

    def _attach(self, request: Request, entry: Any) -> None:
        length = len(entry.tokens)
        tail = length % self.page_size
        last = (length - 1) // self.page_size
        for name, table in entry.pages.items():
            pool = self.cache.pools[name]
            request.tables[name] = list(table)
            pool.retain(table)
            if tail:
                # The partial last page is shared; append to a private copy.
                fresh = self._take_page(request, name)
                self.cache.ensure_capacity()
                self.cache.copy_page(self.cache.groups[name], table[last], fresh, tail)
                pool.release([table[last]])
                request.tables[name][last] = fresh
        if entry.slot >= 0:
            self.cache.copy_slot(entry.slot, request.slot)
        request.length = request.reused = length

    def _take_page(self, request: Request, name: str) -> int:
        pool = self.cache.pools[name]
        if request.reserved.get(name, 0) > 0:
            request.reserved[name] -= 1
            return pool.allocate(reserved=True)
        if pool.available <= 0 and self.prefix_cache:
            self.prefix_cache.evict(pages={name: 1})
        return pool.allocate(reserved=False)

    def _ensure_pages(self, request: Request, end: int) -> None:
        """Give ``request`` pages for every position below ``end``."""
        for name, table in request.tables.items():
            while len(table) * self.page_size < end:
                table.append(self._take_page(request, name))

    def _trim(self, request: Request) -> None:
        """Release sliding-window pages that no later position can see."""
        for name, group in self.cache.groups.items():
            if not group.window:
                continue
            table = request.tables[name]
            pool = self.cache.pools[name]
            keep_from = (request.length - group.window + 1) // self.page_size
            for index in range(min(keep_from, len(table))):
                page = table[index]
                if page < 0:
                    continue
                shared = pool.shared(page)
                pool.release([page])
                table[index] = -1
                # Keep the quota: the window moves on and will need a page.
                if not shared and pool.reserve(1):
                    request.reserved[name] = request.reserved.get(name, 0) + 1

    def _release(self, request: Request) -> None:
        for name, table in request.tables.items():
            pool = self.cache.pools[name]
            pool.release(table)
            pool.unreserve(request.reserved.get(name, 0))
        request.tables, request.reserved = {}, {}
        if request.slot >= 0:
            self.cache.free_slot(request.slot)
            request.slot = -1

    def _finish(
        self, request: Request, state: SequenceState, reason: str, *, keep: bool
    ) -> None:
        if keep and self.prefix_cache and request.length:
            self.prefix_cache.insert(
                request.namespace,
                request.tokens[: request.length],
                request.tables,
                request.slot,
            )
        if request.request_id in self.running:
            self.running.remove(request.request_id)
        self._release(request)
        request.state, request.reason = state, reason
        key = "completed" if state is SequenceState.COMPLETED else "cancelled"
        self.totals[key] += 1

    # --- execution ------------------------------------------------------------

    def step(self) -> list[TokenEvent]:
        """Advance every running request by one token and prefill one chunk."""
        self._check_open()
        self._admit()
        running = [self.requests[i] for i in self.running]
        decoding = [r for r in running if r.state is SequenceState.RUNNING]
        prefilling = next(
            (r for r in running if r.state is SequenceState.PREFILLING), None
        )
        events: list[TokenEvent] = []
        try:
            if decoding:
                events += self._decode(decoding)
            if prefilling is not None:
                events += self._prefill(prefilling, burst=not decoding)
        except Exception:
            for request in running:
                if request.state not in TERMINAL:
                    self._finish(request, SequenceState.CANCELLED, "error", keep=False)
            raise
        self.totals["steps"] += 1
        return events

    def _prefill(self, request: Request, *, burst: bool) -> list[TokenEvent]:
        mx = self.mx
        start = request.length
        size = self.prefill_burst_size if burst else self.prefill_chunk_size
        end = min(request.prompt.size, start + size)
        end = next((c for c in request.checkpoints if start < c < end), end)
        for at, embeddings in request.images:
            # Never split an image: stop before it, or take all of it.
            stop = at + int(embeddings.shape[0])
            if at < end < stop:
                end = at if at > start else stop
        self._ensure_pages(request, end)
        self.cache.ensure_capacity()
        batch = PrefillBatch(
            self.cache,
            mx.array(request.prompt[start:end]),
            start=start,
            tables={
                name: np.asarray(t, dtype=np.int64)
                for name, t in request.tables.items()
            },
            slot=request.slot,
            images=request.images,
        )
        logits = self.model.forward(batch)
        request.length = end
        if end < request.prompt.size:
            mx.eval(logits, *self._arrays())
            if self.prefix_cache and end in request.checkpoints:
                self.prefix_cache.insert(
                    request.namespace,
                    request.prompt[:end],
                    request.tables,
                    request.slot,
                )
            self._trim(request)
            return []
        request.state = SequenceState.RUNNING
        token = self._choose([request], logits[-1:])[0]
        return [self._emit(request, token)]

    def _decode(self, requests: list[Request]) -> list[TokenEvent]:
        mx = self.mx
        for request in requests:
            self._ensure_pages(request, request.length + 1)
        self.cache.ensure_capacity()
        batch = DecodeBatch(
            self.cache,
            mx.array([r.output[-1] for r in requests], dtype=mx.int32),
            positions=np.array([r.length for r in requests], dtype=np.int64),
            tables={
                name: [np.asarray(r.tables[name], dtype=np.int64) for r in requests]
                for name in self.cache.groups
            },
            slots=np.array([r.slot for r in requests])
            if self.cache.state_groups
            else None,
        )
        logits = self.model.forward(batch)
        for request in requests:
            request.length += 1
        tokens = self._choose(requests, logits)
        return [self._emit(r, t) for r, t in zip(requests, tokens)]

    def _choose(self, requests: list[Request], logits: Any) -> list[int]:
        tokens = sample(
            self.mx,
            logits,
            [r.sampling for r in requests],
            [len(r.output) for r in requests],
        )
        return [r.forced.popleft() if r.forced else t for r, t in zip(requests, tokens)]

    def _arrays(self) -> list[Any]:
        cache = self.cache
        states = [a for arrays in cache.states.values() for a in arrays]
        return [*cache.keys.values(), *cache.values.values(), *states]

    def _emit(self, request: Request, token: int) -> TokenEvent:
        if request.first_token is None:
            request.first_token = time.perf_counter()
        request.output.append(token)
        self.totals["generated_tokens"] += 1
        reason = ""
        if token in request.stop:
            reason = "stop"
        elif len(request.output) >= request.max_new_tokens:
            reason = "length"
        elif request.length + 1 >= self.max_model_length:
            reason = "context"
        if reason:
            self._finish(request, SequenceState.COMPLETED, reason, keep=True)
        else:
            self._trim(request)
        return TokenEvent(request.request_id, token, bool(reason), reason)

    def generate(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
        **options: Any,
    ) -> list[int]:
        request_id = self.submit(input_ids, max_new_tokens, eos_token_ids, **options)
        while self.requests[request_id].state not in TERMINAL:
            self.step()
        request = self.requests.pop(request_id)
        if request.state is not SequenceState.COMPLETED:
            raise RuntimeError("generation ended without completing")
        return list(request.output)

    def prefill_logits(self, input_ids: Iterable[int]) -> np.ndarray:
        """Last-position logits of a prompt, outside the scheduler (for checks)."""
        self._check_open()
        request = Request(
            0,
            np.asarray(list(input_ids), dtype=np.int32),
            1,
            frozenset(),
            GREEDY,
            "",
            (),
        )
        if not 0 < request.prompt.size < self.max_model_length:
            raise ValueError("the prompt length is outside the context limit")
        needs = self._needs(request, 0)
        slots = 1 if self.cache.state_groups else 0
        if not self._reserve(needs, slots):
            raise RuntimeError("no K/V room for the prompt")
        request.reserved = needs
        request.tables = {name: [] for name in self.cache.groups}
        if slots:
            request.slot = self.cache.take_slot()
            self.cache.clear_slot(request.slot)
        try:
            for start in range(0, request.prompt.size, self.prefill_chunk_size):
                end = min(request.prompt.size, start + self.prefill_chunk_size)
                self._ensure_pages(request, end)
                self.cache.ensure_capacity()
                batch = PrefillBatch(
                    self.cache,
                    self.mx.array(request.prompt[start:end]),
                    start=start,
                    tables={
                        n: np.asarray(t, dtype=np.int64)
                        for n, t in request.tables.items()
                    },
                    slot=request.slot,
                )
                logits = self.model.forward(batch)
                self.mx.eval(logits, *self._arrays())
                request.length = end
                self._trim(request)
            return np.array(logits[-1])
        finally:
            self._release(request)

    # --- housekeeping ---------------------------------------------------------

    def clear_prefix_cache(self, cache_namespace: str | None = None) -> None:
        if self.prefix_cache:
            self.prefix_cache.clear(cache_namespace)

    def stats(self) -> dict[str, Any]:
        pools = {
            name: {
                "capacity_pages": pool.capacity,
                "used_pages": pool.used,
                "reserved_pages": pool.reserved,
                "page_bytes": self.page_bytes[name],
            }
            for name, pool in self.cache.pools.items()
        }
        return {
            "backend": "mlx",
            "model_type": self.model.model_type,
            "running": len(self.running),
            "waiting": len(self.waiting),
            "max_num_sequences": self.max_num_sequences,
            "max_model_length": self.max_model_length,
            "kv_bits": self.kv_bits,
            "kv_pools": pools,
            "kv_device_bytes": self.cache.device_bytes,
            "prefix_cache": self.prefix_cache.stats() if self.prefix_cache else None,
            **self.totals,
        }

    def build_info(self) -> dict[str, object]:
        return mlx_build_info()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("engine is closed")

    def close(self) -> None:
        if self._closed:
            return
        for request_id in list(self.running) + list(self.waiting):
            self.cancel(request_id)
        self.clear_prefix_cache()
        self.cache.keys.clear()
        self.cache.values.clear()
        self.cache.states.clear()
        self.model.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
