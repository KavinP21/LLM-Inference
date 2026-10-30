"""Reuse of computed prompt prefixes across requests.

An entry is a token prefix whose K/V pages stay pinned (by reference count)
after the request that wrote them has finished, plus a copy of any recurrent
state at exactly that length. A new request starting with the same tokens
attaches the pages instead of computing them again. Agent loops resend the
whole conversation every step, so this is what keeps long tasks fast.

Entries match only at their exact length: sliding-window layers have released
everything before their last window and recurrent state exists only for the
final position, so a prefix can't be resumed from the middle. Entries are
made when a request finishes (its prompt and reply) and at checkpoints the
caller asks for, such as the end of a shared system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .paged_kv import KVCache


@dataclass
class Entry:
    namespace: str
    tokens: np.ndarray
    pages: dict[str, list[int]]
    slot: int
    nbytes: int
    used: int = 0
    hits: int = field(default=0)


class PrefixCache:
    def __init__(
        self,
        cache: KVCache,
        *,
        budget_bytes: int,
        max_entries: int,
        page_bytes: dict[str, int],
        slot_bytes: int,
    ) -> None:
        self.cache = cache
        self.budget_bytes = budget_bytes
        self.max_entries = max_entries
        self.page_bytes = page_bytes
        self.slot_bytes = slot_bytes
        self.entries: list[Entry] = []
        self._clock = 0
        self.counters = {
            "lookups": 0,
            "hits": 0,
            "reused_tokens": 0,
            "inserted": 0,
            "evicted": 0,
        }

    @property
    def nbytes(self) -> int:
        return sum(entry.nbytes for entry in self.entries)

    def find(self, namespace: str, prompt: np.ndarray) -> Entry | None:
        """The longest entry that is a proper prefix of ``prompt``."""
        self.counters["lookups"] += 1
        best = None
        for entry in self.entries:
            n = len(entry.tokens)
            if (
                entry.namespace == namespace
                and n < len(prompt)
                and (best is None or n > len(best.tokens))
                and np.array_equal(prompt[:n], entry.tokens)
            ):
                best = entry
        if best is not None:
            self._clock += 1
            best.used = self._clock
            best.hits += 1
            self.counters["hits"] += 1
            self.counters["reused_tokens"] += len(best.tokens)
        return best

    def insert(
        self, namespace: str, tokens: np.ndarray, pages: dict[str, list[int]], slot: int
    ) -> None:
        if self.budget_bytes <= 0 or len(tokens) == 0:
            return
        for entry in self.entries:
            if entry.namespace == namespace and np.array_equal(entry.tokens, tokens):
                self._clock += 1
                entry.used = self._clock
                return
        nbytes = sum(
            self.page_bytes[name] * sum(page >= 0 for page in table)
            for name, table in pages.items()
        ) + (self.slot_bytes if slot >= 0 else 0)
        if nbytes > self.budget_bytes:
            return
        snapshot = -1
        if slot >= 0:
            if not self.cache.state_slots:
                self.evict(slots=1)
            if not self.cache.state_slots:
                return
            snapshot = self.cache.take_slot()
            self.cache.copy_slot(slot, snapshot)
        kept = {name: list(table) for name, table in pages.items()}
        for name, table in kept.items():
            self.cache.pools[name].retain(table)
        self._clock += 1
        self.entries.append(
            Entry(namespace, tokens.copy(), kept, snapshot, nbytes, self._clock)
        )
        self.counters["inserted"] += 1
        while len(self.entries) > self.max_entries or self.nbytes > self.budget_bytes:
            self._drop(min(self.entries, key=lambda e: e.used))

    def evict(self, *, pages: dict[str, int] | None = None, slots: int = 0) -> bool:
        """Drop least-recently-used entries until the pools can supply ``pages``
        (and ``slots`` state slots). Returns whether they now can."""

        def short() -> bool:
            if slots and len(self.cache.state_slots) < slots:
                return True
            return any(
                self.cache.pools[name].available < count
                for name, count in (pages or {}).items()
            )

        while short() and self.entries:
            self._drop(min(self.entries, key=lambda e: e.used))
        return not short()

    def clear(self, namespace: str | None = None) -> None:
        for entry in [
            e for e in self.entries if namespace is None or e.namespace == namespace
        ]:
            self._drop(entry)

    def _drop(self, entry: Entry) -> None:
        self.entries.remove(entry)
        for name, table in entry.pages.items():
            self.cache.pools[name].release(table)
        self.cache.free_slot(entry.slot)
        self.counters["evicted"] += 1

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self.entries),
            "bytes": self.nbytes,
            "budget_bytes": self.budget_bytes,
            **self.counters,
        }
