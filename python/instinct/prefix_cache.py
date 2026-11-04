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

Entries from one conversation share nearly all their pages, so the budget
counts each page once. A reply that a later reply extends is superseded and
is the first to go when room is needed. Checkpoints are the places a
conversation branches from (the next chat turn, a sibling sub-agent), and
they outlast replies: before this, a long tool loop's replies pushed out the
checkpoint the following turn needed, and that turn prefilled from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .paged_kv import KVCache


@dataclass(eq=False)  # Entries compare by identity, not by token arrays.
class Entry:
    namespace: str
    tokens: np.ndarray
    pages: dict[str, list[int]]
    slot: int
    checkpoint: bool
    used: int = 0
    hits: int = 0
    superseded: bool = False


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
        """Bytes the entries hold, with a page shared by several counted once."""
        total = self.slot_bytes * sum(entry.slot >= 0 for entry in self.entries)
        for name, size in self.page_bytes.items():
            total += size * int(np.count_nonzero(self._references(name)))
        return total

    def _references(self, name: str) -> np.ndarray:
        """How many entries hold each page of group ``name``."""
        tables = [
            np.asarray(entry.pages[name], dtype=np.int64)
            for entry in self.entries
            if name in entry.pages
        ]
        pages = np.concatenate(tables) if tables else np.zeros(0, dtype=np.int64)
        return np.bincount(pages[pages >= 0], minlength=self.cache.pools[name].capacity)

    def _touch(self, entry: Entry) -> None:
        self._clock += 1
        entry.used = self._clock

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
            self._touch(best)
            best.hits += 1
            self.counters["hits"] += 1
            self.counters["reused_tokens"] += len(best.tokens)
        return best

    def insert(
        self,
        namespace: str,
        tokens: np.ndarray,
        pages: dict[str, list[int]],
        slot: int,
        *,
        checkpoint: bool = False,
    ) -> None:
        """Keep ``tokens``. ``checkpoint`` marks a branch point the caller
        asked for; otherwise it's a finished request's prompt and reply."""
        if self.budget_bytes <= 0 or len(tokens) == 0:
            return
        for entry in self.entries:
            if entry.namespace == namespace and np.array_equal(entry.tokens, tokens):
                entry.checkpoint = entry.checkpoint or checkpoint
                self._touch(entry)
                return
        size = sum(
            self.page_bytes[name] * sum(page >= 0 for page in table)
            for name, table in pages.items()
        ) + (self.slot_bytes if slot >= 0 else 0)
        if size > self.budget_bytes:
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
        entry = Entry(namespace, tokens.copy(), kept, snapshot, checkpoint)
        self._touch(entry)
        if not checkpoint:
            for other in self.entries:
                n = len(other.tokens)
                if (
                    not other.checkpoint
                    and other.namespace == namespace
                    and n < len(tokens)
                    and np.array_equal(tokens[:n], other.tokens)
                ):
                    other.superseded = True
        self.entries.append(entry)
        self.counters["inserted"] += 1
        while len(self.entries) > self.max_entries or self.nbytes > self.budget_bytes:
            self._drop(self._victim(keep=entry))

    def _victim(self, keep: Entry | None = None) -> Entry:
        """Superseded replies go first, then the least recently used."""
        return min(
            (entry for entry in self.entries if entry is not keep),
            key=lambda entry: (not entry.superseded, entry.used),
        )

    def evict(self, *, pages: dict[str, int] | None = None, slots: int = 0) -> bool:
        """Drop entries until the pools can supply ``pages`` (and ``slots``
        state slots), and say whether they now can. When even dropping every
        entry wouldn't free enough, running requests hold the pages, so
        nothing is dropped."""
        pages = pages or {}

        def short() -> bool:
            if slots and len(self.cache.state_slots) < slots:
                return True
            return any(
                self.cache.pools[name].available < count
                for name, count in pages.items()
            )

        if not short():
            return True
        held_slots = sum(entry.slot >= 0 for entry in self.entries)
        if slots and len(self.cache.state_slots) + held_slots < slots:
            return False
        for name, count in pages.items():
            pool = self.cache.pools[name]
            if pool.available + pool.held_only_by(self._references(name)) < count:
                return False
        while short() and self.entries:
            self._drop(self._victim())
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
            "checkpoints": sum(entry.checkpoint for entry in self.entries),
            "superseded": sum(entry.superseded for entry in self.entries),
            "bytes": self.nbytes,
            "budget_bytes": self.budget_bytes,
            **self.counters,
        }
