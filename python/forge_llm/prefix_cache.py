"""Bounded engine-local FP16 prefix snapshots and reference-counted KV ownership.

No MLX import: admission, pinning, COW reservations and eviction are portable.
Cache entries contain exact token tuples, never a collision-prone hash alone.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .runtime import KVBlockPool


@dataclass
class _SharedAllocation:
    reservation: int
    live_tokens: int = 0
    blocks: list[int] = field(default_factory=list)
    cow_slots: set[int] = field(default_factory=set)


class SharedKVBlockPool(KVBlockPool):
    """Count unique pages plus pending private/COW capacity, not logical tables.

    Cache pins and requests are independent references. Only the last release
    returns an ID to the free list. Admission reserves future COW before work is
    accepted; a cache snapshot that would overcommit is refused instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._refs: dict[int, int] = {}
        self._pins: dict[int, dict[int, int]] = {}
        self.cow_copies = 0

    def reference_count(self, block: int) -> int:
        return self._refs.get(block, 0)

    def is_referenced(self, block: int) -> bool:
        return block in self._refs

    def is_shared(self, block: int) -> bool:
        return block in self._pins or self.reference_count(block) > 1

    def committed_blocks(self) -> int:
        return len(self._refs) + sum(
            a.reservation - len(a.blocks) + len(a.cow_slots)
            for a in self.allocations.values()
        )

    def admission_blocks(self, maximum_tokens, blocks=(), tokens=0) -> int:
        if maximum_tokens <= 0 or not 0 <= tokens <= maximum_tokens:
            raise ValueError("invalid shared reservation length")
        if len(blocks) != self._blocks_for(tokens) or len(set(blocks)) != len(blocks):
            raise ValueError("invalid shared prefix table")
        if any(b not in self._pins for b in blocks):
            raise ValueError("shared prefix is not cache-pinned")
        if blocks:
            final_slots = tokens % self.block_tokens or self.block_tokens
            if any(
                max(self._pins[b])
                < (final_slots if i == len(blocks) - 1 else self.block_tokens)
                for i, b in enumerate(blocks)
            ):
                raise ValueError("shared prefix exceeds cached token coverage")
        return (
            self._blocks_for(maximum_tokens)
            - len(blocks)
            + int(bool(tokens % self.block_tokens))
        )

    def reserve(self, request_id: int, maximum_tokens: int) -> bool:
        return self.reserve_shared(request_id, maximum_tokens)

    def reserve_shared(
        self, request_id: int, maximum_tokens: int, blocks=(), tokens=0
    ) -> bool:
        if request_id in self.allocations:
            raise ValueError("duplicate shared KV reservation")
        required = self.admission_blocks(maximum_tokens, blocks, tokens)
        if self.committed_blocks() + required > self.total_blocks:
            self.allocation_failures += 1
            return False
        allocation = _SharedAllocation(
            self._blocks_for(maximum_tokens), tokens, list(blocks)
        )
        if tokens % self.block_tokens:
            allocation.cow_slots.add(len(blocks) - 1)
        for block in blocks:
            self._refs[block] += 1
        self.allocations[request_id] = allocation
        return True

    def ensure_tokens(self, request_id: int, live_tokens: int) -> None:
        allocation = self.allocations[request_id]
        needed = self._blocks_for(live_tokens)
        if live_tokens < allocation.live_tokens or needed > allocation.reservation:
            raise ValueError("request violated its shared KV reservation")
        additional = needed - len(allocation.blocks)
        if additional > len(self.free_blocks):
            self.allocation_failures += 1
            raise RuntimeError("reserved shared KV allocation unexpectedly failed")
        for _ in range(additional):
            block = self.free_blocks.pop()
            self._refs[block] = 1
            allocation.blocks.append(block)
        allocation.live_tokens = live_tokens
        self.peak_allocated_blocks = max(self.peak_allocated_blocks, len(self._refs))

    def _drop(self, block: int) -> bool:
        self._refs[block] -= 1
        if self._refs[block] == 0:
            del self._refs[block]
            self.free_blocks.append(block)
            return True
        if not self.is_shared(block):
            for allocation in self.allocations.values():
                for slot in tuple(allocation.cow_slots):
                    if allocation.blocks[slot] == block:
                        allocation.cow_slots.remove(slot)
        return False

    def release(self, request_id: int) -> tuple[int, ...]:
        allocation = self.allocations.pop(request_id, None)
        if allocation is None:
            return ()
        return tuple(b for b in allocation.blocks if self._drop(b))

    def pin(self, blocks: tuple[int, ...], tokens: int) -> bool:
        if (
            tokens <= 0
            or len(blocks) != self._blocks_for(tokens)
            or len(set(blocks)) != len(blocks)
        ):
            raise ValueError("invalid cached prefix coverage")
        if any(not self.is_referenced(b) for b in blocks):
            raise ValueError("cannot pin unowned KV pages")
        partial = tokens % self.block_tokens
        additional = []
        if partial:
            tail = blocks[-1]
            additional = [
                (a, i)
                for a in self.allocations.values()
                for i, b in enumerate(a.blocks)
                if b == tail and i not in a.cow_slots
            ]
        if self.committed_blocks() + len(additional) > self.total_blocks:
            return False
        for allocation, slot in additional:
            allocation.cow_slots.add(slot)
        for i, block in enumerate(blocks):
            count = partial if i == len(blocks) - 1 and partial else self.block_tokens
            pins = self._pins.setdefault(block, {})
            pins[count] = pins.get(count, 0) + 1
            self._refs[block] += 1
        return True

    def unpin(self, blocks: tuple[int, ...], tokens: int) -> tuple[int, ...]:
        # Validate the entire operation before mutating any reference.
        if (
            tokens <= 0
            or len(blocks) != self._blocks_for(tokens)
            or len(set(blocks)) != len(blocks)
        ):
            raise ValueError("invalid cache release coverage")
        counts = [
            min(self.block_tokens, tokens - i * self.block_tokens)
            for i in range(len(blocks))
        ]
        if any(self._pins.get(b, {}).get(n, 0) <= 0 for b, n in zip(blocks, counts)):
            raise ValueError("cache pin was already released")
        freed = []
        for block, count in zip(blocks, counts):
            pins = self._pins[block]
            pins[count] -= 1
            if pins[count] == 0:
                del pins[count]
            if not pins:
                del self._pins[block]
            if self._drop(block):
                freed.append(block)
        return tuple(freed)

    def make_writable(
        self, request_id: int, start: int, count: int, clone: Callable[[int, int], None]
    ) -> None:
        allocation = self.allocations[request_id]
        if start < 0 or count <= 0 or start + count > allocation.live_tokens:
            raise ValueError("write exceeds the allocated live range")
        for slot in range(
            start // self.block_tokens, math.ceil((start + count) / self.block_tokens)
        ):
            source = allocation.blocks[slot]
            if self.is_shared(source):
                if slot not in allocation.cow_slots:
                    raise RuntimeError("attempted to rewrite a shared immutable prefix")
                if not self.free_blocks:
                    raise RuntimeError("reserved COW page unexpectedly unavailable")
                target = self.free_blocks.pop()
                try:
                    clone(source, target)
                except Exception:
                    self.free_blocks.append(target)
                    raise
                self._refs[target] = 1
                allocation.blocks[slot] = target
                self._drop(source)
                self.cow_copies += 1
                self.peak_allocated_blocks = max(
                    self.peak_allocated_blocks, len(self._refs)
                )
            allocation.cow_slots.discard(slot)

    def reset_peak(self) -> None:
        self.peak_allocated_blocks = len(self._refs)

    def stats(self) -> dict:
        active_ids = {b for a in self.allocations.values() for b in a.blocks}
        slots = {b: max(pins) for b, pins in self._pins.items()}
        logical_tokens = sum(a.live_tokens for a in self.allocations.values())
        for allocation in self.allocations.values():
            for i, block in enumerate(allocation.blocks):
                count = max(
                    0,
                    min(
                        self.block_tokens,
                        allocation.live_tokens - i * self.block_tokens,
                    ),
                )
                slots[block] = max(slots.get(block, 0), count)
        allocated = len(self._refs)
        pending = self.committed_blocks() - allocated
        return {
            "layout": self.layout,
            "total_blocks": self.total_blocks,
            "allocated_blocks": allocated,
            "peak_allocated_blocks": self.peak_allocated_blocks,
            "active_allocated_blocks": len(active_ids),
            "active_block_references": sum(
                len(a.blocks) for a in self.allocations.values()
            ),
            "cached_blocks": len(self._pins),
            "cache_only_blocks": len(set(self._pins) - active_ids),
            "shared_blocks": sum(n > 1 for n in self._refs.values()),
            "reserved_blocks": pending,
            "cow_reserved_blocks": sum(
                len(a.cow_slots) for a in self.allocations.values()
            ),
            "free_blocks": len(self.free_blocks),
            "unreserved_blocks": self.total_blocks - self.committed_blocks(),
            "live_tokens": logical_tokens,
            "physical_live_tokens": sum(slots.values()),
            "internal_fragmentation_tokens": allocated * self.block_tokens
            - sum(slots.values()),
            "allocation_failures": self.allocation_failures,
            "occupancy": allocated / self.total_blocks,
            "cow_copies": self.cow_copies,
        }


@dataclass(frozen=True)
class PrefixEntry:
    namespace: str
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    logits: np.ndarray


class PrefixCache:
    """LRU bounded by unique cached physical pages AND entry count.

    Only exact prompts or original chunk boundaries are reusable, preserving
    FP16 prefill GEMM shapes. Last-row logits are copied to bounded host memory
    so a view cannot retain the entire chunk's GPU logits tensor.
    """

    def __init__(
        self,
        pool: SharedKVBlockPool,
        store,
        *,
        max_blocks: int,
        max_entries: int,
        chunk_size: int,
        identity: str,
    ):
        if (
            min(max_blocks, max_entries, chunk_size) <= 0
            or max_blocks > pool.total_blocks
        ):
            raise ValueError("invalid prefix cache limits")
        self.pool, self.store = pool, store
        self.max_blocks, self.max_entries = max_blocks, max_entries
        self.chunk_size, self.identity = chunk_size, identity
        self.entries: OrderedDict[tuple[str, tuple[int, ...]], PrefixEntry] = (
            OrderedDict()
        )
        self.counters = {
            name: 0
            for name in (
                "hits",
                "misses",
                "matched_tokens",
                "reused_tokens",
                "insertions",
                "evictions",
                "capacity_skips",
                "admission_rejections",
            )
        }

    @staticmethod
    def validate_namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or len(namespace.encode("utf-8")) > 1024:
            raise ValueError(
                "cache namespace must be a string of at most 1024 UTF-8 bytes"
            )

    def find(self, tokens: list[int], namespace: str) -> PrefixEntry | None:
        self.validate_namespace(namespace)
        prompt = tuple(tokens)
        best = None
        for (scope, prefix), entry in self.entries.items():
            if scope != namespace or len(prefix) > len(prompt):
                continue
            if len(prefix) != len(prompt) and len(prefix) % self.chunk_size:
                continue
            if prompt[: len(prefix)] == prefix and (
                best is None or len(prefix) > len(best.tokens)
            ):
                best = entry
        if best is not None:
            self.entries.move_to_end((best.namespace, best.tokens))
        return best

    def record_lookup(self, entry: PrefixEntry | None) -> None:
        self.counters["hits" if entry else "misses"] += 1
        if entry:
            self.counters["matched_tokens"] += len(entry.tokens)

    def _evict(self, key) -> None:
        entry = self.entries.pop(key)
        self.store.release(self.pool.unpin(entry.blocks, len(entry.tokens)))
        self.counters["evictions"] += 1

    def make_admission_room(
        self, maximum_tokens: int, entry: PrefixEntry | None
    ) -> None:
        blocks, tokens = (entry.blocks, len(entry.tokens)) if entry else ((), 0)
        required = self.pool.admission_blocks(maximum_tokens, blocks, tokens)
        protected = (entry.namespace, entry.tokens) if entry else None
        for key in tuple(self.entries):
            if self.pool.committed_blocks() + required <= self.pool.total_blocks:
                break
            if key != protected:
                self._evict(key)

    def put(
        self, tokens: list[int], namespace: str, blocks: tuple[int, ...], logits
    ) -> bool:
        self.validate_namespace(namespace)
        key = (namespace, tuple(tokens))
        if key in self.entries:
            self.entries.move_to_end(key)
            return False
        required = math.ceil(len(tokens) / self.pool.block_tokens)
        if required == 0 or required > self.max_blocks:
            self.counters["capacity_skips"] += 1
            return False
        blocks = blocks[:required]
        self.store.validate_prefix(blocks, len(tokens))
        row = np.array(logits, dtype=np.float32, copy=True)
        if row.ndim != 1 or not np.isfinite(row).all():
            raise ValueError("prefix logits must be a finite row")
        while True:
            union = {b for e in self.entries.values() for b in e.blocks} | set(blocks)
            fits = (
                len(self.entries) < self.max_entries and len(union) <= self.max_blocks
            )
            if fits and self.pool.pin(blocks, len(tokens)):
                break
            if not self.entries:
                self.counters["capacity_skips"] += 1
                return False
            self._evict(next(iter(self.entries)))
        row.flags.writeable = False
        self.entries[key] = PrefixEntry(namespace, key[1], blocks, row)
        self.counters["insertions"] += 1
        return True

    def clear(self, namespace: str | None = None) -> None:
        if namespace is not None:
            self.validate_namespace(namespace)
        for key in tuple(self.entries):
            if namespace is None or key[0] == namespace:
                self._evict(key)

    def stats(self) -> dict:
        return {
            "enabled": True,
            "identity_sha256": self.identity,
            "max_blocks": self.max_blocks,
            "max_entries": self.max_entries,
            "entries": len(self.entries),
            "cached_blocks": len(self.pool._pins),
            "host_logits_bytes": sum(e.logits.nbytes for e in self.entries.values()),
            "cached_token_references": sum(
                len(e.tokens) for e in self.entries.values()
            ),
            **self.counters,
        }
