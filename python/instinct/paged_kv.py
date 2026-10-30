"""Paged K/V storage for the MLX engine.

Every attention layer keeps its keys and values in one arena of token rows,
``[rows, kv_heads, head_dim]``. A page is ``page_size`` consecutive rows and
each sequence has a page table mapping its logical pages to physical ones, so
sequences of any length share a buffer with no padding. Writes are scatter
updates that MLX applies in place. Prefill gathers a sequence's rows for
attention; decode reads the arena directly through the page table
(:func:`metal_kernels.paged_decode_attention`).

Layers with the same shape and window form a group with its own page pool.
A sliding-window group hands pages back as soon as they fall out of the
window, so its memory stays bounded however long a sequence gets. Recurrent
layers (Qwen3.5's Gated DeltaNet) keep fixed-size state per sequence in slot
arrays instead of pages.

Arenas start small and double as pages are first used, up to the configured
capacity, so a short chat doesn't commit the whole cache budget.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

INITIAL_PAGES = 64


@dataclass(frozen=True)
class KVGroup:
    """Attention layers that share a K/V shape and window."""

    name: str
    layers: tuple[int, ...]
    kv_heads: int
    head_dim: int
    window: int = 0  # 0 for full attention

    def bytes_per_token(self, bits: int = 16) -> int:
        # 8-bit rows carry a 2-byte scale per head.
        row = self.head_dim * bits // 8 + (2 if bits == 8 else 0)
        return 2 * len(self.layers) * self.kv_heads * row


def quantize(mx: Any, x: Any) -> tuple[Any, Any]:
    """Rows of ``x`` as int8 with one fp16 scale each (absmax / 127)."""
    x = x.astype(mx.float32)
    scale = (mx.maximum(mx.max(mx.abs(x), axis=-1), 1e-4) / 127.0).astype(mx.float16)
    q = mx.round(x / scale[..., None].astype(mx.float32))
    return mx.clip(q, -127, 127).astype(mx.int8), scale


@dataclass(frozen=True)
class StateGroup:
    """Recurrent layers with fixed-size per-sequence state (one array per shape)."""

    name: str
    layers: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[Any, ...]

    def bytes_per_slot(self) -> int:
        return len(self.layers) * sum(
            math.prod(shape) * np.dtype(_numpy_name(dtype)).itemsize
            for shape, dtype in zip(self.shapes, self.dtypes)
        )


def _numpy_name(dtype: Any) -> str:
    name = str(dtype).rsplit(".", 1)[-1]
    return "float16" if name == "bfloat16" else name


class PagePool:
    """Physical page ids with reference counts and admission reservations.

    The lowest free id is always handed out first, so the arena only has to
    cover the high-water mark. ``reserve`` sets aside pages a sequence may
    need later; ``available`` is what is neither in use nor promised.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("a page pool needs at least one page")
        self.capacity = capacity
        self._free = list(range(capacity))
        self._refs = np.zeros(capacity, dtype=np.int32)
        self.reserved = 0
        self.high_water = 0

    @property
    def free(self) -> int:
        return len(self._free)

    @property
    def available(self) -> int:
        return len(self._free) - self.reserved

    @property
    def used(self) -> int:
        return self.capacity - len(self._free)

    def reserve(self, count: int) -> bool:
        if count > self.available:
            return False
        self.reserved += count
        return True

    def unreserve(self, count: int) -> None:
        if not 0 <= count <= self.reserved:
            raise RuntimeError("released more pages than were reserved")
        self.reserved -= count

    def allocate(self, *, reserved: bool) -> int:
        """Take one page, from the caller's reservation if ``reserved``."""
        if reserved:
            self.unreserve(1)
        elif self.available <= 0:
            raise RuntimeError("the K/V page pool is exhausted")
        page = heapq.heappop(self._free)
        self._refs[page] = 1
        self.high_water = max(self.high_water, page + 1)
        return page

    def retain(self, pages: Iterable[int]) -> None:
        for page in pages:
            if page >= 0:
                if self._refs[page] <= 0:
                    raise RuntimeError("retained a page that isn't allocated")
                self._refs[page] += 1

    def release(self, pages: Iterable[int]) -> None:
        for page in pages:
            if page < 0:
                continue
            if self._refs[page] <= 0:
                raise RuntimeError("released a page that isn't allocated")
            self._refs[page] -= 1
            if self._refs[page] == 0:
                heapq.heappush(self._free, page)

    def shared(self, page: int) -> bool:
        return self._refs[page] > 1


class KVCache:
    """Device arrays for every K/V group and recurrent-state group."""

    def __init__(
        self,
        mx: Any,
        groups: Sequence[KVGroup],
        *,
        page_size: int,
        capacities: dict[str, int],
        dtype: Any,
        state_groups: Sequence[StateGroup] = (),
        state_slots: int = 0,
        bits: int = 16,
    ) -> None:
        if bits not in {8, 16}:
            raise ValueError("K/V is stored in 16 or 8 bits")
        self.mx = mx
        self.page_size = page_size
        self.dtype = dtype
        # 8-bit K/V keeps one scale per token and head (absmax / 127).
        self.bits = bits
        self.storage = mx.int8 if bits == 8 else dtype
        self.key_scales: dict[int, Any] = {}
        self.value_scales: dict[int, Any] = {}
        self.groups = {group.name: group for group in groups}
        self.layer_group = {layer: group for group in groups for layer in group.layers}
        self.pools = {group.name: PagePool(capacities[group.name]) for group in groups}
        self.keys: dict[int, Any] = {}
        self.values: dict[int, Any] = {}
        self._pages = {group.name: 0 for group in groups}
        for group in groups:
            self._grow(group, min(INITIAL_PAGES, capacities[group.name]))
        self.state_groups = {group.name: group for group in state_groups}
        self.layer_state = {
            layer: group for group in state_groups for layer in group.layers
        }
        self.states: dict[int, list[Any]] = {
            layer: [
                mx.zeros((state_slots, *shape), dtype)
                for shape, dtype in zip(group.shapes, group.dtypes)
            ]
            for group in state_groups
            for layer in group.layers
        }
        self.state_slots = list(range(state_slots - 1, -1, -1))
        mx.eval(
            *[a for store in self._stores for a in store.values()],
            *[a for arrays in self.states.values() for a in arrays],
        )

    # --- capacity -------------------------------------------------------------

    @property
    def _stores(self) -> tuple[dict[int, Any], ...]:
        """Per-layer arrays indexed by row: K, V and, at 8 bits, their scales."""
        if self.bits == 8:
            return self.keys, self.values, self.key_scales, self.value_scales
        return self.keys, self.values

    def _grow(self, group: KVGroup, pages: int) -> None:
        mx = self.mx
        old = self._pages[group.name]
        if pages <= old:
            return
        extra = (pages - old) * self.page_size
        for layer in group.layers:
            for index, store in enumerate(self._stores):
                tail, dtype = (
                    ((group.head_dim,), self.storage) if index < 2 else ((), mx.float16)
                )
                fresh = mx.zeros((extra, group.kv_heads, *tail), dtype)
                store[layer] = (
                    fresh
                    if layer not in store
                    else mx.concatenate([store[layer], fresh])
                )
        self._pages[group.name] = pages

    def ensure_capacity(self) -> None:
        """Grow any arena whose pool has handed out pages beyond its size."""
        grown = []
        for name, group in self.groups.items():
            needed = self.pools[name].high_water
            if needed > self._pages[name]:
                target = min(
                    self.pools[name].capacity, max(needed, 2 * self._pages[name])
                )
                self._grow(group, target)
                for store in self._stores:
                    grown.extend(store[layer] for layer in group.layers)
        if grown:
            self.mx.eval(*grown)

    @property
    def device_bytes(self) -> int:
        total = sum(array.nbytes for store in self._stores for array in store.values())
        return total + sum(a.nbytes for arrays in self.states.values() for a in arrays)

    # --- pages ----------------------------------------------------------------

    def rows(self, table: np.ndarray, start: int, end: int) -> np.ndarray:
        """Physical rows of logical positions ``[start, end)``."""
        positions = np.arange(start, end, dtype=np.int64)
        pages = table[positions // self.page_size]
        if np.any(pages < 0):
            raise RuntimeError("K/V range includes a released page")
        return (pages * self.page_size + positions % self.page_size).astype(np.int32)

    def copy_page(self, group: KVGroup, source: int, target: int, count: int) -> None:
        """Copy the first ``count`` rows of a page (copy-on-write of a shared tail)."""
        mx = self.mx
        src = mx.arange(source * self.page_size, source * self.page_size + count)
        dst = mx.arange(target * self.page_size, target * self.page_size + count)
        for layer in group.layers:
            for store in self._stores:
                store[layer][dst] = store[layer][src]

    # --- writes and reads -----------------------------------------------------

    def write(self, layer: int, rows: Any, keys: Any, values: Any) -> None:
        if self.bits == 8:
            keys, self.key_scales[layer][rows] = quantize(self.mx, keys)
            values, self.value_scales[layer][rows] = quantize(self.mx, values)
        self.keys[layer][rows] = keys.astype(self.storage)
        self.values[layer][rows] = values.astype(self.storage)

    def gather(self, layer: int, rows: Any) -> tuple[Any, Any]:
        keys, values = self.keys[layer][rows], self.values[layer][rows]
        if self.bits == 8:
            f32 = self.mx.float32
            keys = keys.astype(f32) * self.key_scales[layer][rows][..., None]
            values = values.astype(f32) * self.value_scales[layer][rows][..., None]
            keys, values = keys.astype(self.dtype), values.astype(self.dtype)
        return keys, values

    # --- recurrent state slots ------------------------------------------------

    def take_slot(self) -> int:
        if not self.state_slots:
            raise RuntimeError("no recurrent-state slot is free")
        return self.state_slots.pop()

    def free_slot(self, slot: int) -> None:
        if slot >= 0:
            self.state_slots.append(slot)

    def copy_slot(self, source: int, target: int) -> None:
        for arrays in self.states.values():
            for array in arrays:
                array[target] = array[source]

    def clear_slot(self, slot: int) -> None:
        for arrays in self.states.values():
            for array in arrays:
                array[slot] = 0
