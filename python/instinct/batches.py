"""What a model forward sees: positions, attention over the paged cache, state.

Model adapters are written once against this interface. A
:class:`PrefillBatch` is a chunk of one sequence's prompt; a
:class:`DecodeBatch` is one new token from each of several sequences. Both
take queries, keys and values as ``[tokens, heads, head_dim]``, write the
keys and values to the cache, and return attention output as
``[tokens, q_heads * head_dim]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .backends.metal_kernels import paged_decode_attention
from .paged_kv import KVCache

# Bound on one prefill score tile ([heads, queries, keys], float32), since
# MLX materializes scores for large head dims.
SCORE_TILE_BYTES = 1 << 29


class PrefillBatch:
    prefill = True

    def __init__(
        self,
        cache: KVCache,
        tokens: Any,
        *,
        start: int,
        tables: dict[str, np.ndarray],
        slot: int = -1,
    ) -> None:
        mx = cache.mx
        self.cache = cache
        self.mx = mx
        self.tokens = tokens
        self.count = int(tokens.shape[0])
        self.start = start
        self.slot = slot
        end = start + self.count
        self._write: dict[str, Any] = {}
        self._read: dict[str, tuple[Any, int]] = {}
        for name, group in cache.groups.items():
            table = tables[name]
            self._write[name] = mx.array(cache.rows(table, start, end))
            first = max(0, start - group.window + 1) if group.window else 0
            self._read[name] = (mx.array(cache.rows(table, first, end)), first)

    @property
    def positions(self) -> Any:
        return self.mx.arange(self.start, self.start + self.count)

    def rope(
        self, x: Any, dims: int, *, base: float | None = None, freqs: Any = None
    ) -> Any:
        """Rotate ``[tokens, heads, head_dim]`` at this chunk's positions."""
        mx = self.mx
        rotated = mx.fast.rope(
            x.transpose(1, 0, 2)[None],
            dims,
            traditional=False,
            base=base,
            scale=1.0,
            offset=self.start,
            freqs=freqs,
        )
        return rotated[0].transpose(1, 0, 2)

    def attend(
        self, layer: int, q: Any, k: Any, v: Any, *, scale: float, softcap: float = 0.0
    ) -> Any:
        mx = self.mx
        group = self.cache.layer_group[layer]
        self.cache.write(layer, self._write[group.name], k, v)
        rows, first = self._read[group.name]
        keys, values = self.cache.gather(layer, rows)
        heads = int(q.shape[1])
        queries = q.transpose(1, 0, 2)[None]
        keys = keys.transpose(1, 0, 2)[None]
        values = values.transpose(1, 0, 2)[None]
        length = int(keys.shape[2])
        tile = max(16, SCORE_TILE_BYTES // (4 * heads * length))
        outputs = []
        for begin in range(0, self.count, tile):
            end = min(self.count, begin + tile)
            # Keys up to this tile's last query; the causal mask is aligned to
            # the bottom-right corner, i.e. to the newest key.
            stop = self.start + end - first
            mask: Any = "causal"
            if group.window or softcap:
                query_positions = mx.arange(self.start + begin, self.start + end)[
                    :, None
                ]
                key_positions = mx.arange(first, first + stop)[None, :]
                mask = key_positions <= query_positions
                if group.window:
                    mask = mask & (key_positions > query_positions - group.window)
            outputs.append(
                _attention(
                    mx,
                    queries[:, :, begin:end],
                    keys[:, :, :stop],
                    values[:, :, :stop],
                    scale=scale,
                    mask=mask,
                    softcap=softcap,
                )
            )
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=2)
        return out[0].transpose(1, 0, 2).reshape(self.count, -1)

    def state(self, layer: int) -> list[Any]:
        return [array[self.slot : self.slot + 1] for array in self.cache.states[layer]]

    def set_state(self, layer: int, arrays: list[Any]) -> None:
        for target, value in zip(self.cache.states[layer], arrays):
            target[self.slot : self.slot + 1] = value


class DecodeBatch:
    prefill = False

    def __init__(
        self,
        cache: KVCache,
        tokens: Any,
        *,
        positions: np.ndarray,
        tables: dict[str, list[np.ndarray]],
        slots: np.ndarray | None = None,
    ) -> None:
        mx = cache.mx
        self.cache = cache
        self.mx = mx
        self.tokens = tokens
        self.count = int(tokens.shape[0])
        self.offsets = mx.array(positions.astype(np.int32))
        self.lengths = mx.array((positions + 1).astype(np.int32))
        self.longest = int(positions.max()) + 1
        self.slots = None if slots is None else mx.array(slots.astype(np.int32))
        page = cache.page_size
        self._write: dict[str, Any] = {}
        self._tables: dict[str, Any] = {}
        for name in cache.groups:
            group_tables = tables[name]
            width = max(len(table) for table in group_tables)
            packed = np.zeros((self.count, width), dtype=np.int32)
            for row, table in enumerate(group_tables):
                packed[row, : len(table)] = table
            pages = packed[np.arange(self.count), positions // page]
            self._write[name] = mx.array(
                (pages * page + positions % page).astype(np.int32)
            )
            self._tables[name] = mx.array(packed)

    @property
    def positions(self) -> Any:
        return self.offsets

    def rope(
        self, x: Any, dims: int, *, base: float | None = None, freqs: Any = None
    ) -> Any:
        rotated = self.mx.fast.rope(
            x[:, :, None, :],
            dims,
            traditional=False,
            base=base,
            scale=1.0,
            offset=self.offsets,
            freqs=freqs,
        )
        return rotated[:, :, 0, :]

    def attend(
        self, layer: int, q: Any, k: Any, v: Any, *, scale: float, softcap: float = 0.0
    ) -> Any:
        group = self.cache.layer_group[layer]
        self.cache.write(layer, self._write[group.name], k, v)
        out = paged_decode_attention(
            self.mx,
            q,
            self.cache.keys[layer],
            self.cache.values[layer],
            self._tables[group.name],
            self.lengths,
            page_size=self.cache.page_size,
            longest=self.longest,
            scale=scale,
            window=group.window,
            softcap=softcap,
        )
        return out.reshape(self.count, -1)

    def state(self, layer: int) -> list[Any]:
        return [array[self.slots] for array in self.cache.states[layer]]

    def set_state(self, layer: int, arrays: list[Any]) -> None:
        for target, value in zip(self.cache.states[layer], arrays):
            target[self.slots] = value


def _attention(
    mx: Any, q: Any, k: Any, v: Any, *, scale: float, mask: Any, softcap: float
) -> Any:
    if not softcap:
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)
    scores = (q.astype(mx.float32) * scale) @ k.astype(mx.float32).transpose(0, 1, 3, 2)
    scores = mx.tanh(scores / softcap) * softcap
    scores = mx.where(mask, scores, -mx.inf)
    return (mx.softmax(scores, axis=-1) @ v.astype(mx.float32)).astype(q.dtype)
