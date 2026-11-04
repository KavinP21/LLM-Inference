"""What a model forward sees: positions, attention over the paged cache, state.

Model adapters are written once against this interface. A
:class:`PrefillBatch` is a chunk of one sequence's prompt; a
:class:`DecodeBatch` is one new token from each of several sequences. Both
take queries, keys and values as ``[tokens, heads, head_dim]``, write the
keys and values to the cache, and return attention output as
``[tokens, q_heads * head_dim]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .backends import portable
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
        images: Sequence[tuple[int, Any]] = (),
    ) -> None:
        mx = cache.mx
        self.cache = cache
        self.mx = mx
        self.tokens = tokens
        self.count = int(tokens.shape[0])
        self.start = start
        self.slot = slot
        end = start + self.count
        # Image soft tokens inside this chunk (the scheduler never splits one)
        # replace the placeholder embeddings and attend to each other both ways.
        self.images = [(at, emb) for at, emb in images if start <= at < end]
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

    def inject(self, x: Any) -> Any:
        """Put image embeddings in place of their placeholder tokens."""
        for at, embeddings in self.images:
            offset = at - self.start
            x[offset : offset + int(embeddings.shape[0])] = embeddings.astype(x.dtype)
        return x

    def rope(
        self, x: Any, dims: int, *, base: float | None = None, freqs: Any = None
    ) -> Any:
        """Rotate ``[tokens, heads, head_dim]`` at this chunk's positions."""
        rotated = portable.rope(
            self.mx,
            x.transpose(1, 0, 2)[None],
            dims,
            base=base,
            freqs=freqs,
            offset=self.start,
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
            # the bottom-right corner, i.e. to the newest key. An image in the
            # chunk also sees its own later tokens, so it needs every key.
            stop = length if self.images else self.start + end - first
            mask: Any = "causal"
            if group.window or softcap or self.images:
                query_positions = mx.arange(self.start + begin, self.start + end)[
                    :, None
                ]
                key_positions = mx.arange(first, first + stop)[None, :]
                mask = key_positions <= query_positions
                if group.window:
                    mask = mask & (key_positions > query_positions - group.window)
                for at, embeddings in self.images:
                    span_end = at + int(embeddings.shape[0])
                    inside = (query_positions >= at) & (query_positions < span_end)
                    mask = mask | (
                        inside & (key_positions >= at) & (key_positions < span_end)
                    )
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

    def inject(self, x: Any) -> Any:
        return x

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
        # Without the Metal kernel, each group's keys are gathered into a
        # dense block (the last ``span`` of each sequence) for plain attention.
        self._metal = portable.metal_kernels(mx)
        self._dense: dict[str, tuple[Any, Any]] = {}
        if not self._metal:
            for name, group in cache.groups.items():
                span = min(self.longest, group.window or self.longest)
                self._dense[name] = portable.decode_rows(
                    mx, self._tables[name], self.lengths, page_size=page, span=span
                )

    @property
    def positions(self) -> Any:
        return self.offsets

    def rope(
        self, x: Any, dims: int, *, base: float | None = None, freqs: Any = None
    ) -> Any:
        rotated = portable.rope(
            self.mx, x[:, :, None, :], dims, base=base, freqs=freqs, offset=self.offsets
        )
        return rotated[:, :, 0, :]

    def attend(
        self, layer: int, q: Any, k: Any, v: Any, *, scale: float, softcap: float = 0.0
    ) -> Any:
        group = self.cache.layer_group[layer]
        self.cache.write(layer, self._write[group.name], k, v)
        if not self._metal:
            return self._attend_dense(layer, group.name, q, scale, softcap)
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
            scales=(self.cache.key_scales[layer], self.cache.value_scales[layer])
            if self.cache.bits == 8
            else None,
        )
        return out.reshape(self.count, -1)

    def _attend_dense(
        self, layer: int, name: str, q: Any, scale: float, softcap: float
    ) -> Any:
        rows, valid = self._dense[name]
        batch, span = int(rows.shape[0]), int(rows.shape[1])
        keys, values = self.cache.gather(layer, rows.reshape(-1))
        heads, dim = int(keys.shape[1]), int(keys.shape[2])
        keys = keys.reshape(batch, span, heads, dim).transpose(0, 2, 1, 3)
        values = values.reshape(batch, span, heads, dim).transpose(0, 2, 1, 3)
        out = portable.attention(
            self.mx,
            q[:, :, None, :],
            keys,
            values,
            scale=scale,
            mask=valid[:, None, None, :],
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
    return portable.attention(mx, q, k, v, scale=scale, mask=mask, softcap=softcap)
