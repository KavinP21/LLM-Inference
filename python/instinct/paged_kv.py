"""Physical 16-token K/V pages stored as independent MLX Metal buffers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class _DeviceBlock:
    layers: list[Any | None]
    written_masks: list[int]


class MlxPagedKVStore:
    """Lazily materialized physical pages shared by all live sequences.

    Each physical block owns one independently replaceable array per layer. A
    write replaces only the touched layer-page rather than concatenating or
    copying a sequence's complete history. Token positions are append-only.
    """

    def __init__(
        self,
        mx: Any,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_tokens: int = 16,
        is_shared: Callable[[int], bool] | None = None,
    ) -> None:
        if min(num_layers, num_kv_heads, head_dim, block_tokens) <= 0:
            raise ValueError("invalid paged K/V dimensions")
        self.mx = mx
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_tokens = block_tokens
        self.block_shape = (
            2,
            block_tokens,
            num_kv_heads,
            head_dim,
        )
        self.bytes_per_layer_block = 2 * block_tokens * num_kv_heads * head_dim * 2
        self.bytes_per_block = num_layers * self.bytes_per_layer_block
        self._blocks: dict[int, _DeviceBlock] = {}
        self.peak_allocated_blocks = 0
        self._is_shared = is_shared or (lambda _: False)

    def _check_writable(self, physical_ids: Iterable[int]) -> None:
        if any(self._is_shared(int(p)) for p in physical_ids):
            raise RuntimeError("shared K/V page requires copy-on-write before mutation")

    def clone_block(self, source: int, target: int) -> None:
        if source not in self._blocks or target in self._blocks or target < 0:
            raise ValueError("invalid K/V copy-on-write source or target")
        original = self._blocks[source]
        # MLX arrays are immutable: duplicate ownership/masks now, new buffers
        # only when touched layers are replaced by an append write.
        self._blocks[target] = _DeviceBlock(
            list(original.layers), list(original.written_masks)
        )
        self.peak_allocated_blocks = max(
            self.peak_allocated_blocks, self.allocated_blocks
        )

    def validate_prefix(self, table: tuple[int, ...], tokens: int) -> None:
        if (
            tokens <= 0
            or len(table) != (tokens + self.block_tokens - 1) // self.block_tokens
        ):
            raise ValueError("invalid prefix page coverage")
        for i, physical_id in enumerate(table):
            block = self._blocks.get(physical_id)
            count = min(self.block_tokens, tokens - i * self.block_tokens)
            expected = (1 << count) - 1
            if (
                block is None
                or any(a is None for a in block.layers)
                or any(m & expected != expected for m in block.written_masks)
            ):
                raise RuntimeError("cannot cache incomplete K/V pages")

    @property
    def allocated_blocks(self) -> int:
        return len(self._blocks)

    @property
    def allocated_bytes(self) -> int:
        materialized_layers = sum(
            layer is not None
            for block in self._blocks.values()
            for layer in block.layers
        )
        return materialized_layers * self.bytes_per_layer_block

    def _block(self, physical_id: int) -> _DeviceBlock:
        if physical_id < 0:
            raise ValueError("negative physical K/V block id")
        block = self._blocks.get(physical_id)
        if block is None:
            block = _DeviceBlock(
                [None] * self.num_layers,
                [0] * self.num_layers,
            )
            self._blocks[physical_id] = block
            self.peak_allocated_blocks = max(
                self.peak_allocated_blocks, self.allocated_blocks
            )
        return block

    def _layer_array(self, physical_id: int, layer: int) -> Any:
        block = self._block(physical_id)
        array = block.layers[layer]
        if array is None:
            array = self.mx.zeros(self.block_shape, dtype=self.mx.float16)
            block.layers[layer] = array
        return array

    def _write_segments(
        self, block_table: tuple[int, ...], start_position: int, token_count: int
    ) -> list[tuple[int, int, int]]:
        segments: list[tuple[int, int, int]] = []
        source_offset = 0
        while source_offset < token_count:
            position = start_position + source_offset
            logical_block = position // self.block_tokens
            if logical_block >= len(block_table):
                raise ValueError("K/V write exceeds the sequence block table")
            physical_id = int(block_table[logical_block])
            block_offset = position % self.block_tokens
            take = min(token_count - source_offset, self.block_tokens - block_offset)
            bit_mask = ((1 << take) - 1) << block_offset
            segments.append((physical_id, bit_mask, take))
            source_offset += take
        return segments

    def write_layer(
        self,
        layer: int,
        block_table: tuple[int, ...],
        start_position: int,
        key: Any,
        value: Any,
    ) -> tuple[int, ...]:
        if not 0 <= layer < self.num_layers:
            raise ValueError("K/V layer index is out of range")
        if start_position < 0 or key.shape != value.shape:
            raise ValueError("invalid paged K/V write")
        if len(key.shape) != 3 or tuple(key.shape[1:]) != (
            self.num_kv_heads,
            self.head_dim,
        ):
            raise ValueError("unexpected K/V tensor shape")

        touched: list[int] = []
        source_offset = 0
        self._check_writable(
            p
            for p, _, _ in self._write_segments(
                block_table, start_position, int(key.shape[0])
            )
        )
        for physical_id, bit_mask, take in self._write_segments(
            block_table, start_position, int(key.shape[0])
        ):
            block = self._block(physical_id)
            if block.written_masks[layer] & bit_mask:
                raise RuntimeError("attempted to overwrite an existing K/V token")

            source = slice(source_offset, source_offset + take)
            block_offset = (start_position + source_offset) % self.block_tokens
            target = slice(block_offset, block_offset + take)
            page = self._layer_array(physical_id, layer)
            updated = page.at[0, target, :, :].add(key[source])
            updated = updated.at[1, target, :, :].add(value[source])
            block.layers[layer] = updated
            block.written_masks[layer] |= bit_mask
            touched.append(physical_id)
            source_offset += take
        return tuple(dict.fromkeys(touched))

    def prepare_prefill_write(
        self,
        layer: int,
        block_table: tuple[int, ...],
        start_position: int,
        token_count: int,
    ) -> tuple[tuple[int, ...], Any]:
        if not 0 <= layer < self.num_layers:
            raise ValueError("K/V layer index is out of range")
        if start_position < 0 or token_count <= 0:
            raise ValueError("invalid paged K/V write")
        segments = self._write_segments(block_table, start_position, token_count)
        physical_ids = tuple(item[0] for item in segments)
        self._check_writable(physical_ids)
        for physical_id, bit_mask, _ in segments:
            if self._block(physical_id).written_masks[layer] & bit_mask:
                raise RuntimeError("attempted to overwrite an existing K/V token")
        pages = self.mx.stack(
            [self._layer_array(physical_id, layer) for physical_id in physical_ids],
            axis=0,
        )
        return physical_ids, pages

    def commit_prefill_write(
        self,
        layer: int,
        physical_ids: tuple[int, ...],
        pages: Any,
        start_position: int,
        token_count: int,
    ) -> None:
        if int(pages.shape[0]) != len(physical_ids):
            raise ValueError("updated prefill pages do not match their physical ids")
        self._check_writable(physical_ids)
        offset = 0
        for index, physical_id in enumerate(physical_ids):
            take = min(
                token_count - offset,
                self.block_tokens - ((start_position + offset) % self.block_tokens),
            )
            bit_offset = (start_position + offset) % self.block_tokens
            bit_mask = ((1 << take) - 1) << bit_offset
            block = self._block(physical_id)
            block.layers[layer] = pages[index]
            block.written_masks[layer] |= bit_mask
            offset += take

    def prepare_decode_write(
        self,
        layer: int,
        block_tables: list[tuple[int, ...]],
        positions: list[int],
    ) -> tuple[tuple[int, ...], Any]:
        if not 0 <= layer < self.num_layers:
            raise ValueError("K/V layer index is out of range")
        if len(block_tables) != len(positions) or not positions:
            raise ValueError("decode K/V metadata does not align")
        if any(position < 0 for position in positions):
            raise ValueError("decode K/V positions must be non-negative")
        physical_ids = tuple(
            int(table[position // self.block_tokens])
            for table, position in zip(block_tables, positions)
        )
        if len(set(physical_ids)) != len(physical_ids):
            raise RuntimeError("live sequences unexpectedly share writable K/V pages")
        self._check_writable(physical_ids)
        for physical_id, position in zip(physical_ids, positions):
            bit_mask = 1 << (position % self.block_tokens)
            if self._block(physical_id).written_masks[layer] & bit_mask:
                raise RuntimeError("attempted to overwrite an existing K/V token")
        pages = self.mx.stack(
            [self._layer_array(physical_id, layer) for physical_id in physical_ids],
            axis=0,
        )
        return physical_ids, pages

    def commit_decode_write(
        self,
        layer: int,
        physical_ids: tuple[int, ...],
        pages: Any,
        positions: list[int],
    ) -> None:
        if int(pages.shape[0]) != len(physical_ids):
            raise ValueError("updated decode pages do not match their physical ids")
        self._check_writable(physical_ids)
        for index, (physical_id, position) in enumerate(zip(physical_ids, positions)):
            block = self._block(physical_id)
            block.layers[layer] = pages[index]
            block.written_masks[layer] |= 1 << (position % self.block_tokens)

    def gather_layer(
        self, layer: int, block_table: tuple[int, ...], token_count: int
    ) -> tuple[Any, Any]:
        if token_count <= 0:
            raise ValueError("cannot gather an empty K/V prefix")
        return self.gather_layer_range(layer, block_table, 0, token_count)

    def gather_layer_range(
        self,
        layer: int,
        block_table: tuple[int, ...],
        start_position: int,
        end_position: int,
    ) -> tuple[Any, Any]:
        """Gather one validated half-open logical K/V range.

        Sliding-window prefill uses this path so it neither copies nor scores
        pages that are already outside every query token's receptive field.
        """
        if start_position < 0 or end_position <= start_position:
            raise ValueError("cannot gather an empty or negative K/V range")
        if not 0 <= layer < self.num_layers:
            raise ValueError("K/V layer index is out of range")
        first_block = start_position // self.block_tokens
        final_block = (end_position - 1) // self.block_tokens
        if final_block >= len(block_table):
            raise ValueError("K/V block table is shorter than the requested range")
        keys = []
        values = []
        for logical_block in range(first_block, final_block + 1):
            physical_id = block_table[logical_block]
            try:
                block = self._blocks[int(physical_id)]
            except KeyError as exc:
                raise RuntimeError(
                    "K/V block was allocated but never materialized"
                ) from exc
            page_start = max(0, start_position - logical_block * self.block_tokens)
            page_end = min(
                self.block_tokens,
                end_position - logical_block * self.block_tokens,
            )
            take = page_end - page_start
            expected = ((1 << take) - 1) << page_start
            if (block.written_masks[layer] & expected) != expected:
                raise RuntimeError("K/V range contains unwritten token slots")
            page = block.layers[layer]
            if page is None:
                raise RuntimeError("K/V layer page was never materialized")
            keys.append(page[0, page_start:page_end, :, :])
            values.append(page[1, page_start:page_end, :, :])
        if len(keys) == 1:
            return keys[0], values[0]
        return self.mx.concatenate(keys, axis=0), self.mx.concatenate(values, axis=0)

    def pack_layer(
        self,
        layer: int,
        block_tables: list[tuple[int, ...]],
        token_counts: list[int],
        *,
        window_size: int = 0,
    ) -> tuple[Any, Any, Any]:
        if len(block_tables) != len(token_counts) or not block_tables:
            raise ValueError("paged attention metadata does not align")
        if not 0 <= layer < self.num_layers:
            raise ValueError("K/V layer index is out of range")
        if min(token_counts) <= 0:
            raise ValueError("paged attention lengths must be positive")
        if window_size < 0:
            raise ValueError("paged attention window must be non-negative")
        unique_ids: list[int] = []
        remap: dict[int, int] = {}
        rows: list[list[int]] = []
        maximum_blocks = max(
            (count + self.block_tokens - 1) // self.block_tokens
            for count in token_counts
        )
        for table, token_count in zip(block_tables, token_counts):
            required = (token_count + self.block_tokens - 1) // self.block_tokens
            first_token = max(0, token_count - window_size) if window_size else 0
            first_block = first_token // self.block_tokens
            row: list[int] = [0] * first_block
            remaining = token_count - first_block * self.block_tokens
            for physical_id in table[first_block:required]:
                physical_id = int(physical_id)
                take = min(remaining, self.block_tokens)
                block = self._blocks.get(physical_id)
                expected = (1 << take) - 1
                if block is None or (block.written_masks[layer] & expected) != expected:
                    raise RuntimeError(
                        "paged attention encountered unwritten K/V slots"
                    )
                if block.layers[layer] is None:
                    raise RuntimeError(
                        "paged attention encountered a missing layer page"
                    )
                if physical_id not in remap:
                    remap[physical_id] = len(unique_ids)
                    unique_ids.append(physical_id)
                row.append(remap[physical_id])
                remaining -= take
            if remaining:
                raise ValueError(
                    "K/V block table is shorter than the requested attention prefix"
                )
            row.extend([0] * (maximum_blocks - len(row)))
            rows.append(row)
        pages = self.mx.stack(
            [self._blocks[physical_id].layers[layer] for physical_id in unique_ids],
            axis=0,
        )
        return (
            pages,
            self.mx.array(rows, dtype=self.mx.int32),
            self.mx.array(token_counts, dtype=self.mx.int32),
        )

    def materialize(self, physical_ids: Iterable[int]) -> None:
        arrays = [
            layer
            for block_id in dict.fromkeys(physical_ids)
            if block_id in self._blocks
            for layer in self._blocks[block_id].layers
            if layer is not None
        ]
        if arrays:
            self.mx.eval(*arrays)

    def truncate(self, block_table: tuple[int, ...], token_count: int) -> None:
        """Discard a speculative suffix from an independently owned sequence.

        Clear both occupancy masks and the underlying values. Writes use
        addition into zero slots, so clearing masks alone would silently add
        rejected K/V values to replacement tokens. Physical identifiers in the
        caller's reservation remain valid and may be materialized again.
        """
        if (
            type(token_count) is not int
            or token_count < 0
            or token_count > len(block_table) * self.block_tokens
            or len(set(block_table)) != len(block_table)
        ):
            raise ValueError("invalid K/V truncation coverage")
        first = token_count // self.block_tokens
        affected = block_table[first:]
        self._check_writable(affected)
        if token_count:
            covered = (token_count + self.block_tokens - 1) // self.block_tokens
            self.validate_prefix(block_table[:covered], token_count)
        keep = token_count % self.block_tokens
        for index, physical_id in enumerate(affected):
            if index or not keep:
                self._blocks.pop(int(physical_id), None)
                continue
            block = self._blocks.get(int(physical_id))
            if block is None:
                continue
            mask = (1 << keep) - 1
            for layer, page in enumerate(block.layers):
                if page is not None:
                    block.layers[layer] = self.mx.concatenate(
                        [page[:, :keep], self.mx.zeros_like(page[:, keep:])], axis=1
                    )
                block.written_masks[layer] &= mask

    def release(self, physical_ids: Iterable[int]) -> None:
        for physical_id in physical_ids:
            self._blocks.pop(int(physical_id), None)

    def clear(self) -> None:
        self._blocks.clear()
        self.peak_allocated_blocks = 0

    def reset_peak(self) -> None:
        self.peak_allocated_blocks = self.allocated_blocks
