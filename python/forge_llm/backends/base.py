"""Structural contract shared by Forge decoder-family adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..format import ModelConfig
from ..model_file import ModelFile
from ..paged_kv import MlxPagedKVStore


class MlxDecoderModel(Protocol):
    """The model-family surface consumed by :class:`MlxEngine`.

    Keeping scheduling and K/V ownership outside the family adapter lets the
    same runtime serve architectures with different normalization, attention,
    and MLP semantics.
    """

    mx: Any
    file: ModelFile
    config: ModelConfig
    max_model_length: int
    attention_tile_size: int
    custom_metal: bool
    metal_paged_attention: bool

    @property
    def head_dim(self) -> int: ...

    def forward_paged_chunk(
        self,
        input_ids: list[int] | np.ndarray,
        *,
        start_position: int,
        block_table: tuple[int, ...],
        store: MlxPagedKVStore,
    ) -> Any: ...

    def decode_paged_batch(
        self,
        input_ids: list[int],
        *,
        positions: list[int],
        block_tables: list[tuple[int, ...]],
        store: MlxPagedKVStore,
    ) -> Any: ...

    def close(self) -> None: ...
