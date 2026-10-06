"""Public request-oriented Engine implementation for the MLX backend."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

from .backends.factory import create_mlx_model
from .backends.mlx import mlx_build_info
from .paged_kv import MlxPagedKVStore
from .runtime import IterationScheduler, KVBlockPool, SequenceState, TokenEvent


class MlxEngine:
    """Greedy multi-family engine with paged K/V and continuous batching."""

    backend = "mlx"
    block_tokens = 16

    def __init__(
        self,
        model_path: str | Path,
        max_num_sequences: int = 16,
        max_model_length: int | None = None,
        kv_cache_bytes: int = 512 << 20,
        prefill_chunk_size: int = 512,
        attention_tile_size: int = 1024,
        custom_metal: bool = True,
        metal_paged_attention: bool = True,
        int8_mode: str = "auto",
    ) -> None:
        if max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be positive")
        if kv_cache_bytes <= 0:
            raise ValueError("kv_cache_bytes must be positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        self.model = create_mlx_model(
            str(model_path),
            max_model_length=max_model_length,
            attention_tile_size=attention_tile_size,
            custom_metal=custom_metal,
            metal_paged_attention=metal_paged_attention,
            int8_mode=int8_mode,
        )
        self.max_num_sequences = max_num_sequences
        self.max_model_length = self.model.max_model_length
        self.prefill_chunk_size = prefill_chunk_size
        config = self.model.config
        self.bytes_per_token = (
            config.num_hidden_layers
            * 2
            * config.num_key_value_heads
            * self.model.head_dim
            * 2
        )
        self.bytes_per_block = self.bytes_per_token * self.block_tokens
        self.cache_pool = KVBlockPool(
            kv_cache_bytes,
            self.bytes_per_token,
            self.block_tokens,
            layout="paged",
        )
        self.scheduler = IterationScheduler(
            max_num_sequences,
            self.max_model_length,
            config.vocab_size,
            self.cache_pool,
        )
        self.kv_store = MlxPagedKVStore(
            self.model.mx,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=self.model.head_dim,
            block_tokens=self.block_tokens,
        )
        self._prefill_offsets: dict[int, int] = {}
        projections = {
            name: info
            for name, info in self.model.file.tensors.items()
            if name.startswith("model.layers.") and name.endswith("_proj.weight")
        }
        original_bytes = sum(
            info.nbytes * (2 if name in self.model.file.quantization else 1)
            for name, info in projections.items()
        )
        quantized_bytes = sum(
            projections[name].nbytes * 2 for name in self.model.file.quantization
        )
        self._precision_stats = {
            "retained_fp16_projections": sorted(
                set(projections) - set(self.model.file.quantization)
            ),
            "quantized_projection_fraction": quantized_bytes / original_bytes
            if original_bytes
            else 0.0,
        }

    def submit(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
    ) -> int:
        return self.scheduler.submit(input_ids, max_new_tokens, eos_token_ids)

    def step(self) -> list[TokenEvent]:
        mx = self.model.mx
        events: list[TokenEvent] = []

        schedule = self.scheduler.next()
        page_snapshots = {
            request_id: self.cache_pool.block_table(request_id)
            for request_id in schedule.decode
        }
        if schedule.prefill is not None:
            page_snapshots[schedule.prefill] = self.cache_pool.block_table(
                schedule.prefill
            )
        try:
            if schedule.decode:
                requests = [
                    self.scheduler.request(request_id) for request_id in schedule.decode
                ]
                logits = self.model.decode_paged_batch(
                    [request.output[-1] for request in requests],
                    positions=[request.live_tokens - 1 for request in requests],
                    block_tables=[
                        page_snapshots[request_id] for request_id in schedule.decode
                    ],
                    store=self.kv_store,
                )
                tokens = [int(token) for token in mx.argmax(logits, axis=-1).tolist()]
                for request_id, token in zip(schedule.decode, tokens):
                    finished = self.scheduler.append_token(request_id, token)
                    if finished:
                        self.kv_store.release(page_snapshots[request_id])
                    events.append(TokenEvent(request_id, token, finished))

            if schedule.prefill is not None:
                request_id = schedule.prefill
                request = self.scheduler.request(request_id)
                offset = self._prefill_offsets.get(request_id, 0)
                end = min(offset + self.prefill_chunk_size, len(request.prompt))
                logits = self.model.forward_paged_chunk(
                    request.prompt[offset:end],
                    start_position=offset,
                    block_table=page_snapshots[request_id],
                    store=self.kv_store,
                )
                if end < len(request.prompt):
                    self._prefill_offsets[request_id] = end
                else:
                    self._prefill_offsets.pop(request_id, None)
                    token = int(mx.argmax(logits[-1]).item())
                    self.scheduler.finish_prefill(request_id)
                    finished = self.scheduler.append_token(request_id, token)
                    if finished:
                        self.kv_store.release(page_snapshots[request_id])
                    events.append(TokenEvent(request_id, token, finished))
        except Exception:
            for request_id, physical_ids in page_snapshots.items():
                self.scheduler.cancel(request_id)
                self.kv_store.release(physical_ids)
                self._prefill_offsets.pop(request_id, None)
            raise
        return events

    def cancel(self, request_id: int) -> None:
        request_id = int(request_id)
        physical_ids = (
            self.cache_pool.block_table(request_id)
            if self.cache_pool.contains(request_id)
            else ()
        )
        self.scheduler.cancel(request_id)
        self.kv_store.release(physical_ids)
        self._prefill_offsets.pop(request_id, None)

    def generate(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
    ) -> list[int]:
        request_id = self.submit(input_ids, max_new_tokens, eos_token_ids)
        while True:
            self.step()
            request = self.scheduler.request(request_id)
            if request.state is SequenceState.COMPLETED:
                return list(request.output)
            if request.state in {SequenceState.CANCELLED, SequenceState.REJECTED}:
                raise RuntimeError("generation terminated without output")

    def debug_prefill_logits(self, input_ids: Iterable[int]) -> list[float]:
        tokens = [int(token) for token in input_ids]
        if not tokens or len(tokens) > self.max_model_length:
            raise ValueError(
                "debug prompt length is outside the configured context limit"
            )
        block_count = (len(tokens) + self.block_tokens - 1) // self.block_tokens
        temporary = MlxPagedKVStore(
            self.model.mx,
            num_layers=self.model.config.num_hidden_layers,
            num_kv_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.head_dim,
            block_tokens=self.block_tokens,
        )
        table = tuple(range(block_count))
        logits = None
        try:
            for start in range(0, len(tokens), self.prefill_chunk_size):
                logits = self.model.forward_paged_chunk(
                    tokens[start : start + self.prefill_chunk_size],
                    start_position=start,
                    block_table=table,
                    store=temporary,
                )
            if logits is None:
                raise RuntimeError("debug prefill produced no logits")
            return logits[-1].tolist()
        finally:
            temporary.clear()

    def stats(self) -> dict[str, object]:
        return {
            "backend": "mlx",
            "model_type": self.model.config.model_type,
            "scheduler": self.scheduler.stats(),
            "kv_cache": self.cache_pool.stats(),
            "model_bytes": self.model.file.data_size,
            "weight_storage_bytes": sum(
                {
                    info.offset: info.nbytes
                    for info in self.model.file.tensors.values()
                }.values()
            ),
            "quantization": (
                "symmetric_int8_per_output_channel"
                if self.model.file.quantization
                else "fp16"
            ),
            "quantized_matrices": len(self.model.file.quantization),
            "quantization_status": "experimental"
            if self.model.file.quantization
            else "not_quantized",
            "int8_mode": self.model.int8_mode,
            **self._precision_stats,
            "kv_device_bytes": self.kv_store.allocated_bytes,
            "kv_capacity_bytes": self.cache_pool.total_blocks * self.bytes_per_block,
            "kv_materialized_blocks": self.kv_store.allocated_blocks,
            "kv_peak_materialized_blocks": self.kv_store.peak_allocated_blocks,
            "kv_peak_device_bytes": (
                self.kv_store.peak_allocated_blocks * self.kv_store.bytes_per_block
            ),
            "prefill_chunk_size": self.prefill_chunk_size,
            "attention_tile_size": self.model.attention_tile_size,
            "custom_metal": self.model.custom_metal,
            "metal_paged_attention": self.model.metal_paged_attention,
        }

    def build_info(self) -> dict[str, object]:
        return mlx_build_info()

    def close(self) -> None:
        self._prefill_offsets.clear()
        self.kv_store.clear()
        self.model.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
