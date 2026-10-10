"""Public request-oriented Engine implementation for the MLX backend."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Self

from .backends.factory import create_mlx_model
from .backends.mlx import mlx_build_info
from .paged_kv import MlxPagedKVStore
from .prefix_cache import PrefixCache, SharedKVBlockPool
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
        decode_mode: str = "batched",
        prefix_cache_bytes: int = 0,
        prefix_cache_max_entries: int = 64,
        prefix_cache_namespace: str = "",
    ) -> None:
        if max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be positive")
        if kv_cache_bytes <= 0:
            raise ValueError("kv_cache_bytes must be positive")
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if decode_mode not in {"batched", "rowwise"}:
            raise ValueError("decode_mode must be batched or rowwise")
        if (
            type(prefix_cache_bytes) is not int
            or not 0 <= prefix_cache_bytes <= kv_cache_bytes
        ):
            raise ValueError("prefix_cache_bytes must be between zero and KV capacity")
        if type(prefix_cache_max_entries) is not int or prefix_cache_max_entries <= 0:
            raise ValueError("prefix_cache_max_entries must be positive")
        PrefixCache.validate_namespace(prefix_cache_namespace)
        self.model = create_mlx_model(
            str(model_path),
            max_model_length=max_model_length,
            attention_tile_size=attention_tile_size,
            custom_metal=custom_metal,
            metal_paged_attention=metal_paged_attention,
            int8_mode=int8_mode,
            decode_mode=decode_mode,
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
        if prefix_cache_bytes and (
            self.model.file.quantization or prefix_cache_bytes < self.bytes_per_block
        ):
            self.model.close()
            raise ValueError(
                "prefix caching requires FP16 and capacity for at least one page"
            )
        self._closed = False
        pool_type = SharedKVBlockPool if prefix_cache_bytes else KVBlockPool
        self.cache_pool = pool_type(
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
            is_shared=self.cache_pool.is_shared if prefix_cache_bytes else None,
        )
        self.prefix_cache = None
        self._prefix_matches = {}
        self._request_namespaces: dict[int, str] = {}
        self._prefix_settings = self._execution_settings()
        if prefix_cache_bytes:
            identity = {
                "schema": "forge_prefix_fp16_v1",
                "model_data_sha256": self.model.file.data_sha256,
                "config": asdict(config),
                "settings": self._prefix_settings,
                "namespace": prefix_cache_namespace,
                "block_tokens": self.block_tokens,
            }
            self.prefix_cache = PrefixCache(
                self.cache_pool,
                self.kv_store,
                max_blocks=prefix_cache_bytes // self.bytes_per_block,
                max_entries=prefix_cache_max_entries,
                chunk_size=prefill_chunk_size,
                identity=hashlib.sha256(
                    json.dumps(identity, sort_keys=True).encode()
                ).hexdigest(),
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
        *,
        cache_namespace: str = "",
    ) -> int:
        self._check_open()
        PrefixCache.validate_namespace(cache_namespace)
        prompt, eos = [int(t) for t in input_ids], [int(t) for t in eos_token_ids]
        match = None
        valid = (
            bool(prompt)
            and max_new_tokens > 0
            and len(prompt) + max_new_tokens <= self.max_model_length
            and all(0 <= t < self.model.config.vocab_size for t in prompt + eos)
            and self.scheduler.stats()["waiting"] + self.scheduler.stats()["running"]
            < self.max_num_sequences
        )
        if self.prefix_cache and valid:
            match = self.prefix_cache.find(prompt, cache_namespace)
            self.prefix_cache.make_admission_room(
                len(prompt) + int(max_new_tokens), match
            )
        try:
            request_id = self.scheduler.submit(
                prompt,
                max_new_tokens,
                eos,
                shared_blocks=match.blocks if match else (),
                shared_tokens=len(match.tokens) if match else 0,
            )
        except ValueError:
            if self.prefix_cache:
                self.prefix_cache.counters["admission_rejections"] += 1
            raise
        if self.prefix_cache:
            self.prefix_cache.record_lookup(match)
            self._request_namespaces[request_id] = cache_namespace
            if match:
                self._prefix_matches[request_id] = match
                self._prefill_offsets[request_id] = len(match.tokens)
        return request_id

    def _execution_settings(self):
        return (
            self.prefill_chunk_size,
            self.max_model_length,
            self.model.attention_tile_size,
            self.model.custom_metal,
            self.model.metal_paged_attention,
            self.model.decode_mode,
            self.model.int8_mode,
        )

    def _check_open(self):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self.prefix_cache and self._execution_settings() != self._prefix_settings:
            raise RuntimeError("prefix execution settings changed; create a new engine")

    def _release_unowned(self, blocks):
        self.kv_store.release(
            [b for b in blocks if not self.cache_pool.is_referenced(b)]
            if self.prefix_cache
            else blocks
        )

    def _prepare_write(self, request_id, start, count):
        if self.prefix_cache:
            self.cache_pool.make_writable(
                request_id, start, count, self.kv_store.clone_block
            )
        return self.cache_pool.block_table(request_id)

    def _prefill_chunk(self, request_id):
        request = self.scheduler.request(request_id)
        offset = self._prefill_offsets.get(request_id, 0)
        match = self._prefix_matches.pop(request_id, None)
        if match:
            self.prefix_cache.counters["reused_tokens"] += len(match.tokens)
            if offset == len(request.prompt):
                return self.model.mx.array(match.logits[None, :]), offset
        end = min(offset + self.prefill_chunk_size, len(request.prompt))
        table = self._prepare_write(request_id, offset, end - offset)
        logits = self.model.forward_paged_chunk(
            request.prompt[offset:end],
            start_position=offset,
            block_table=table,
            store=self.kv_store,
        )
        if self.prefix_cache:
            self.prefix_cache.put(
                request.prompt[:end],
                self._request_namespaces[request_id],
                table,
                logits[-1],
            )
        return logits, end

    def step(self) -> list[TokenEvent]:
        self._check_open()
        mx = self.model.mx
        events: list[TokenEvent] = []

        try:
            schedule = self.scheduler.next()
        except Exception:
            # Admission reserves capacity, so this is an exceptional allocator
            # failure. The scheduler may already have moved a waiting request
            # to PREFILLING; reclaim that request without disturbing decodes.
            if self.scheduler.prefilling is not None:
                self.cancel(self.scheduler.prefilling)
            raise
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
                for request in requests:
                    page_snapshots[request.request_id] = self._prepare_write(
                        request.request_id, request.live_tokens - 1, 1
                    )
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
                        self._release_unowned(page_snapshots[request_id])
                        self._request_namespaces.pop(request_id, None)
                    events.append(TokenEvent(request_id, token, finished))

            if schedule.prefill is not None:
                request_id = schedule.prefill
                request = self.scheduler.request(request_id)
                logits, end = self._prefill_chunk(request_id)
                page_snapshots[request_id] = self.cache_pool.block_table(request_id)
                if end < len(request.prompt):
                    self._prefill_offsets[request_id] = end
                else:
                    self._prefill_offsets.pop(request_id, None)
                    token = int(mx.argmax(logits[-1]).item())
                    self.scheduler.finish_prefill(request_id)
                    finished = self.scheduler.append_token(request_id, token)
                    if finished:
                        self._release_unowned(page_snapshots[request_id])
                        self._request_namespaces.pop(request_id, None)
                    events.append(TokenEvent(request_id, token, finished))
        except Exception:
            for request_id, physical_ids in page_snapshots.items():
                current = (
                    self.cache_pool.block_table(request_id)
                    if self.cache_pool.contains(request_id)
                    else ()
                )
                self.cancel(request_id)
                self._release_unowned((*physical_ids, *current))
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
        self._release_unowned(physical_ids)
        self._prefill_offsets.pop(request_id, None)
        self._prefix_matches.pop(request_id, None)
        self._request_namespaces.pop(request_id, None)

    def forget(self, request_id: int) -> None:
        """Release completed/cancelled request history in long-lived workers."""
        self.scheduler.forget(request_id)

    def clear_prefix_cache(self, cache_namespace: str | None = None) -> None:
        """Evict cache pins without cancelling or invalidating admitted requests."""
        if self.prefix_cache:
            self.prefix_cache.clear(cache_namespace)

    def generate(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
        *,
        cache_namespace: str = "",
    ) -> list[int]:
        request_id = self.submit(
            input_ids, max_new_tokens, eos_token_ids, cache_namespace=cache_namespace
        )
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
            "decode_mode": self.model.decode_mode,
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
            "prefix_cache": {
                **(
                    self.prefix_cache.stats()
                    if self.prefix_cache
                    else {"enabled": False, "entries": 0, "cached_blocks": 0}
                ),
                "pending_match_logits_bytes": sum(
                    e.logits.nbytes for e in self._prefix_matches.values()
                ),
            },
        }

    def build_info(self) -> dict[str, object]:
        return mlx_build_info()

    def close(self) -> None:
        if self._closed:
            return
        for request in self.scheduler.requests.values():
            if request.state in {
                SequenceState.WAITING,
                SequenceState.PREFILLING,
                SequenceState.RUNNING,
            }:
                self.cancel(request.request_id)
        self.clear_prefix_cache()
        self._prefill_offsets.clear()
        self.kv_store.clear()
        self.model.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
