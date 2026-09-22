"""Public request-oriented Engine implementation for the MLX backend."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .backends.mlx import MlxQwenModel, QwenKVCache, mlx_build_info
from .runtime import IterationScheduler, KVBlockPool, SequenceState, TokenEvent


class MlxEngine:
    """Greedy Qwen engine with the same request API as the CUDA extension.

    This checkpoint deliberately stores per-sequence contiguous MLX K/V arrays.
    The scheduler-facing block accounting is already reservation based, while
    physical paged Metal storage belongs to the next long-context milestone.
    """

    backend = "mlx"
    block_tokens = 16

    def __init__(self, model_path: str | Path, max_num_sequences: int = 16,
                 max_model_length: int = 2048, kv_cache_bytes: int = 512 << 20) -> None:
        if max_num_sequences <= 0:
            raise ValueError("max_num_sequences must be positive")
        if kv_cache_bytes <= 0:
            raise ValueError("kv_cache_bytes must be positive")
        self.model = MlxQwenModel(str(model_path), max_model_length=max_model_length)
        self.max_num_sequences = max_num_sequences
        self.max_model_length = max_model_length
        config = self.model.config
        self.bytes_per_token = (
            config.num_hidden_layers * 2 * config.num_key_value_heads
            * self.model.head_dim * 2
        )
        self.bytes_per_block = self.bytes_per_token * self.block_tokens
        self.cache_pool = KVBlockPool(kv_cache_bytes, self.bytes_per_token, self.block_tokens)
        self.scheduler = IterationScheduler(
            max_num_sequences,
            max_model_length,
            config.vocab_size,
            self.cache_pool,
        )
        self._caches: dict[int, QwenKVCache] = {}

    def submit(self, input_ids: Iterable[int], max_new_tokens: int,
               eos_token_ids: Iterable[int] = ()) -> int:
        return self.scheduler.submit(input_ids, max_new_tokens, eos_token_ids)

    def step(self) -> list[TokenEvent]:
        mx = self.model.mx
        events: list[TokenEvent] = []

        schedule = self.scheduler.next()
        try:
            for request_id in schedule.decode:
                request = self.scheduler.request(request_id)
                logits, cache = self.model.forward([request.output[-1]], self._caches[request_id])
                self._caches[request_id] = cache
                token = int(mx.argmax(logits[-1]).item())
                finished = self.scheduler.append_token(request_id, token)
                if finished:
                    self._caches.pop(request_id, None)
                events.append(TokenEvent(request_id, token, finished))

            if schedule.prefill is not None:
                request_id = schedule.prefill
                request = self.scheduler.request(request_id)
                logits, cache = self.model.forward(request.prompt)
                self._caches[request_id] = cache
                token = int(mx.argmax(logits[-1]).item())
                self.scheduler.finish_prefill(request_id)
                finished = self.scheduler.append_token(request_id, token)
                if finished:
                    self._caches.pop(request_id, None)
                events.append(TokenEvent(request_id, token, finished))
        except Exception:
            for request_id in schedule.decode:
                self.scheduler.cancel(request_id)
                self._caches.pop(request_id, None)
            if schedule.prefill is not None:
                self.scheduler.cancel(schedule.prefill)
                self._caches.pop(schedule.prefill, None)
            raise
        return events

    def cancel(self, request_id: int) -> None:
        self.scheduler.cancel(request_id)
        self._caches.pop(int(request_id), None)

    def generate(self, input_ids: Iterable[int], max_new_tokens: int,
                 eos_token_ids: Iterable[int] = ()) -> list[int]:
        request_id = self.submit(input_ids, max_new_tokens, eos_token_ids)
        while True:
            self.step()
            request = self.scheduler.request(request_id)
            if request.state is SequenceState.COMPLETED:
                return list(request.output)
            if request.state in {SequenceState.CANCELLED, SequenceState.REJECTED}:
                raise RuntimeError("generation terminated without output")

    def debug_prefill_logits(self, input_ids: Iterable[int]) -> list[float]:
        return self.model.prefill_logits([int(token) for token in input_ids]).tolist()

    def stats(self) -> dict[str, object]:
        return {
            "backend": "mlx",
            "scheduler": self.scheduler.stats(),
            "kv_cache": self.cache_pool.stats(),
            "model_bytes": self.model.file.data_size,
            "kv_device_bytes": self.cache_pool.total_blocks * self.bytes_per_block,
        }

    def build_info(self) -> dict[str, object]:
        return mlx_build_info()

    def close(self) -> None:
        self._caches.clear()
        self.model.close()

    def __enter__(self) -> "MlxEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
