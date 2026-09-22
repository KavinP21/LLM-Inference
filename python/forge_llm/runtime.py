"""Backend-neutral request scheduling and KV-capacity accounting.

The C++ CUDA runtime implements the same state machine.  Keeping the Python
version free of MLX imports makes its invariants independently testable and
gives future Apple execution code a narrow orchestration boundary.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class SequenceState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TokenEvent:
    request_id: int
    token: int
    finished: bool


@dataclass(frozen=True)
class Schedule:
    prefill: int | None
    decode: tuple[int, ...]


@dataclass
class Request:
    request_id: int
    prompt: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    state: SequenceState = SequenceState.WAITING
    output: list[int] = field(default_factory=list)

    @property
    def live_tokens(self) -> int:
        return len(self.prompt) + len(self.output)

    @property
    def maximum_tokens(self) -> int:
        return len(self.prompt) + self.max_new_tokens


@dataclass
class _Allocation:
    reservation: int
    live_tokens: int = 0
    allocated_blocks: int = 0


class KVBlockPool:
    """Reservation-aware logical block allocator.

    This checkpoint accounts for MLX's contiguous K/V arrays in 16-token units.
    The interface intentionally matches the eventual physical paged allocator.
    """

    def __init__(self, capacity_bytes: int, bytes_per_token: int, block_tokens: int = 16) -> None:
        if capacity_bytes <= 0 or bytes_per_token <= 0 or block_tokens <= 0:
            raise ValueError("invalid KV cache dimensions")
        self.block_tokens = block_tokens
        self.bytes_per_block = bytes_per_token * block_tokens
        self.total_blocks = capacity_bytes // self.bytes_per_block
        if self.total_blocks <= 0:
            raise ValueError("KV cache is too small for one block")
        self.allocations: dict[int, _Allocation] = {}
        self.allocation_failures = 0

    def _blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.block_tokens) if tokens else 0

    def reserve(self, request_id: int, maximum_tokens: int) -> bool:
        if request_id in self.allocations or maximum_tokens <= 0:
            raise ValueError("invalid or duplicate KV reservation")
        blocks = self._blocks_for(maximum_tokens)
        reserved = sum(allocation.reservation for allocation in self.allocations.values())
        if reserved + blocks > self.total_blocks:
            self.allocation_failures += 1
            return False
        self.allocations[request_id] = _Allocation(blocks)
        return True

    def ensure_tokens(self, request_id: int, live_tokens: int) -> None:
        try:
            allocation = self.allocations[request_id]
        except KeyError as exc:
            raise KeyError("request has no KV reservation") from exc
        blocks = self._blocks_for(live_tokens)
        if live_tokens < allocation.live_tokens or blocks > allocation.reservation:
            raise ValueError("request violated its KV reservation")
        allocation.live_tokens = live_tokens
        allocation.allocated_blocks = blocks

    def release(self, request_id: int) -> None:
        self.allocations.pop(request_id, None)

    def stats(self) -> dict[str, int | float | str]:
        allocated = sum(item.allocated_blocks for item in self.allocations.values())
        reserved = sum(item.reservation - item.allocated_blocks for item in self.allocations.values())
        live_tokens = sum(item.live_tokens for item in self.allocations.values())
        return {
            "layout": "contiguous",
            "total_blocks": self.total_blocks,
            "allocated_blocks": allocated,
            "reserved_blocks": reserved,
            "free_blocks": self.total_blocks - allocated,
            "live_tokens": live_tokens,
            "internal_fragmentation_tokens": allocated * self.block_tokens - live_tokens,
            "allocation_failures": self.allocation_failures,
            "occupancy": allocated / self.total_blocks,
        }


class IterationScheduler:
    """FCFS scheduler: all runnable decodes plus at most one prompt per step."""

    def __init__(self, max_sequences: int, max_model_length: int,
                 vocab_size: int, cache: KVBlockPool) -> None:
        if max_sequences <= 0 or max_model_length <= 0 or vocab_size <= 0:
            raise ValueError("invalid scheduler limits")
        self.max_sequences = max_sequences
        self.max_model_length = max_model_length
        self.vocab_size = vocab_size
        self.cache = cache
        self.next_id = 1
        self.waiting: deque[int] = deque()
        self.running: list[int] = []
        self.requests: dict[int, Request] = {}
        self.totals = {"submitted": 0, "completed": 0, "cancelled": 0, "rejected": 0}

    @staticmethod
    def _tokens(values: Iterable[int]) -> list[int]:
        return [int(value) for value in values]

    def submit(self, input_ids: Iterable[int], max_new_tokens: int,
               eos_token_ids: Iterable[int]) -> int:
        prompt = self._tokens(input_ids)
        eos = frozenset(self._tokens(eos_token_ids))
        request_id = self.next_id
        self.next_id += 1
        self.totals["submitted"] += 1
        request = Request(request_id, prompt, int(max_new_tokens), eos)
        active_states = {SequenceState.WAITING, SequenceState.PREFILLING, SequenceState.RUNNING}
        active = sum(item.state in active_states for item in self.requests.values())
        invalid = (
            not prompt
            or max_new_tokens <= 0
            or request.maximum_tokens > self.max_model_length
            or any(token < 0 or token >= self.vocab_size for token in prompt)
            or any(token < 0 or token >= self.vocab_size for token in eos)
        )
        if invalid or active >= self.max_sequences or not self.cache.reserve(
                request_id, request.maximum_tokens):
            request.state = SequenceState.REJECTED
            self.requests[request_id] = request
            self.totals["rejected"] += 1
            raise ValueError(
                "request rejected: invalid length, token id, concurrency limit, "
                "or insufficient KV capacity"
            )
        self.requests[request_id] = request
        self.waiting.append(request_id)
        return request_id

    def next(self) -> Schedule:
        prefill = None
        if self.waiting:
            prefill = self.waiting.popleft()
            request = self.request(prefill)
            request.state = SequenceState.PREFILLING
            self.cache.ensure_tokens(prefill, len(request.prompt))
        return Schedule(prefill, tuple(self.running))

    def finish_prefill(self, request_id: int) -> None:
        request = self.request(request_id)
        if request.state is not SequenceState.PREFILLING:
            raise RuntimeError("request is not prefilling")
        request.state = SequenceState.RUNNING
        self.running.append(request_id)

    def append_token(self, request_id: int, token: int) -> bool:
        request = self.request(request_id)
        if request.state is not SequenceState.RUNNING:
            raise RuntimeError("request is not running")
        request.output.append(int(token))
        finished = (
            token in request.eos_token_ids
            or len(request.output) >= request.max_new_tokens
            or request.live_tokens >= self.max_model_length
        )
        if finished:
            self._finish(request, SequenceState.COMPLETED)
        else:
            self.cache.ensure_tokens(request_id, request.live_tokens)
        return finished

    def cancel(self, request_id: int) -> None:
        request = self.request(request_id)
        if request.state in {
            SequenceState.COMPLETED,
            SequenceState.CANCELLED,
            SequenceState.REJECTED,
        }:
            return
        try:
            self.waiting.remove(request_id)
        except ValueError:
            pass
        self._finish(request, SequenceState.CANCELLED)

    def _finish(self, request: Request, state: SequenceState) -> None:
        try:
            self.running.remove(request.request_id)
        except ValueError:
            pass
        self.cache.release(request.request_id)
        request.state = state
        self.totals["completed" if state is SequenceState.COMPLETED else "cancelled"] += 1

    def request(self, request_id: int) -> Request:
        try:
            return self.requests[int(request_id)]
        except KeyError as exc:
            raise KeyError(f"unknown request id {request_id}") from exc

    def stats(self) -> dict[str, int]:
        return {
            **self.totals,
            "waiting": len(self.waiting),
            "running": sum(
                item.state in {SequenceState.PREFILLING, SequenceState.RUNNING}
                for item in self.requests.values()
            ),
        }

