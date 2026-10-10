"""Greedy draft-and-verify decoding with owned, rollback-safe paged K/V.

The portable controller is independent of MLX. The MLX adapter verifies a
causal block in one forward pass; it does not call the target once per proposal.
Native FP16 block execution has an explicit numerical contract (see docs).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Protocol


class DraftProvider(Protocol):
    def start(self, prompt: Sequence[int], max_new_tokens: int) -> None: ...
    def propose(
        self, history: Sequence[int], count: int, cancelled: Callable[[], bool]
    ) -> Sequence[int]: ...
    def finish(self) -> None: ...


class GreedyTarget(Protocol):
    vocab_size: int
    max_model_length: int
    cached_tokens: int

    def start(self, prompt: Sequence[int], max_new_tokens: int) -> int: ...
    def append(self, tokens: Sequence[int], *, block: bool) -> list[int]: ...
    def truncate(self, token_count: int) -> None: ...
    def finish(self) -> None: ...


class NGramDraft:
    """Copy continuations of a repeated suffix from the existing token history.

    Useful for code, quotations, repeated formatting and source-copying tasks.
    No second model, tokenizer or additional device memory is needed. The most
    recent match at the longest available n-gram wins deterministically.
    """

    def __init__(
        self, min_match: int = 2, max_match: int = 8, max_search_tokens: int = 4096
    ) -> None:
        if any(type(v) is not int for v in (min_match, max_match, max_search_tokens)):
            raise ValueError("n-gram bounds must be integers")
        if not 0 < min_match <= max_match <= max_search_tokens:
            raise ValueError("invalid n-gram bounds")
        self.min_match, self.max_match = min_match, max_match
        self.max_search_tokens = max_search_tokens

    def start(self, prompt: Sequence[int], max_new_tokens: int) -> None:
        pass

    def propose(
        self,
        history: Sequence[int],
        count: int,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> list[int]:
        if count <= 0 or cancelled():
            return []
        tokens = list(history[-self.max_search_tokens :])
        for size in range(min(self.max_match, len(tokens) - 1), self.min_match - 1, -1):
            suffix = tokens[-size:]
            for start in range(len(tokens) - size - 1, -1, -1):
                if tokens[start : start + size] == suffix:
                    return tokens[start + size : start + size + count]
        return []

    def finish(self) -> None:
        pass


class NoDraft(NGramDraft):
    """Canonical one-token decode baseline with the same adapter and prefill."""

    def propose(self, history, count, cancelled=lambda: False):
        return []


@dataclass
class SpeculativeStats:
    output_tokens: int = 0
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    verification_blocks: int = 0
    target_decode_calls: int = 0
    rollback_tokens: int = 0
    draft_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    prefill_calls: int = 0
    peak_kv_bytes: int = 0

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["acceptance_rate"] = (
            self.accepted_draft_tokens / self.draft_tokens if self.draft_tokens else 0.0
        )
        return result


@dataclass(frozen=True)
class SpeculativeResult:
    tokens: list[int]
    finish_reason: str
    stats: SpeculativeStats
    verification_mode: str


class GenerationCancelled(RuntimeError):
    def __init__(self, result: SpeculativeResult) -> None:
        super().__init__("speculative generation was cancelled")
        self.result = result


def _token_ids(values: Iterable[int], vocab_size: int, name: str) -> list[int]:
    # Reject lossy coercion (floats/bools), including in user-provided drafts.
    import operator

    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain integer token IDs")  # noqa: TRY004 - consistent token-validation API
        try:
            token = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must contain integer token IDs") from exc
        if not 0 <= token < vocab_size:
            raise ValueError(f"{name} token is outside the vocabulary")
        result.append(token)
    return result


class GreedySpeculator:
    """Portable, bounded greedy speculative controller.

    Cache invariant: after each emitted block, all history except its last
    output token is cached. Verify ``pending + proposals`` once. Each row
    predicts the next proposal; the final row supplies a bonus token. On a
    mismatch, retain only pending + accepted proposals and emit the correction.
    """

    def __init__(
        self,
        target: GreedyTarget,
        draft: DraftProvider | None = None,
        *,
        draft_tokens: int = 4,
        verification_mode: str = "block",
        adaptive: bool = True,
    ) -> None:
        if type(draft_tokens) is not int or draft_tokens < 1:
            raise ValueError("draft_tokens must be a positive integer")
        if verification_mode not in {"block", "sequential"}:
            raise ValueError("verification_mode must be block or sequential")
        self.target = target
        self.draft = draft if draft is not None else NGramDraft()
        self.draft_tokens = draft_tokens
        self.verification_mode = verification_mode
        self.adaptive = adaptive
        self._lock = threading.Lock()

    def generate_result(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] = (),
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> SpeculativeResult:
        prompt = _token_ids(input_ids, self.target.vocab_size, "prompt")
        eos = set(_token_ids(eos_token_ids, self.target.vocab_size, "EOS"))
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not prompt or len(prompt) + max_new_tokens > self.target.max_model_length:
            raise ValueError("prompt plus output budget exceeds the context limit")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "a speculative engine instance runs one request at a time"
            )
        if getattr(self, "_closed", False):
            # Close may have completed after the adapter's initial open check.
            self._lock.release()
            raise RuntimeError("engine is closed")
        cancellation_callback = cancelled or (lambda: False)
        cancellation_requested = False

        def cancelled() -> bool:
            nonlocal cancellation_requested
            # Once observed, cancellation remains effective even if a caller's
            # callback subsequently clears its event or changes its answer.
            cancellation_requested = cancellation_requested or bool(
                cancellation_callback()
            )
            return cancellation_requested

        stats, output = SpeculativeStats(), []
        started = time.perf_counter()
        reason, budget = "length", self.draft_tokens
        try:
            for participant in (self.target, self.draft):
                if hasattr(participant, "set_cancelled"):
                    participant.set_cancelled(cancelled)
            if cancelled():
                reason = "cancelled"
            else:
                first = self.target.start(prompt, max_new_tokens)
                if not 0 <= first < self.target.vocab_size:
                    raise RuntimeError("target returned an invalid greedy token")
                if cancelled():
                    reason = "cancelled"
                else:
                    self.draft.start(prompt, max_new_tokens)
                    if cancelled():
                        reason = "cancelled"
                    else:
                        output.append(first)
                while output and len(output) < max_new_tokens and output[-1] not in eos:
                    if cancelled():
                        reason = "cancelled"
                        break
                    remaining = max_new_tokens - len(output)
                    history = prompt + output
                    begin_draft = time.perf_counter()
                    allowed = min(budget, remaining - 1)
                    raw = self.draft.propose(history, allowed, cancelled)
                    # Read at most one excess token. An accidental unbounded
                    # iterator cannot bypass the request's resource budget.
                    proposals = _token_ids(
                        islice(raw, allowed + 1), self.target.vocab_size, "draft"
                    )
                    stats.draft_seconds += time.perf_counter() - begin_draft
                    if len(proposals) > allowed:
                        raise ValueError("draft provider exceeded its token budget")
                    for index, token in enumerate(proposals):
                        if token in eos:
                            proposals = proposals[: index + 1]
                            break
                    if cancelled():
                        reason = "cancelled"
                        break
                    before = self.target.cached_tokens
                    if before != len(history) - 1:
                        raise RuntimeError(
                            "target violated the committed K/V invariant"
                        )
                    inputs = [output[-1]] + proposals
                    block = bool(proposals) and self.verification_mode == "block"
                    predictions = self.target.append(inputs, block=block)
                    if self.target.cached_tokens != before + len(inputs):
                        raise RuntimeError(
                            "target failed to cache the verification block"
                        )
                    if len(predictions) != len(inputs) or any(
                        type(token) is not int
                        or not 0 <= token < self.target.vocab_size
                        for token in predictions
                    ):
                        raise RuntimeError(
                            "target returned invalid verification predictions"
                        )
                    stats.target_decode_calls += 1 if block else len(inputs)
                    stats.verification_blocks += int(block)
                    stats.draft_tokens += len(proposals)
                    accepted = 0
                    while (
                        accepted < len(proposals)
                        and proposals[accepted] == predictions[accepted]
                    ):
                        accepted += 1
                    # Prefix is causally independent of the rejected suffix.
                    keep = before + 1 + accepted
                    rollback = len(proposals) - accepted
                    if rollback:
                        self.target.truncate(keep)
                        stats.rollback_tokens += rollback
                    stats.accepted_draft_tokens += accepted
                    stats.rejected_draft_tokens += rollback
                    if cancelled():
                        reason = "cancelled"
                        break
                    additions = proposals[:accepted] + [predictions[accepted]]
                    for token in additions:
                        output.append(token)
                        if token in eos or len(output) == max_new_tokens:
                            break
                    if self.adaptive and proposals:
                        if accepted == len(proposals):
                            budget = min(self.draft_tokens, budget + 1)
                        elif accepted * 2 < len(proposals):
                            budget = max(1, budget // 2)
                if reason != "cancelled" and output and output[-1] in eos:
                    reason = "eos"
        finally:
            try:
                self.draft.finish()
            finally:
                try:
                    stats.prefill_calls = getattr(self.target, "prefill_calls", 0)
                    stats.peak_kv_bytes = getattr(self.target, "peak_kv_bytes", 0)
                    self.target.finish()
                finally:
                    stats.output_tokens = len(output)
                    stats.elapsed_seconds = time.perf_counter() - started
                    self._lock.release()
        return SpeculativeResult(output, reason, stats, self.verification_mode)

    def generate(self, input_ids, max_new_tokens, eos_token_ids=(), *, cancelled=None):
        result = self.generate_result(
            input_ids, max_new_tokens, eos_token_ids, cancelled=cancelled
        )
        if result.finish_reason == "cancelled":
            raise GenerationCancelled(result)
        return result.tokens


class _MlxTarget:
    def __init__(self, model: Any, kv_cache_bytes: int, prefill_chunk_size: int):
        from .paged_kv import MlxPagedKVStore

        self.model = model
        self.vocab_size = model.config.vocab_size
        self.max_model_length = model.max_model_length
        self.kv_cache_bytes = kv_cache_bytes
        self.prefill_chunk_size = prefill_chunk_size
        self.bytes_per_block = (
            model.config.num_hidden_layers
            * 2
            * 16
            * model.config.num_key_value_heads
            * model.head_dim
            * 2
        )
        self.store = MlxPagedKVStore(
            model.mx,
            num_layers=model.config.num_hidden_layers,
            num_kv_heads=model.config.num_key_value_heads,
            head_dim=model.head_dim,
        )
        self.table: tuple[int, ...] = ()
        self.cached_tokens = self.prefill_calls = self.peak_kv_bytes = 0
        self.cancelled: Callable[[], bool] = lambda: False

    def set_cancelled(self, cancelled: Callable[[], bool]) -> None:
        self.cancelled = cancelled

    def _greedy(self, logits) -> list[int]:
        return [int(token) for token in self.model.mx.argmax(logits, axis=-1).tolist()]

    def start(self, prompt: Sequence[int], max_new_tokens: int) -> int:
        self.finish()
        blocks = (len(prompt) + max_new_tokens + 15) // 16
        if blocks * self.bytes_per_block > self.kv_cache_bytes:
            raise ValueError("KV capacity cannot reserve prompt plus output budget")
        self.table = tuple(range(blocks))
        self.prefill_calls = self.peak_kv_bytes = 0
        logits = None
        for start in range(0, len(prompt), self.prefill_chunk_size):
            if self.cancelled():
                return 0  # Controller observes cancellation before emitting.
            tokens = prompt[start : start + self.prefill_chunk_size]
            logits = self.model.forward_paged_chunk(
                tokens,
                start_position=start,
                block_table=self.table,
                store=self.store,
            )
            self.cached_tokens += len(tokens)
            self.prefill_calls += 1
            self.peak_kv_bytes = max(self.peak_kv_bytes, self.store.allocated_bytes)
        return self._greedy(logits[-1:])[0]

    def append(self, tokens: Sequence[int], *, block: bool) -> list[int]:
        predictions = []
        if block:
            logits = self.model.forward_paged_chunk(
                list(tokens),
                start_position=self.cached_tokens,
                block_table=self.table,
                store=self.store,
            )
            self.cached_tokens += len(tokens)
            predictions = self._greedy(logits)
        else:
            for token in tokens:
                logits = self.model.decode_paged_batch(
                    [token],
                    positions=[self.cached_tokens],
                    block_tables=[self.table],
                    store=self.store,
                )
                self.cached_tokens += 1
                predictions.extend(self._greedy(logits))
        self.peak_kv_bytes = max(self.peak_kv_bytes, self.store.allocated_bytes)
        return predictions

    def truncate(self, token_count: int) -> None:
        if token_count > self.cached_tokens:
            raise ValueError("cannot extend K/V through truncation")
        self.store.truncate(self.table, token_count)
        self.cached_tokens = token_count

    def finish(self) -> None:
        self.store.clear()
        self.table = ()
        self.cached_tokens = 0


class _ModelDraft:
    def __init__(self, target: _MlxTarget) -> None:
        self.target = target
        self.history: list[int] = []
        self.next_token = 0

    def set_cancelled(self, cancelled: Callable[[], bool]) -> None:
        self.target.set_cancelled(cancelled)

    def start(self, prompt, max_new_tokens):
        self.history = list(prompt)
        self.next_token = self.target.start(prompt, max_new_tokens)

    def propose(self, history, count, cancelled):
        if count <= 0 or cancelled():
            return []
        common = 0
        while (
            common < min(len(history), len(self.history))
            and history[common] == self.history[common]
        ):
            common += 1
        if common < len(self.history):
            self.target.truncate(common)
            self.history = self.history[:common]
            # Normally a correction supplies at least one new token. Avoid
            # stale next logits if a caller supplies an exact shorter prefix.
            if common == len(history):
                raise RuntimeError(
                    "draft synchronization requires a committed correction"
                )
        for token in history[common:]:
            if cancelled():
                return []
            self.next_token = self.target.append([token], block=False)[0]
            self.history.append(token)
        proposals = []
        for _ in range(count):
            if cancelled():
                break
            token = self.next_token
            proposals.append(token)
            self.history.append(token)
            self.next_token = self.target.append([token], block=False)[0]
        return proposals

    def finish(self):
        self.target.finish()
        self.history.clear()


class SpeculativeEngine(GreedySpeculator):
    """One-request MLX speculative engine for FP16 Qwen2 or text Gemma 3.

    Scale independent instances through Forge's worker/orchestration layer.
    Different draft artifacts require a matching explicit tokenizer fingerprint
    because Forge artifacts currently contain weights/config, not tokenizers.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        draft_model_path: str | Path | None = None,
        draft: DraftProvider | None = None,
        draft_tokens: int = 4,
        max_model_length: int | None = None,
        kv_cache_bytes: int = 512 << 20,
        draft_kv_cache_bytes: int = 128 << 20,
        prefill_chunk_size: int = 512,
        custom_metal: bool = True,
        metal_paged_attention: bool = True,
        attention_tile_size: int = 1024,
        verification_mode: str = "block",
        adaptive: bool = True,
        target_tokenizer_fingerprint: str = "",
        draft_tokenizer_fingerprint: str = "",
    ) -> None:
        from .backends.factory import create_mlx_model

        if draft is not None and draft_model_path is not None:
            raise ValueError("choose a draft provider or a draft model")
        if any(
            type(v) is not int or v <= 0
            for v in (kv_cache_bytes, draft_kv_cache_bytes, prefill_chunk_size)
        ):
            raise ValueError(
                "cache budgets and prefill chunk size must be positive integers"
            )
        self.model = self.draft_model = None
        self._closed = False
        options = {
            "max_model_length": max_model_length,
            "attention_tile_size": attention_tile_size,
            "custom_metal": custom_metal,
            "metal_paged_attention": metal_paged_attention,
            "decode_mode": "rowwise",
        }
        try:
            self.model = create_mlx_model(str(model_path), **options)
            if self.model.file.quantization:
                raise ValueError(
                    "speculative decoding currently requires FP16 artifacts"
                )
            target = _MlxTarget(self.model, kv_cache_bytes, prefill_chunk_size)
            if draft_model_path is not None:
                self.draft_model = create_mlx_model(str(draft_model_path), **options)
                if self.draft_model.file.quantization:
                    raise ValueError("draft decoding currently requires FP16 artifacts")
                a, b = self.model.config, self.draft_model.config
                if (a.model_type, a.vocab_size, a.eos_token_id) != (
                    b.model_type,
                    b.vocab_size,
                    b.eos_token_id,
                ):
                    raise ValueError(
                        "draft and target must use the same token vocabulary/family"
                    )
                if (
                    self.model.file.data_sha256 != self.draft_model.file.data_sha256
                    and (
                        not target_tokenizer_fingerprint
                        or target_tokenizer_fingerprint != draft_tokenizer_fingerprint
                    )
                ):
                    raise ValueError(
                        "distinct draft and target artifacts require matching tokenizer fingerprints"
                    )
                if self.draft_model.max_model_length < self.model.max_model_length:
                    raise ValueError(
                        "draft context limit must cover the target context limit"
                    )
                draft = _ModelDraft(
                    _MlxTarget(
                        self.draft_model, draft_kv_cache_bytes, prefill_chunk_size
                    )
                )
            super().__init__(
                target,
                draft,
                draft_tokens=draft_tokens,
                verification_mode=verification_mode,
                adaptive=adaptive,
            )
        except Exception:
            if self.draft_model is not None:
                self.draft_model.close()
            if self.model is not None:
                self.model.close()
            raise
        self.last_result: SpeculativeResult | None = None

    def generate_result(self, *args, **kwargs) -> SpeculativeResult:
        if self._closed:
            raise RuntimeError("engine is closed")
        result = super().generate_result(*args, **kwargs)
        self.last_result = result
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "mlx",
            "verification_mode": self.verification_mode,
            "model_type": self.model.config.model_type,
            "kv_device_bytes": self.target.store.allocated_bytes,
            "last_generation": self.last_result.stats.to_dict()
            if self.last_result
            else None,
        }

    def close(self) -> None:
        if self._closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "cancel and finish active generation before closing the engine"
            )
        try:
            try:
                self.draft.finish()
            finally:
                self.target.finish()
                if self.draft_model is not None:
                    self.draft_model.close()
                self.model.close()
                self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
