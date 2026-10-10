"""Bounded greedy speculation over MLX-VLM 0.7.6 hybrid transactions.

The engine owns preparation, media positions, requests and ordinary cache
admission. This module owns one speculative round. It calls the upstream
Qwen3_5ExactSpeculativeVerifier through ``speculative_verify=True`` and commits
or aborts every recurrent, convolution and attention cache together. The block
numerical contract is opt-in and requires measured canonical qualification.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any, Protocol

from .qwen35_state import Qwen35StateGeometry
from .speculative import DraftProvider, NGramDraft, _token_ids


class Qwen35Cancelled(RuntimeError):
    """Cancellation observed before a speculative round was committed."""


class Verification(Protocol):
    predictions: list[int]

    def commit(self, retained_inputs: int) -> None: ...
    def abort(self) -> None: ...


class Qwen35Target(Protocol):
    vocab_size: int

    @property
    def cached_tokens(self) -> int: ...

    def decode_one(self, token: int) -> int: ...
    def verify(self, tokens: Sequence[int]) -> Verification: ...


@dataclass
class Qwen35SpeculativeStats:
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    verification_blocks: int = 0
    target_decode_calls: int = 0
    rollback_tokens: int = 0
    draft_seconds: float = 0.0
    verification_seconds: float = 0.0
    commit_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["acceptance_rate"] = (
            self.accepted_draft_tokens / self.draft_tokens if self.draft_tokens else 0.0
        )
        return result


class _ManagedVerification:
    def __init__(self, target, transaction, predictions, release, before, length):
        self.target, self.transaction = target, transaction
        self.predictions, self.release = predictions, release
        self.before, self.length, self.active = before, length, True

    def _release(self) -> None:
        self.active = False
        self.target._active = None
        self.transaction = None
        self.release()

    def commit(self, retained_inputs: int) -> None:
        if not self.active:
            raise RuntimeError("verification transaction is already closed")
        if type(retained_inputs) is not int or not 0 <= retained_inputs <= self.length:
            raise ValueError("invalid verification prefix length")
        try:
            self.transaction.commit(retained_inputs)
            if self.target.cached_tokens != self.before + retained_inputs:
                raise RuntimeError("upstream transaction retained an incorrect prefix")
            self.target._detach_committed_state(retained_inputs, self.length)
            self.target.observe_cache()
        except BaseException:
            # A validated upstream commit is synchronous state selection. If a
            # failure occurs before its completion, restore the original state.
            # Once closed, the engine must discard this request on an error.
            if getattr(self.transaction, "active", False):
                self.transaction.abort()
            raise
        finally:
            self._release()

    def abort(self) -> None:
        if not self.active:
            return
        try:
            self.transaction.abort()
            if self.target.cached_tokens != self.before:
                raise RuntimeError("upstream transaction failed to restore the cache")
        finally:
            self._release()


class UpstreamQwen35Target:
    """Bridge engine-owned forwards to the actual released hybrid verifier.

    ``forward(tokens, speculative_verify=...)`` must call the upstream language
    model with this request's prepared position metadata. ``reserve_history``
    must reserve extra retained bytes before forwarding and return an idempotent
    release callback. All hooks execute on the engine's single owner thread.
    """

    def __init__(
        self,
        mx: Any,
        *,
        caches: list[Any],
        forward: Callable,
        observe_cache: Callable[[], None],
        reserve_history: Callable[[int], Callable[[], None]],
        cached_tokens: Callable[[], int],
        geometry: Qwen35StateGeometry,
        vocab_size: int,
    ) -> None:
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("invalid target vocabulary")
        self.mx, self.caches, self.forward = mx, caches, forward
        self.observe_cache, self.reserve_history = observe_cache, reserve_history
        self._cached_tokens, self.geometry, self.vocab_size = (
            cached_tokens,
            geometry,
            vocab_size,
        )
        self._active: _ManagedVerification | None = None

    @property
    def cached_tokens(self) -> int:
        value = self._cached_tokens()
        if type(value) is not int or value < 0:
            raise RuntimeError("invalid hybrid cache position")
        return value

    def _predictions(self, output: Any, length: int) -> list[int]:
        logits = output.logits
        if logits is None or tuple(logits.shape) != (1, length, self.vocab_size):
            raise RuntimeError("invalid Qwen3.5 verification logits shape")
        # Force deferred GPU errors and all retained recurrent history before a
        # transaction is committed. Argmax alone might omit unused state outputs.
        retained = []
        for cache in self.caches:
            retained.extend(
                [getattr(cache, "keys", None), getattr(cache, "values", None)]
            )
            retained.extend(getattr(cache, "cache", []) or [])
            transaction = getattr(cache, "_speculation", None)
            if transaction is not None:
                retained.extend(transaction["initial_state"])
                for record in transaction["records"].values():
                    retained.extend(record[1:3])
        self.mx.eval(
            logits,
            *[
                value
                for value in retained
                if value is not None and not isinstance(value, int)
            ],
        )
        finite = self.mx.all(self.mx.isfinite(logits)).item()
        if not finite:
            raise RuntimeError("Qwen3.5 target returned nonfinite logits")
        predictions = self.mx.argmax(logits, axis=-1).reshape(-1).tolist()
        if len(predictions) != length or any(
            type(token) is not int or not 0 <= token < self.vocab_size
            for token in predictions
        ):
            raise RuntimeError("invalid Qwen3.5 target predictions")
        return predictions

    def decode_one(self, token: int) -> int:
        if self._active is not None:
            raise RuntimeError("cannot decode during an active verification")
        before = self.cached_tokens
        output = self.forward([token], speculative_verify=False)
        predictions = self._predictions(output, 1)
        self.observe_cache()
        if self.cached_tokens != before + 1:
            raise RuntimeError("canonical decode failed to advance hybrid cache")
        return predictions[0]

    def _detach_committed_state(self, retained_inputs: int, block_length: int) -> None:
        # Upstream selects a view into its intermediate-state stack on partial
        # acceptance. That view can keep the entire stack alive after the
        # transaction lease is released. A contiguous convolution slice can
        # similarly retain its larger source buffer. mx.array(array) explicitly
        # copies in MLX 0.32.2; copy.deepcopy and mx.contiguous do not guarantee it.
        copies = []
        partial = 0 < retained_inputs < block_length
        for cache in self.caches:
            arrays = getattr(cache, "cache", None)
            if arrays is None:
                continue
            for index in (0, 1) if partial else (0,):
                value = arrays[index]
                if value is not None:
                    copies.append((arrays, index, self.mx.array(value)))
        self.mx.eval(*[value for _, _, value in copies])
        for arrays, index, value in copies:
            arrays[index] = value

    def verify(self, tokens: Sequence[int]) -> Verification:
        if self._active is not None:
            raise RuntimeError("a verification transaction is already active")
        tokens = _token_ids(tokens, self.vocab_size, "verification")
        if not 2 <= len(tokens) <= 9:
            raise ValueError("speculative verification needs 2 to 9 input positions")
        before = self.cached_tokens
        release = self.reserve_history(
            self.geometry.speculative_extra_bytes(len(tokens))
        )
        transaction = None
        try:
            # Released LanguageModel.__call__ selects the exact verifier and
            # starts its own bounded transaction before mutating any layer.
            output = self.forward(tokens, speculative_verify=True)
            transaction = getattr(output, "gdn_states", None)
            if transaction is None or not getattr(transaction, "active", False):
                raise RuntimeError(
                    "upstream verifier did not return an active transaction"
                )
            predictions = self._predictions(output, len(tokens))
            self.observe_cache()
            if self.cached_tokens != before + len(tokens):
                raise RuntimeError("verification failed to advance the complete block")
            result = _ManagedVerification(
                self, transaction, predictions, release, before, len(tokens)
            )
            self._active = result
            return result
        except BaseException:
            try:
                if transaction is not None:
                    transaction.abort()
            finally:
                release()
            raise

    def abort(self) -> None:
        if self._active is not None:
            self._active.abort()


class Qwen35Speculator:
    """Produce one bounded output block over an engine-owned request.

    Cache invariant: ``history[:-1]`` is committed, and ``history[-1]`` is
    pending. Full acceptance commits pending plus every proposal and emits a
    bonus. Rejection commits pending plus the matching prefix and emits the
    correction. Sequential mode calls ordinary decode exactly once per step.
    """

    def __init__(
        self,
        target: Qwen35Target,
        *,
        draft_tokens: int = 4,
        verification_mode: str = "block",
        draft: DraftProvider | None = None,
        adaptive: bool = True,
    ) -> None:
        if type(draft_tokens) is not int or not 1 <= draft_tokens <= 8:
            raise ValueError("draft_tokens must be between 1 and 8")
        if verification_mode not in {"block", "sequential"}:
            raise ValueError("verification_mode must be block or sequential")
        if type(adaptive) is not bool:
            raise ValueError("adaptive must be boolean")
        self.target, self.draft_tokens = target, draft_tokens
        self.verification_mode, self.adaptive = verification_mode, adaptive
        self.draft = draft if draft is not None else NGramDraft()
        self.stats = Qwen35SpeculativeStats()
        self._budget = draft_tokens
        self._prompt_length = self._max_tokens = 0
        self._started = self._cancel_requested = False
        self._lock = threading.Lock()

    def start(self, prompt: Sequence[int], max_new_tokens: int) -> None:
        if self._started:
            raise RuntimeError("speculative request is already started")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        prompt = _token_ids(prompt, self.target.vocab_size, "prompt")
        if not prompt:
            raise ValueError("prompt must not be empty")
        self.stats = Qwen35SpeculativeStats()
        self._budget, self._cancel_requested = self.draft_tokens, False
        self._prompt_length, self._max_tokens = len(prompt), max_new_tokens
        # Mark started before calling arbitrary user code so a failed start can
        # still be finalized by the engine's ordinary request cleanup.
        self._started = True
        try:
            self.draft.start(prompt, max_new_tokens)
        except BaseException:
            self.finish()
            raise

    def next_block(
        self,
        history: Sequence[int],
        remaining: int,
        eos_token_ids: Sequence[int] = (),
        *,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> list[int]:
        if not self._started:
            raise RuntimeError("start the speculative request before decoding")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("speculative rounds cannot run concurrently")
        transaction = None

        def check() -> bool:
            self._cancel_requested = self._cancel_requested or bool(cancelled())
            if self._cancel_requested:
                raise Qwen35Cancelled("Qwen3.5 speculative generation cancelled")
            return False

        try:
            history = _token_ids(history, self.target.vocab_size, "history")
            eos = set(_token_ids(eos_token_ids, self.target.vocab_size, "EOS"))
            if not history or type(remaining) is not int or remaining <= 0:
                raise ValueError("invalid pending history or remaining output budget")
            produced = len(history) - self._prompt_length
            if (
                not 1 <= produced < self._max_tokens
                or remaining != self._max_tokens - produced
            ):
                raise ValueError(
                    "remaining budget does not match this request's history"
                )
            if history[-1] in eos:
                raise ValueError("cannot decode after EOS")
            if self.target.cached_tokens != len(history) - 1:
                raise RuntimeError("hybrid cache does not match the committed history")
            check()
            allowed = min(self._budget, remaining - 1)
            proposals = []
            if allowed and self.verification_mode == "block":
                begin = time.perf_counter()
                proposals = _token_ids(
                    islice(self.draft.propose(history, allowed, check), allowed + 1),
                    self.target.vocab_size,
                    "draft",
                )
                self.stats.draft_seconds += time.perf_counter() - begin
                if len(proposals) > allowed:
                    raise ValueError("draft provider exceeded its bounded token budget")
                for index, token in enumerate(proposals):
                    if token in eos:
                        proposals = proposals[: index + 1]
                        break
            check()
            if not proposals:
                token = self.target.decode_one(history[-1])
                self.stats.target_decode_calls += 1
                check()
                if type(token) is not int or not 0 <= token < self.target.vocab_size:
                    raise RuntimeError("invalid canonical target token")
                return [token]
            begin = time.perf_counter()
            transaction = self.target.verify([history[-1], *proposals])
            self.stats.verification_seconds += time.perf_counter() - begin
            self.stats.verification_blocks += 1
            self.stats.target_decode_calls += 1
            self.stats.draft_tokens += len(proposals)
            check()
            predictions = transaction.predictions
            if len(predictions) != len(proposals) + 1 or any(
                type(token) is not int or not 0 <= token < self.target.vocab_size
                for token in predictions
            ):
                raise RuntimeError("invalid verification prediction block")
            accepted = 0
            while (
                accepted < len(proposals)
                and proposals[accepted] == predictions[accepted]
            ):
                accepted += 1
            additions = proposals[:accepted] + [predictions[accepted]]
            for index, token in enumerate(additions):
                if token in eos:
                    additions = additions[: index + 1]
                    break
            # If an accepted proposal is EOS, retain only through the preceding
            # output token; EOS remains pending and will never be decoded.
            retained = min(1 + accepted, len(additions))
            check()
            begin = time.perf_counter()
            transaction.commit(retained)
            self.stats.commit_seconds += time.perf_counter() - begin
            transaction = None
            rollback = len(proposals) - accepted
            self.stats.accepted_draft_tokens += accepted
            self.stats.rejected_draft_tokens += rollback
            self.stats.rollback_tokens += len(proposals) + 1 - retained
            if self.adaptive:
                if accepted == len(proposals):
                    self._budget = min(self.draft_tokens, self._budget + 1)
                elif accepted * 2 < len(proposals):
                    self._budget = max(1, self._budget // 2)
            return additions
        finally:
            try:
                if transaction is not None:
                    transaction.abort()
            finally:
                self._lock.release()

    def finish(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            abort = getattr(self.target, "abort", None)
            if abort is not None:
                abort()
        finally:
            self.draft.finish()
