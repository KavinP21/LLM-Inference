"""Portable causal-oracle tests: exact acceptance, rollback and failure paths."""

from __future__ import annotations

import threading
from itertools import product

import pytest
from forge_llm.speculative import (
    GenerationCancelled,
    GreedySpeculator,
    NGramDraft,
    NoDraft,
)


class Oracle:
    vocab_size = 11
    max_model_length = 128

    def __init__(self, predict=None):
        self.predict = predict or (lambda history: (sum(history) + len(history)) % 11)
        self.cache = []
        self.calls = []
        self.truncations = []
        self.finished = 0

    @property
    def cached_tokens(self):
        return len(self.cache)

    def start(self, prompt, max_new_tokens):
        self.cache = list(prompt)
        return self.predict(self.cache)

    def append(self, tokens, *, block):
        self.calls.append((list(tokens), block))
        predictions = []
        for token in tokens:
            self.cache.append(token)
            predictions.append(self.predict(self.cache))
        return predictions

    def truncate(self, token_count):
        self.truncations.append((len(self.cache), token_count))
        del self.cache[token_count:]

    def finish(self):
        self.cache.clear()
        self.finished += 1


class Draft:
    def __init__(self, target, correct_prefix=100):
        self.target = target
        self.correct_prefix = correct_prefix
        self.finished = 0

    def start(self, prompt, max_new_tokens):
        pass

    def propose(self, history, count, cancelled):
        history = list(history)
        result = []
        for index in range(count):
            token = self.target.predict(history)
            if index == self.correct_prefix:
                token = (token + 1) % self.target.vocab_size
            result.append(token)
            history.append(token)
        return result

    def finish(self):
        self.finished += 1


def reference(prompt, count, target, eos=()):
    history, output = list(prompt), []
    for _ in range(count):
        token = target.predict(history)
        history.append(token)
        output.append(token)
        if token in eos:
            break
    return output


@pytest.mark.parametrize(
    "budget,draft_length,correct_prefix",
    list(product([1, 2, 8, 17], [1, 3, 5], [0, 1, 2, 100])),
)
def test_all_acceptance_positions_preserve_greedy_output(
    budget, draft_length, correct_prefix
):
    target = Oracle()
    draft = Draft(target, correct_prefix)
    engine = GreedySpeculator(target, draft, draft_tokens=draft_length, adaptive=False)
    result = engine.generate_result([2, 7, 5], budget)
    assert result.tokens == reference([2, 7, 5], budget, target)
    assert result.finish_reason == "length"
    assert target.cache == []
    assert draft.finished == target.finished == 1
    assert all(before > after for before, after in target.truncations)
    assert result.stats.target_decode_calls == len(target.calls)
    assert (
        result.stats.accepted_draft_tokens + result.stats.rejected_draft_tokens
        == result.stats.draft_tokens
    )
    assert all(block == (len(tokens) > 1) for tokens, block in target.calls)


def test_perfect_draft_uses_one_target_call_per_block_and_bonus():
    target = Oracle()
    result = GreedySpeculator(target, Draft(target), draft_tokens=4).generate_result(
        [1], 16
    )
    assert result.tokens == reference([1], 16, target)
    assert len(target.calls) == 3
    assert [len(inputs) for inputs, _ in target.calls] == [5, 5, 5]
    assert result.stats.accepted_draft_tokens == 12
    assert result.stats.rollback_tokens == 0


@pytest.mark.parametrize("correct_prefix", [0, 1, 100])
@pytest.mark.parametrize("eos_at", [0, 1, 3, 7])
def test_eos_never_emits_rejected_or_post_eos_tokens(correct_prefix, eos_at):
    # Generate 0,1,2,... so the first matching EOS is unambiguous.
    target = Oracle(lambda history: (len(history) - 1) % 11)
    expected = reference([7], 10, target, [eos_at])
    result = GreedySpeculator(
        target, Draft(target, correct_prefix), draft_tokens=4
    ).generate_result([7], 10, [eos_at])
    assert result.tokens == expected
    assert result.finish_reason == "eos"
    assert target.cache == []


def test_no_draft_is_canonical_decode_and_sequential_fallback_is_honest():
    target = Oracle()
    baseline = GreedySpeculator(target, NoDraft()).generate_result([3], 8)
    assert all(not block and len(inputs) == 1 for inputs, block in target.calls)
    assert baseline.stats.verification_blocks == 0
    target = Oracle()
    result = GreedySpeculator(
        target, Draft(target), verification_mode="sequential"
    ).generate_result([3], 8)
    assert result.tokens == baseline.tokens
    assert result.stats.target_decode_calls == sum(
        len(inputs) for inputs, _ in target.calls
    )
    assert result.stats.verification_blocks == 0


def test_ngram_prefers_longest_latest_suffix_and_respects_budget():
    draft = NGramDraft(min_match=2, max_match=3)
    assert draft.propose([1, 2, 3, 4, 5, 1, 2, 3], 2) == [4, 5]
    assert draft.propose([1, 2, 7, 1, 2, 8, 1, 2], 3) == [8, 1, 2]
    assert draft.propose([1, 2, 3, 4], 3) == []
    assert draft.propose([1, 2, 1, 2], 0) == []
    assert draft.propose([1, 2, 1, 2], 2, lambda: True) == []


@pytest.mark.parametrize("stage", ["before", "draft", "verification"])
def test_cancelled_requests_cleanup_and_expose_partial_output(stage):
    target = Oracle()
    event = threading.Event()
    draft = Draft(target)
    if stage == "before":
        event.set()
    elif stage == "draft":
        original = draft.propose

        def cancel(*args):
            result = original(*args)
            event.set()
            return result

        draft.propose = cancel
    else:
        original = target.append

        def cancel(*args, **kwargs):
            result = original(*args, **kwargs)
            event.set()
            return result

        target.append = cancel
    engine = GreedySpeculator(target, draft)
    result = engine.generate_result([1], 8, cancelled=event.is_set)
    assert result.finish_reason == "cancelled"
    assert result.tokens == ([] if stage == "before" else reference([1], 1, target))
    assert target.cache == []
    with pytest.raises(GenerationCancelled) as error:
        engine.generate([1], 8, cancelled=event.is_set)
    assert error.value.result.finish_reason == "cancelled"


@pytest.mark.parametrize("failure", ["start", "draft", "verify", "truncate"])
def test_failures_release_caches_and_allow_reuse(failure):
    target = Oracle()
    draft = Draft(target, correct_prefix=0)
    engine = GreedySpeculator(target, draft)
    obj, method = {
        "start": (target, "start"),
        "draft": (draft, "propose"),
        "verify": (target, "append"),
        "truncate": (target, "truncate"),
    }[failure]
    original = getattr(obj, method)

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    setattr(obj, method, fail)
    with pytest.raises(RuntimeError, match="injected"):
        engine.generate([1], 8)
    assert target.cache == []
    setattr(obj, method, original)
    assert engine.generate([1], 8) == reference([1], 8, target)


@pytest.mark.parametrize(
    "prompt,budget,eos",
    [
        ([], 1, []),
        ([11], 1, []),
        ([1.5], 1, []),
        ([True], 1, []),
        ([1], 0, []),
        ([1], 1.5, []),
        ([1], 128, []),
        ([1], 1, [-1]),
    ],
)
def test_reject_invalid_requests_before_device_work(prompt, budget, eos):
    target = Oracle()
    with pytest.raises(ValueError):
        GreedySpeculator(target).generate(prompt, budget, eos)
    assert target.finished == 0


@pytest.mark.parametrize("bad", [[-1], [11], [1.2], [True], [1] * 10])
def test_bad_draft_provider_fails_closed(bad):
    target = Oracle()
    draft = Draft(target)
    draft.propose = lambda *args: bad
    with pytest.raises(ValueError):
        GreedySpeculator(target, draft).generate([1], 8)
    assert target.cache == []


def test_reentrant_generation_and_close_are_rejected():
    target = Oracle()
    engine = GreedySpeculator(target, NoDraft())
    original = target.append

    def nested(*args, **kwargs):
        with pytest.raises(RuntimeError, match="one request"):
            engine.generate([1], 1)
        return original(*args, **kwargs)

    target.append = nested
    assert engine.generate([1], 8) == reference([1], 8, target)


def test_truncate_clears_values_masks_pages_and_preserves_prefix():
    import numpy as np
    from forge_llm.paged_kv import MlxPagedKVStore, _DeviceBlock

    store = MlxPagedKVStore(np, num_layers=2, num_kv_heads=1, head_dim=2)
    pages = [np.arange(64, dtype=np.float16).reshape(2, 16, 1, 2) for _ in range(2)]
    store._blocks[0] = _DeviceBlock([page.copy() for page in pages], [65535, 65535])
    store._blocks[1] = _DeviceBlock([page.copy() for page in pages], [65535, 65535])
    store.truncate((0, 1), 7)
    assert store.allocated_blocks == 1
    assert store._blocks[0].written_masks == [127, 127]
    for actual, original in zip(store._blocks[0].layers, pages):
        np.testing.assert_array_equal(actual[:, :7], original[:, :7])
        assert not actual[:, 7:].any()
    store.truncate((0, 1), 0)
    assert store.allocated_blocks == 0


def test_truncate_rejects_shared_pages_before_mutating_anything():
    import numpy as np
    from forge_llm.paged_kv import MlxPagedKVStore, _DeviceBlock

    store = MlxPagedKVStore(
        np, num_layers=1, num_kv_heads=1, head_dim=2, is_shared=lambda page: page == 1
    )
    store._blocks[0] = _DeviceBlock([np.ones((2, 16, 1, 2))], [65535])
    store._blocks[1] = _DeviceBlock([np.ones((2, 16, 1, 2))], [65535])
    with pytest.raises(RuntimeError, match="copy-on-write"):
        store.truncate((0, 1), 7)
    assert store._blocks[0].written_masks == [65535]
    assert store._blocks[0].layers[0].all()


def test_draft_cleanup_failure_still_releases_target_and_request_lock():
    target = Oracle()
    draft = Draft(target)

    def fail():
        raise RuntimeError("injected draft cleanup failure")

    draft.finish = fail
    engine = GreedySpeculator(target, draft)
    with pytest.raises(RuntimeError, match="cleanup"):
        engine.generate([1], 8)
    assert target.cache == []
    engine.draft = NoDraft()
    assert engine.generate([1], 8) == reference([1], 8, target)


def test_unbounded_draft_iterator_is_consumed_only_to_one_excess_token():
    target = Oracle()
    draft = Draft(target)
    consumed = []

    def infinite():
        while True:
            consumed.append(1)
            yield 1

    draft.propose = lambda *args: infinite()
    with pytest.raises(ValueError, match="budget"):
        GreedySpeculator(target, draft, draft_tokens=4).generate([1], 8)
    assert len(consumed) == 5
    assert target.cache == []
    assert draft.finished == target.finished == 1


def test_cancellation_is_sticky_when_callback_returns_true_only_once():
    target = Oracle()
    calls = []

    def once():
        calls.append(1)
        return len(calls) == 2

    target.set_cancelled = lambda callback: setattr(target, "cancelled", callback)
    original = target.start

    def start(*args):
        original(*args)
        assert target.cancelled()
        return 0  # Mirrors a prefill that noticed cancellation mid-chunk.

    target.start = start
    result = GreedySpeculator(target, NoDraft()).generate_result([1], 8, cancelled=once)
    assert result.finish_reason == "cancelled"
    assert result.tokens == []
    assert len(calls) == 2  # The raw callback is never re-polled after True.
    assert target.cache == []
