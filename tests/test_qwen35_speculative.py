"""Portable hybrid-state speculation checks; no model or Metal initialization."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from forge_llm.qwen35_speculative import (
    Qwen35Cancelled,
    Qwen35Speculator,
    UpstreamQwen35Target,
)
from forge_llm.qwen35_state import (
    Qwen35StateGeometry,
    retained_array_bytes,
    retained_cache_bytes,
)
from forge_llm.speculative import NoDraft


def text_config():
    return {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "layer_types": [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(32)
        ],
    }


def test_actual_hybrid_geometry_and_live_transaction_reservation():
    state = Qwen35StateGeometry.from_text_config(text_config())
    assert state.kv_bytes_per_token == 32768
    assert state.recurrent_bytes == 48 << 20
    assert state.conv_bytes == 1179648
    assert state.fixed_bytes == (48 << 20) + 1179648
    assert state.speculative_extra_bytes(5) == (240 << 20) + 4325376
    assert (
        state.fixed_bytes + state.speculative_extra_bytes(5) + 32768 * 32768 < 2 << 30
    )
    converted = text_config()
    converted.pop("layer_types")
    assert Qwen35StateGeometry.from_text_config(converted) == state


@pytest.mark.parametrize(
    "field,value",
    [
        ("head_dim", 160),
        ("num_hidden_layers", 28),
        ("linear_num_value_heads", 16),
        ("linear_key_head_dim", True),
        ("mamba_ssm_dtype", "bfloat16"),
        ("layer_types", ["full_attention"] * 32),
    ],
)
def test_incompatible_geometry_rejected(field, value):
    config = text_config()
    config[field] = value
    with pytest.raises(ValueError):
        Qwen35StateGeometry.from_text_config(config)


def test_retained_history_is_charged_without_double_counting_references():
    old = np.zeros((2, 3), dtype=np.float32)
    new = np.ones_like(old)
    history = np.zeros((4, 2, 3), dtype=np.float32)
    cache = SimpleNamespace(
        cache=[new],
        _speculation={
            "initial_state": [old],
            "records": {1: ("states", history, new)},
        },
    )
    assert retained_cache_bytes([cache]) == old.nbytes + new.nbytes + history.nbytes
    cyclic = [old]
    cyclic.append(cyclic)
    assert retained_array_bytes(cyclic) == old.nbytes
    assert retained_array_bytes([old, old[:, :]]) == old.nbytes * 2


class HybridOracle:
    """History-dependent recurrent state plus causal convolution and KV history."""

    vocab_size = 17

    def __init__(self, prompt, *, predict=None):
        self.history = []
        self.state = 0
        self.conv = [0, 0, 0]
        self.calls, self.commits, self.aborts = [], [], []
        self.held = 0
        self.predict = predict or self.predict_history
        for token in prompt:
            self.consume(token)

    @property
    def cached_tokens(self):
        return len(self.history)

    def consume(self, token):
        self.history.append(token)
        self.state = (self.state * 3 + token + sum(self.conv)) % self.vocab_size
        self.conv = [*self.conv[1:], token]
        return self.predict(self.history)

    def snapshot(self):
        return list(self.history), self.state, list(self.conv)

    def restore(self, snapshot):
        history, self.state, conv = snapshot
        self.history, self.conv = list(history), list(conv)

    def predict_history(self, history):
        state, conv = 0, [0, 0, 0]
        for token in history:
            state = (state * 3 + token + sum(conv)) % self.vocab_size
            conv = [*conv[1:], token]
        return (state + 5 * sum(conv) + len(history)) % self.vocab_size

    def decode_one(self, token):
        self.calls.append((token,))
        return self.consume(token)

    def verify(self, tokens):
        self.calls.append(tuple(tokens))
        snapshots = [self.snapshot()]
        predictions = []
        for token in tokens:
            predictions.append(self.consume(token))
            snapshots.append(self.snapshot())
        self.held += 1
        oracle = self

        class Transaction:
            active = True

            def __init__(self):
                self.predictions = predictions

            def commit(self, retained):
                assert self.active
                oracle.restore(snapshots[retained])
                oracle.commits.append(retained)
                self.active = False
                oracle.held -= 1

            def abort(self):
                if self.active:
                    oracle.restore(snapshots[0])
                    oracle.aborts.append(len(tokens))
                    self.active = False
                    oracle.held -= 1

        return Transaction()


class OracleDraft:
    def __init__(self, oracle, correct_prefix=100):
        self.oracle, self.correct_prefix = oracle, correct_prefix
        self.finished = self.started = 0

    def start(self, prompt, max_new_tokens):
        self.started += 1

    def propose(self, history, count, cancelled):
        history, tokens = list(history), []
        for index in range(count):
            if cancelled():
                return tokens
            token = self.oracle.predict(history)
            if index == self.correct_prefix:
                token = (token + 1) % self.oracle.vocab_size
            tokens.append(token)
            history.append(token)
        return tokens

    def finish(self):
        self.finished += 1


def reference(oracle, prompt, count, eos=()):
    history, output = list(prompt), []
    for _ in range(count):
        token = oracle.predict(history)
        history.append(token)
        output.append(token)
        if token in eos:
            break
    return output


def generate(oracle, decoder, prompt, count, eos=()):
    decoder.start(prompt, count)
    output = [oracle.predict(prompt)]
    try:
        while len(output) < count and output[-1] not in eos:
            block = decoder.next_block(prompt + output, count - len(output), eos)
            assert 0 < len(block) <= count - len(output)
            output.extend(block)
            assert oracle.history == (prompt + output)[:-1]
            assert oracle.state == HybridOracle(oracle.history).state
            assert oracle.conv == ([0, 0, 0] + oracle.history)[-3:]
        return output
    finally:
        decoder.finish()


@pytest.mark.parametrize("count,prefix", list(product([1, 2, 8, 31], [0, 1, 2, 3, 4])))
def test_every_rejection_prefix_and_continued_decoding_preserve_all_states(
    count, prefix
):
    prompt = [2, 7, 9, 1]
    oracle = HybridOracle(prompt)
    draft = OracleDraft(oracle, prefix)
    decoder = Qwen35Speculator(oracle, draft=draft, adaptive=False)
    expected = reference(oracle, prompt, count)
    assert generate(oracle, decoder, prompt, count) == expected
    assert oracle.held == 0 and draft.finished == 1
    assert all(length <= 5 for length in map(len, oracle.calls))
    assert (
        decoder.stats.accepted_draft_tokens + decoder.stats.rejected_draft_tokens
        == decoder.stats.draft_tokens
    )


def test_full_acceptance_is_committed_once_and_emits_bonus():
    prompt = [3]
    oracle = HybridOracle(prompt)
    decoder = Qwen35Speculator(oracle, draft=OracleDraft(oracle), adaptive=False)
    assert generate(oracle, decoder, prompt, 16) == reference(oracle, prompt, 16)
    assert [len(tokens) for tokens in oracle.calls] == [5, 5, 5]
    assert oracle.commits == [5, 5, 5] and oracle.aborts == []
    assert decoder.stats.accepted_draft_tokens == 12


@pytest.mark.parametrize("prefix,eos", list(product([0, 1, 4], [1, 2, 4, 7])))
def test_eos_never_emits_or_commits_post_eos_tokens(prefix, eos):
    prompt = [9]
    oracle = HybridOracle(prompt, predict=lambda history: len(history) % 17)
    decoder = Qwen35Speculator(
        oracle, draft=OracleDraft(oracle, prefix), adaptive=False
    )
    assert generate(oracle, decoder, prompt, 12, [eos]) == reference(
        oracle, prompt, 12, [eos]
    )
    assert oracle.history[-1] != eos


@pytest.mark.parametrize("mode,draft", [("block", NoDraft()), ("sequential", None)])
def test_canonical_fallback_decodes_only_pending_without_draft_work(mode, draft):
    prompt = [5]
    oracle = HybridOracle(prompt)
    provider = draft or OracleDraft(oracle)
    decoder = Qwen35Speculator(oracle, draft=provider, verification_mode=mode)
    assert generate(oracle, decoder, prompt, 12) == reference(oracle, prompt, 12)
    assert all(len(call) == 1 for call in oracle.calls)
    assert oracle.commits == [] and decoder.stats.draft_tokens == 0


def test_cancellation_after_verification_aborts_all_state_and_remains_sticky():
    prompt = [3]
    oracle = HybridOracle(prompt)
    decoder = Qwen35Speculator(oracle, draft=OracleDraft(oracle))
    decoder.start(prompt, 6)
    before = oracle.snapshot()
    with pytest.raises(Qwen35Cancelled):
        decoder.next_block(
            prompt + [oracle.predict(prompt)], 5, cancelled=lambda: oracle.held > 0
        )
    assert oracle.snapshot() == before and oracle.held == 0 and oracle.aborts == [5]
    with pytest.raises(Qwen35Cancelled):
        decoder.next_block(prompt + [oracle.predict(prompt)], 5)
    decoder.finish()


def test_malformed_prediction_aborts_before_any_partial_output_or_commit():
    prompt = [1]
    oracle = HybridOracle(prompt)
    original = oracle.verify

    def broken(tokens):
        transaction = original(tokens)
        transaction.predictions[-1] = True
        return transaction

    oracle.verify = broken
    decoder = Qwen35Speculator(oracle, draft=OracleDraft(oracle))
    decoder.start(prompt, 6)
    before = oracle.snapshot()
    with pytest.raises(RuntimeError, match="prediction"):
        decoder.next_block(prompt + [oracle.predict(prompt)], 5)
    assert oracle.snapshot() == before and oracle.held == 0 and oracle.commits == []
    decoder.finish()


def test_infinite_draft_consumes_only_allowed_plus_one_and_never_verifies():
    prompt = [2]
    oracle = HybridOracle(prompt)
    calls = []

    class Infinite(OracleDraft):
        def propose(self, history, count, cancelled):
            while True:
                calls.append(1)
                yield 1

    decoder = Qwen35Speculator(oracle, draft=Infinite(oracle))
    decoder.start(prompt, 5)
    with pytest.raises(ValueError, match="bounded token budget"):
        decoder.next_block(prompt + [oracle.predict(prompt)], 4)
    assert len(calls) == 4 and oracle.calls == [] and oracle.held == 0
    decoder.finish()


@pytest.mark.parametrize("value", [0, 9, True, 1.5])
def test_draft_depth_is_bounded(value):
    with pytest.raises(ValueError):
        Qwen35Speculator(HybridOracle([1]), draft_tokens=value)


def test_request_output_budget_cannot_be_increased_by_a_step_argument():
    oracle = HybridOracle([2])
    decoder = Qwen35Speculator(oracle, draft=OracleDraft(oracle))
    decoder.start([2], 5)
    with pytest.raises(ValueError, match="remaining budget"):
        decoder.next_block([2, oracle.predict([2])], 20)
    assert oracle.calls == []
    decoder.finish()


def test_failed_draft_start_is_finalized_and_next_request_can_start():
    oracle = HybridOracle([2])

    class FailedStart(OracleDraft):
        def start(self, prompt, max_new_tokens):
            super().start(prompt, max_new_tokens)
            if self.started == 1:
                raise ValueError("invalid draft state")

    draft = FailedStart(oracle)
    decoder = Qwen35Speculator(oracle, draft=draft)
    with pytest.raises(ValueError, match="draft state"):
        decoder.start([2], 5)
    assert draft.finished == 1
    decoder.start([2], 5)
    decoder.finish()
    assert draft.finished == 2


@dataclass
class FakeMX:
    fail_eval: bool = False

    def eval(self, *arrays):
        if self.fail_eval:
            raise RuntimeError("deferred GPU failure")

    all = staticmethod(np.all)
    isfinite = staticmethod(np.isfinite)
    argmax = staticmethod(np.argmax)
    array = staticmethod(lambda value: np.array(value, copy=True))


def upstream_bridge(
    *, fail_forward=False, fail_eval=False, fail_observe=False, deny=False
):
    oracle = HybridOracle([3])
    retained, releases, modes = [], [], []
    mx = FakeMX(fail_eval)

    def forward(tokens, *, speculative_verify):
        modes.append(speculative_verify)
        if fail_forward:
            raise RuntimeError("upstream forward failure")
        if speculative_verify:
            transaction = oracle.verify(tokens)
            predictions = transaction.predictions
        else:
            transaction = None
            predictions = [oracle.decode_one(tokens[0])]
        logits = np.zeros((1, len(tokens), oracle.vocab_size), np.float32)
        for index, token in enumerate(predictions):
            logits[0, index, token] = 1
        return SimpleNamespace(logits=logits, gdn_states=transaction)

    def observe():
        if fail_observe:
            raise RuntimeError("cache exceeds capacity")

    def reserve(amount):
        if deny:
            raise ValueError("history capacity unavailable")
        retained.append(amount)
        return lambda: releases.append(amount)

    target = UpstreamQwen35Target(
        mx,
        caches=[],
        forward=forward,
        observe_cache=observe,
        reserve_history=reserve,
        cached_tokens=lambda: oracle.cached_tokens,
        geometry=Qwen35StateGeometry(),
        vocab_size=oracle.vocab_size,
    )
    return target, oracle, retained, releases, modes


def test_bridge_uses_actual_upstream_flag_reserves_before_forward_and_releases():
    target, oracle, retained, releases, modes = upstream_bridge()
    transaction = target.verify([2, 4, 5])
    assert oracle.cached_tokens == 4 and retained and not releases and modes == [True]
    transaction.commit(2)
    assert oracle.cached_tokens == 3 and releases == retained and target._active is None
    with pytest.raises(RuntimeError, match="closed"):
        transaction.commit(1)
    transaction.abort()
    assert releases == retained
    target.decode_one(9)
    assert modes == [True, False]


@pytest.mark.parametrize(
    "options",
    [
        {"fail_forward": True},
        {"fail_eval": True},
        {"fail_observe": True},
        {"deny": True},
    ],
)
def test_bridge_error_paths_abort_history_and_release_reservation(options):
    target, oracle, retained, releases, modes = upstream_bridge(**options)
    before = oracle.snapshot()
    with pytest.raises((RuntimeError, ValueError)):
        target.verify([2, 4])
    assert oracle.snapshot() == before and oracle.held == 0
    assert releases == retained and target._active is None
    if options.get("deny"):
        assert modes == []


def test_engine_cleanup_can_abort_unconsumed_verification_once():
    target, oracle, retained, releases, _ = upstream_bridge()
    before = oracle.snapshot()
    target.verify([2, 4])
    target.abort()
    target.abort()
    assert oracle.snapshot() == before and oracle.held == 0
    assert releases == retained


def test_nonfinite_logits_abort_before_commit_and_release_history():
    target, oracle, retained, releases, _ = upstream_bridge()
    forward = target.forward

    def nonfinite(*args, **kwargs):
        output = forward(*args, **kwargs)
        output.logits[0, -1, 0] = np.nan
        return output

    target.forward = nonfinite
    before = oracle.snapshot()
    with pytest.raises(RuntimeError, match="nonfinite"):
        target.verify([2, 4])
    assert oracle.snapshot() == before and oracle.held == 0
    assert releases == retained


def test_upstream_commit_failure_aborts_and_releases_once():
    target, oracle, retained, releases, _ = upstream_bridge()
    forward = target.forward

    def broken_commit(*args, **kwargs):
        output = forward(*args, **kwargs)

        def commit(_length):
            raise RuntimeError("invalid temporal history")

        output.gdn_states.commit = commit
        return output

    target.forward = broken_commit
    before = oracle.snapshot()
    transaction = target.verify([2, 4])
    with pytest.raises(RuntimeError, match="temporal history"):
        transaction.commit(1)
    transaction.abort()
    assert oracle.snapshot() == before and oracle.held == 0
    assert releases == retained and target._active is None


def test_partial_commit_detaches_selected_state_before_releasing_history():
    target, oracle, retained, releases, _ = upstream_bridge()
    states = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
    convolution = np.arange(9 * 2, dtype=np.float16).reshape(9, 2)
    cache = SimpleNamespace(cache=[convolution[-3:], states[2]])
    target.caches = [cache]
    transaction = target.verify([2, 4, 5, 6, 7])
    expected_state, expected_conv = cache.cache[1].copy(), cache.cache[0].copy()
    transaction.commit(3)
    assert np.array_equal(cache.cache[1], expected_state)
    assert np.array_equal(cache.cache[0], expected_conv)
    assert not np.shares_memory(cache.cache[1], states)
    assert not np.shares_memory(cache.cache[0], convolution)
    assert releases == retained and oracle.held == 0
