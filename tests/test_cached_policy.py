from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from forge_llm.cached_decode import replay_cached
from forge_llm.cached_policy import (
    DEFAULT_SEARCH,
    score,
    select_and_repair,
    validate_search,
)
from forge_llm.calibrate_cached import cached_metrics, capture_cached_inputs
from forge_llm.precision_policy import validate_corpora
from forge_llm.runtime import KVBlockPool


def summary(changes, loss=0.1):
    return {
        "top1_changes": changes,
        "observations": 32,
        "minimum_logit_cosine": 0.9999,
        "mean_squared_relative_error": loss,
    }


def inventory():
    return [
        {"weight": n, "fp16_bytes": 100, "relative_block_objective": i + 1}
        for i, n in enumerate("abcd")
    ]


def test_joint_repair_requires_full_calibration_improvement():
    calls = []

    def evaluate(names, full):
        calls.append((names, full))
        # a wins the cheap probe, but c repairs the FULL corpus. b overfits probes.
        return summary(
            {"a": 2, "b": 3, "c": 0, "d": 4}[next(iter(names))]
            if full
            else {"a": 0, "b": 1, "c": 2, "d": 3}[next(iter(names))]
        )

    result = select_and_repair(
        inventory(), evaluate, {**DEFAULT_SEARCH, "candidate_pool": 4}
    )
    assert result["seed_weights"] == ["a"]
    assert result["quantized_weights"] == ["c"]
    assert result["repairs"][0]["accepted"]
    assert len(calls) == len(set(calls)) <= result["evaluation_bound"]
    assert result["quantized_projection_fraction"] == 0.25
    assert select_and_repair(
        list(reversed(inventory())), evaluate, {**DEFAULT_SEARCH, "candidate_pool": 4}
    )["quantized_weights"] == ["c"]


def test_probe_only_win_or_equal_score_cannot_authorize_swap():
    def evaluate(names, full):
        return summary(1 if full else (0 if "a" in names else 1))

    result = select_and_repair(inventory(), evaluate)
    assert result["quantized_weights"] == ["a"]
    assert len(result["repairs"]) == 1 and not result["repairs"][0]["accepted"]


def test_floor_is_kept_for_every_full_repair_trial():
    local = inventory()
    local[0]["fp16_bytes"] = 200
    local[1]["fp16_bytes"] = 10
    visited = []

    def evaluate(names, full):
        visited.append((names, full))
        return summary(1, 0.1 if "a" in names else 0.2)

    result = select_and_repair(local, evaluate)
    sizes = {o["weight"]: o["fp16_bytes"] for o in local}
    assert all(
        sum(sizes[n] for n in names) * 4 >= 410 for names, full in visited if full
    )
    assert result["quantized_projection_fraction"] >= 0.25


@pytest.mark.parametrize(
    "change",
    [{"probe_cases": True}, {"repair_rounds": 0}, {"candidate_pool": 33}, {"extra": 1}],
)
def test_search_bounds_fail_closed(change):
    with pytest.raises(ValueError):
        validate_search({**DEFAULT_SEARCH, **change})


def test_search_observations_and_inventory_fail_closed():
    for s in [
        summary(-1),
        summary(33),
        summary(0, float("nan")),
        {**summary(0), "minimum_logit_cosine": float("inf")},
        {**summary(0), "observations": 0},
    ]:
        with pytest.raises(ValueError):
            score(s)
    with pytest.raises(ValueError):
        select_and_repair(inventory() * 2, lambda *args: summary(0))
    with pytest.raises(RuntimeError, match="execution failure"):
        select_and_repair(
            inventory(),
            lambda *args: (_ for _ in ()).throw(RuntimeError("execution failure")),
        )


def test_vectorized_cached_summary_matches_independent_metrics():
    from forge_llm.validate_int8 import compare_logits

    a = np.random.default_rng(31).normal(size=(7, 17)).astype(np.float16)
    b = a.copy()
    b[2, 0] += 10
    metrics = [compare_logits(x, y) for x, y in zip(a, b)]
    result = cached_metrics(a, b)
    assert result["top1_changes"] == sum(not m["argmax_equal"] for m in metrics)
    assert result["minimum_logit_cosine"] == pytest.approx(
        min(m["cosine"] for m in metrics)
    )
    assert result["mean_squared_relative_error"] == pytest.approx(
        np.mean([m["relative_l2_error"] ** 2 for m in metrics])
    )
    assert (
        cached_metrics(np.zeros((2, 4)), np.zeros((2, 4)))["minimum_logit_cosine"] == 1
    )
    for invalid in [b[:1], np.full_like(b, np.nan)]:
        with pytest.raises(ValueError):
            cached_metrics(a, invalid)


class FakeStore:
    instances = []

    def __init__(self, *args, **kwargs):
        self.ids = set()
        self.peak_allocated_blocks = 0
        self.instances.append(self)

    def touch(self, table):
        self.ids.update(table)
        self.peak_allocated_blocks = max(self.peak_allocated_blocks, len(self.ids))

    @property
    def allocated_blocks(self):
        return len(self.ids)

    @property
    def allocated_bytes(self):
        return len(self.ids) * 16

    def clear(self):
        self.ids.clear()


def fake_engine(monkeypatch):
    import forge_llm.cached_decode as cached

    FakeStore.instances.clear()
    monkeypatch.setattr(cached, "MlxPagedKVStore", FakeStore)
    calls = []

    def prefill(ids, *, start_position, block_table, store):
        calls.append(("prefill", list(ids), start_position, block_table))
        store.touch(block_table)
        return np.eye(8, dtype=np.float16)[[(i + 1) % 8 for i in ids]]

    def decode(ids, *, positions, block_tables, store):
        calls.append(("decode", list(ids), positions[0], block_tables[0]))
        store.touch(block_tables[0])
        return np.eye(8, dtype=np.float16)[[(i + 1) % 8 for i in ids]]

    engine = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=8, num_hidden_layers=1, num_key_value_heads=1
            ),
            head_dim=1,
            mx=SimpleNamespace(eval=lambda *a: None, argmax=np.argmax),
            forward_paged_chunk=prefill,
            decode_paged_batch=decode,
        ),
        cache_pool=KVBlockPool(128, 1, 4),
        max_model_length=128,
        bytes_per_block=4,
        bytes_per_token=1,
        block_tokens=4,
        prefill_chunk_size=3,
    )
    return engine, calls


def test_cached_replay_uses_chunks_actual_positions_and_lazy_pages(monkeypatch):
    engine, calls = fake_engine(monkeypatch)
    result = replay_cached(engine, [7, 7, 7, 7, 1], [2, 3, 4, 5], [0, 2, 3])
    assert result["greedy_tokens"] == [2, 3, 4, 5]
    assert result["logits"].dtype == np.float16
    assert np.argmax(result["logits"], axis=1).tolist() == [2, 4, 5]
    assert [(kind, pos, len(table)) for kind, _, pos, table in calls] == [
        ("prefill", 0, 1),
        ("prefill", 3, 2),
        ("decode", 5, 2),
        ("decode", 6, 2),
        ("decode", 7, 2),
    ]
    assert result["cache_reclaimed"] and not engine.cache_pool.allocations
    assert not FakeStore.instances[-1].ids


@pytest.mark.parametrize(
    "prompt,tokens,positions",
    [
        ([], [1], None),
        ([1], [], None),
        ([True], [1], None),
        ([1.2], [1], None),
        ([8], [1], None),
        ([1], [-1], None),
        ([1], [1, 2], [1, 0]),
        ([1], [1, 2], [0, 0]),
        ([1], [1], [1]),
    ],
)
def test_replay_rejects_bad_inputs_before_device_allocation(
    monkeypatch, prompt, tokens, positions
):
    engine, _ = fake_engine(monkeypatch)
    with pytest.raises(ValueError):
        replay_cached(engine, prompt, tokens, positions)
    assert not FakeStore.instances


def test_replay_capacity_idle_guard_and_exception_cleanup(monkeypatch):
    engine, _ = fake_engine(monkeypatch)
    with pytest.raises(ValueError, match="context"):
        replay_cached(engine, [1] * 128, [2])
    engine.cache_pool = KVBlockPool(4, 1, 4)
    with pytest.raises(ValueError, match="KV capacity"):
        replay_cached(engine, [1] * 4, [2])
    engine.cache_pool.reserve(99, 4)
    with pytest.raises(ValueError, match="idle"):
        replay_cached(engine, [1], [2])
    engine.cache_pool.release(99)
    engine.model.decode_paged_batch = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("injected")
    )
    with pytest.raises(RuntimeError, match="injected"):
        replay_cached(engine, [1], [2, 3])
    assert not FakeStore.instances[-1].ids and not engine.cache_pool.allocations


def test_activation_capture_includes_repeated_decode_rows_and_restores_on_error():
    def original(x, *args, **kwargs):
        return x

    model = SimpleNamespace(_linear=original, mx=SimpleNamespace(array=np.array))
    with pytest.raises(RuntimeError):
        with capture_cached_inputs(model, {"cov": ["q", "k"]}, 2) as samples:
            for _ in range(3):
                model._linear(np.ones((1, 5)), "q", rowwise=True)
                model._linear(np.ones((1, 5)), "k")
            model._linear(np.ones((7, 5)), "q")
            raise RuntimeError("injected")
    assert model._linear is original
    assert [len(x) for x in samples["cov"]] == [1, 1, 1, 2]


def test_new_corpora_are_disjoint_from_all_previous_experiments():
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    old = [
        p
        for name in [
            "quality-prompts.json",
            "calibration-prompts.json",
            "second-order-calibration-prompts.json",
            "second-order-regression-prompts.json",
        ]
        for p in json.loads((root / name).read_text())
    ]
    calibration = json.loads((root / "cached-calibration-prompts.json").read_text())
    regression = json.loads((root / "cached-regression-prompts.json").read_text())
    assert len(calibration) == 32 and len(regression) == 25
    validate_corpora(calibration, old + regression)
