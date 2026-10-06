from __future__ import annotations

import sys

import numpy as np
import pytest

from forge_llm.mlx_engine import MlxEngine
from forge_llm.runtime import SequenceState
from test_quantization import tiny_artifact

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


def options(**kwargs):
    return dict(
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=7,
        decode_mode="rowwise",
        **kwargs,
    )


def run(engine, prompt, budget=8, namespace=""):
    rows, calls = [], []
    prefill, decode = engine._prefill_chunk, engine.model.decode_paged_batch

    def observe_prefill(request_id):
        result = prefill(request_id)
        request = engine.scheduler.request(request_id)
        if result[1] == len(request.prompt):
            rows.append(np.array(result[0][-1]))
        calls.append(result[1])
        return result

    def observe_decode(*a, **k):
        logits = decode(*a, **k)
        rows.extend(np.array(row) for row in logits)
        return logits

    engine._prefill_chunk, engine.model.decode_paged_batch = (
        observe_prefill,
        observe_decode,
    )
    try:
        tokens = engine.generate(prompt, budget, cache_namespace=namespace)
    finally:
        engine._prefill_chunk, engine.model.decode_paged_batch = prefill, decode
    return tokens, np.stack(rows), calls


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
@pytest.mark.parametrize("length", [15, 16, 17, 31, 32, 33])
def test_cold_exact_hit_and_partial_branch_are_bitwise_equal(tmp_path, family, length):
    path = tmp_path / "model.engine"
    tiny_artifact(path, family)
    prompt = [i % 32 for i in range(length)]
    with (
        MlxEngine(path, **options()) as baseline,
        MlxEngine(path, prefix_cache_bytes=1 << 18, **options()) as cached,
    ):
        reference = run(baseline, prompt)
        cold, warm = run(cached, prompt), run(cached, prompt)
        assert reference[0] == cold[0] == warm[0]
        assert reference[1].tobytes() == cold[1].tobytes() == warm[1].tobytes()
        assert len(warm[2]) == 1
        # A changed suffix reuses only the original 7-token chunk checkpoints.
        branch = prompt[:-1] + [(prompt[-1] + 1) % 32]
        expected, actual = run(baseline, branch), run(cached, branch)
        assert expected[0] == actual[0]
        assert expected[1].tobytes() == actual[1].tobytes()
        assert cached.stats()["prefix_cache"]["hits"] >= 2
        assert cached.stats()["kv_cache"]["cow_copies"] > 0
        assert cached.stats()["kv_cache"]["reserved_blocks"] == 0
        cached.clear_prefix_cache()
        assert cached.stats()["kv_device_bytes"] == 0
        assert cached.stats()["kv_cache"]["allocated_blocks"] == 0


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_shared_decode_pages_cancel_evict_failure_and_close(tmp_path, family):
    path = tmp_path / "model.engine"
    tiny_artifact(path, family)
    engine = MlxEngine(path, prefix_cache_bytes=1 << 18, **options())
    prompt = [i % 32 for i in range(31)]
    engine.generate(prompt, 2)
    first, second = engine.submit(prompt, 8), engine.submit(prompt, 8)
    assert engine.cache_pool.block_table(first) == engine.cache_pool.block_table(second)
    engine.step()
    engine.step()
    assert (
        engine.cache_pool.block_table(first)[0]
        == engine.cache_pool.block_table(second)[0]
    )
    # The shared partial tail becomes a distinct physical page per writer.
    assert (
        engine.cache_pool.block_table(first)[1]
        != engine.cache_pool.block_table(second)[1]
    )
    engine.clear_prefix_cache()
    engine.cancel(first)
    assert engine.scheduler.request(second).state is SequenceState.RUNNING

    def fail(*a, **k):
        raise RuntimeError("injected decode failure")

    engine.model.decode_paged_batch = fail
    with pytest.raises(RuntimeError, match="injected"):
        engine.step()
    assert engine.stats()["kv_device_bytes"] == 0
    assert engine.stats()["kv_cache"]["allocated_blocks"] == 0
    assert engine.stats()["kv_cache"]["reserved_blocks"] == 0
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.submit(prompt, 1)


def test_namespace_execution_identity_and_rejected_int8(tmp_path):
    from forge_llm.quantization import quantize_model

    path, packed = tmp_path / "model.engine", tmp_path / "int8.engine"
    tiny_artifact(path)
    with MlxEngine(path, prefix_cache_bytes=1 << 18, **options()) as engine:
        prompt = list(range(15))
        engine.generate(prompt, 2, cache_namespace="tenant A")
        engine.generate(prompt, 2, cache_namespace="tenant B")
        assert engine.stats()["prefix_cache"]["hits"] == 0
        engine.generate(prompt, 2, cache_namespace="tenant A")
        assert engine.stats()["prefix_cache"]["hits"] == 1
        engine.clear_prefix_cache("tenant A")
        assert all(
            e.namespace == "tenant B" for e in engine.prefix_cache.entries.values()
        )
        engine.prefill_chunk_size = 8
        with pytest.raises(RuntimeError, match="settings changed"):
            engine.submit(prompt, 1)
    quantize_model(path, packed)
    with pytest.raises(ValueError, match="FP16"):
        MlxEngine(packed, prefix_cache_bytes=1 << 18, **options())


def test_clone_aliases_immutable_arrays_and_store_rejects_direct_shared_write(tmp_path):
    path = tmp_path / "model.engine"
    tiny_artifact(path)
    with MlxEngine(path, prefix_cache_bytes=1 << 18, **options()) as engine:
        engine.generate(list(range(15)), 1)
        entry = engine.prefix_cache.find(list(range(15)), "")
        source = entry.blocks[-1]
        page = np.array(engine.kv_store._blocks[source].layers[0])
        key = engine.model.mx.zeros((1, 1, 4), dtype=engine.model.mx.float16)
        with pytest.raises(RuntimeError, match="copy-on-write"):
            engine.kv_store.write_layer(0, entry.blocks, 15, key, key)
        request = engine.submit(list(range(15)), 3)
        engine.step()
        engine.step()
        np.testing.assert_array_equal(
            page, np.array(engine.kv_store._blocks[source].layers[0])
        )
        engine.cancel(request)


def test_close_releases_waiting_hits_and_partial_prefills(tmp_path):
    path = tmp_path / "model.engine"
    tiny_artifact(path)
    engine = MlxEngine(path, prefix_cache_bytes=1 << 18, **options())
    engine.generate(list(range(15)), 1)
    engine.submit(list(range(15)), 8)
    engine.submit(list(range(20)), 8)
    engine.step()
    engine.close()
    assert engine.stats()["kv_cache"]["allocated_blocks"] == 0
    assert engine.stats()["kv_cache"]["reserved_blocks"] == 0
    assert engine.stats()["kv_device_bytes"] == 0


def test_allocator_failure_after_admission_reclaims_request(tmp_path, monkeypatch):
    path = tmp_path / "model.engine"
    tiny_artifact(path)
    with MlxEngine(path, prefix_cache_bytes=1 << 18, **options()) as engine:
        request = engine.submit(list(range(20)), 8)

        def fail(*a, **k):
            raise RuntimeError("injected allocation failure")

        monkeypatch.setattr(engine.cache_pool, "ensure_tokens", fail)
        with pytest.raises(RuntimeError, match="injected allocation"):
            engine.step()
        assert engine.scheduler.request(request).state is SequenceState.CANCELLED
        assert engine.stats()["kv_cache"]["reserved_blocks"] == 0
        assert engine.stats()["kv_device_bytes"] == 0
