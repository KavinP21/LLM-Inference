from __future__ import annotations

import sys

import numpy as np
import pytest
from forge_llm.mlx_engine import MlxEngine
from forge_llm.speculative import NoDraft, SpeculativeEngine
from test_quantization import tiny_artifact

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires Apple Metal")


class ReplayDraft:
    def __init__(self, prompt, output, *, wrong_at=None):
        self.prompt_length = len(prompt)
        self.output = output
        self.wrong_at = wrong_at

    def start(self, prompt, budget):
        pass

    def propose(self, history, count, cancelled):
        offset = len(history) - self.prompt_length
        result = self.output[offset : offset + count]
        if self.wrong_at is not None and len(result) > self.wrong_at:
            result[self.wrong_at] = (result[self.wrong_at] + 1) % 32
        return result

    def finish(self):
        pass


def options(**kwargs):
    return dict(
        max_model_length=64, kv_cache_bytes=1 << 20, prefill_chunk_size=7, **kwargs
    )


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
@pytest.mark.parametrize("length", [15, 16, 17, 31, 32, 33])
def test_block_accept_reject_and_replacement_match_greedy_on_tiny_models(
    tmp_path, family, length
):
    path = tmp_path / "model.engine"
    tiny_artifact(path, family)
    prompt = [i % 32 for i in range(length)]
    with MlxEngine(path, decode_mode="rowwise", **options()) as baseline:
        expected = baseline.generate(prompt, 12)
    with SpeculativeEngine(path, draft=NoDraft(), **options()) as canonical:
        assert canonical.generate(prompt, 12) == expected
    for wrong_at in [None, 0, 1, 3]:
        draft = ReplayDraft(prompt, expected, wrong_at=wrong_at)
        with SpeculativeEngine(
            path, draft=draft, adaptive=False, **options()
        ) as engine:
            result = engine.generate_result(prompt, 12)
            assert result.tokens == expected
            assert result.stats.verification_blocks > 0
            if wrong_at is not None:
                assert result.stats.rollback_tokens > 0
            assert engine.stats()["kv_device_bytes"] == 0
            # After cleanup a second run must not observe rejected K/V values.
            assert engine.generate(prompt, 12) == expected


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_real_store_rollback_rewrite_zeros_rejected_values(tmp_path, family):
    path = tmp_path / "model.engine"
    tiny_artifact(path, family)
    prompt = list(range(15))
    with SpeculativeEngine(path, **options()) as engine:
        target = engine.target
        first = target.start(prompt, 16)
        target.append([first, 5, 6, 7], block=True)
        target.truncate(16)
        table = target.table
        assert target.store._blocks[table[0]].written_masks == [65535, 65535]
        assert table[1] not in target.store._blocks
        target.append([8, 9], block=True)
        snapshots = [np.array(layer) for layer in target.store._blocks[table[1]].layers]
        target.truncate(17)
        for layer, snapshot in zip(target.store._blocks[table[1]].layers, snapshots):
            np.testing.assert_array_equal(np.array(layer)[:, :1], snapshot[:, :1])
            assert not np.array(layer)[:, 1:].any()
        target.append([10], block=False)
        assert target.store._blocks[table[1]].written_masks == [3, 3]
        target.finish()


def test_model_draft_context_compatibility_resources_and_close(tmp_path):
    qwen, gemma = tmp_path / "qwen.engine", tmp_path / "gemma.engine"
    tiny_artifact(qwen)
    tiny_artifact(gemma, "gemma3_text")
    with pytest.raises(ValueError, match="vocabulary/family"):
        SpeculativeEngine(qwen, draft_model_path=gemma, **options())
    with SpeculativeEngine(
        qwen, draft_model_path=qwen, adaptive=False, **options()
    ) as engine:
        result = engine.generate_result([1, 2, 3], 12)
        assert result.stats.accepted_draft_tokens > 0
        assert engine.target.store.allocated_bytes == 0
        assert engine.draft.target.store.allocated_bytes == 0
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate([1], 1)
    with SpeculativeEngine(qwen, kv_cache_bytes=128, max_model_length=64) as engine:
        with pytest.raises(ValueError, match="KV capacity"):
            engine.generate([1], 1)
        assert engine.target.store.allocated_bytes == 0


def test_sequential_fallback_exact_and_target_forward_failure_cleanup(tmp_path):
    path = tmp_path / "model.engine"
    tiny_artifact(path)
    prompt = list(range(17))
    with SpeculativeEngine(path, draft=NoDraft(), **options()) as baseline:
        expected = baseline.generate(prompt, 12)
    with SpeculativeEngine(
        path,
        draft=ReplayDraft(prompt, expected),
        verification_mode="sequential",
        **options(),
    ) as engine:
        result = engine.generate_result(prompt, 12)
        assert result.tokens == expected
        assert result.stats.verification_blocks == 0
    with SpeculativeEngine(
        path, draft=ReplayDraft(prompt, expected), **options()
    ) as engine:
        original = engine.model.forward_paged_chunk

        def fail(*args, **kwargs):
            if kwargs["start_position"] >= len(prompt):
                raise RuntimeError("injected verification failure")
            return original(*args, **kwargs)

        engine.model.forward_paged_chunk = fail
        with pytest.raises(RuntimeError, match="injected"):
            engine.generate(prompt, 12)
        assert engine.target.store.allocated_bytes == 0
        engine.model.forward_paged_chunk = original
        assert engine.generate(prompt, 12) == expected


def test_cancellation_between_prefill_chunks_reclaims_pages(tmp_path):
    import threading

    path = tmp_path / "model.engine"
    tiny_artifact(path)
    event = threading.Event()
    with SpeculativeEngine(path, **options()) as engine:
        original = engine.model.forward_paged_chunk
        calls = []

        def cancel(*args, **kwargs):
            logits = original(*args, **kwargs)
            calls.append(kwargs["start_position"])
            event.set()
            return logits

        engine.model.forward_paged_chunk = cancel
        result = engine.generate_result(list(range(31)), 8, cancelled=event.is_set)
        assert result.tokens == []
        assert result.finish_reason == "cancelled"
        assert calls == [0]
        assert engine.stats()["kv_device_bytes"] == 0


def test_distinct_model_artifacts_require_matching_tokenizer_fingerprints(tmp_path):
    from forge_llm.format import write_engine
    from forge_llm.model_file import ModelFile

    path, different = tmp_path / "model.engine", tmp_path / "different.engine"
    tiny_artifact(path)
    with ModelFile(path) as source:
        tensors = {name: np.array(source.tensor_numpy(name)) for name in source.tensors}
        tensors["model.layers.0.input_layernorm.weight"] *= np.float16(1.01)
        write_engine(different, source.config, tensors)
    with pytest.raises(ValueError, match="tokenizer fingerprints"):
        SpeculativeEngine(path, draft_model_path=different, **options())
    with pytest.raises(ValueError, match="tokenizer fingerprints"):
        SpeculativeEngine(
            path,
            draft_model_path=different,
            target_tokenizer_fingerprint="a",
            draft_tokenizer_fingerprint="b",
            **options(),
        )
    with SpeculativeEngine(
        path,
        draft_model_path=different,
        target_tokenizer_fingerprint="same",
        draft_tokenizer_fingerprint="same",
        **options(),
    ) as engine:
        assert len(engine.generate([1, 2], 4)) == 4
