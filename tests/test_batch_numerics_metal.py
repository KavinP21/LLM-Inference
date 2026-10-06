from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from forge_llm.mlx_engine import MlxEngine
from forge_llm.paged_kv import MlxPagedKVStore
from forge_llm.quantization import quantize_model
from test_quantization import tiny_artifact


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
@pytest.mark.parametrize("precision", ["fp16", "metal", "reconstruct", "dequantize"])
def test_rowwise_decode_has_exact_independent_logits_and_retains_batched_attention(
    tmp_path, monkeypatch, family, precision
):
    source, packed = tmp_path / "fp16.engine", tmp_path / "int8.engine"
    tiny_artifact(source, family)
    if precision != "fp16":
        quantize_model(
            source, packed, retain_fp16=["model.layers.0.mlp.down_proj.weight"]
        )
    prompts = [[(i * 7) % 32 for i in range(n)] for n in [15, 16, 17, 31, 32, 33]]
    with MlxEngine(
        source if precision == "fp16" else packed,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        decode_mode="rowwise",
        int8_mode="auto" if precision == "fp16" else precision,
    ) as engine:
        model = engine.model
        stores = [
            MlxPagedKVStore(
                mx,
                num_layers=model.config.num_hidden_layers,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=model.head_dim,
            )
            for _ in range(2)
        ]
        tables = [tuple(range(i * 4, i * 4 + 4)) for i in range(len(prompts))]
        for prompt, table in zip(prompts, tables):
            for store in stores:
                model.forward_paged_chunk(
                    prompt, start_position=0, block_table=table, store=store
                )
        attention = model._paged_decode_attention_batch
        widths = []

        def traced(query, *args, **kwargs):
            widths.append(query.shape[0])
            return attention(query, *args, **kwargs)

        monkeypatch.setattr(model, "_paged_decode_attention_batch", traced)
        try:
            for step in range(8):
                tokens = [(i + step) % 32 for i in range(len(prompts))]
                positions = [len(p) + step for p in prompts]
                expected = [
                    np.array(
                        model.decode_paged_batch(
                            [t], positions=[p], block_tables=[table], store=stores[0]
                        )[0]
                    )
                    for t, p, table in zip(tokens, positions, tables)
                ]
                actual = np.array(
                    model.decode_paged_batch(
                        tokens,
                        positions=positions,
                        block_tables=tables,
                        store=stores[1],
                    )
                )
                np.testing.assert_array_equal(actual, np.stack(expected))
            assert widths.count(len(prompts)) == 8 * model.config.num_hidden_layers
            assert engine.stats()["decode_mode"] == "rowwise"
        finally:
            for store in stores:
                store.clear()
        assert engine.stats()["kv_device_bytes"] == 0


def test_rowwise_decode_failure_reclaims_all_scheduled_pages(tmp_path, monkeypatch):
    source = tmp_path / "model.engine"
    tiny_artifact(source)
    with MlxEngine(
        source, max_model_length=64, kv_cache_bytes=1 << 20, decode_mode="rowwise"
    ) as engine:
        requests = [engine.submit([1, 2, 3], 8) for _ in range(2)]
        engine.step()
        engine.step()

        def fail(*args, **kwargs):
            raise RuntimeError("injected decode failure")

        monkeypatch.setattr(engine.model, "_decode_linear", fail)
        with pytest.raises(RuntimeError, match="injected"):
            engine.step()
        assert engine.stats()["kv_device_bytes"] == 0
        assert engine.stats()["kv_cache"]["reserved_blocks"] == 0
        assert all(
            engine.scheduler.request(r).state.name == "CANCELLED" for r in requests
        )
