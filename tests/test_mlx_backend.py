from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from forge_llm import create_engine
from forge_llm.backends.mlx import MlxQwenModel
from forge_llm.format import ModelConfig, write_engine
from forge_llm.mlx_engine import MlxEngine, SequenceState
from forge_llm.paged_kv import MlxPagedKVStore

mx = pytest.importorskip("mlx.core")


def _fixture(
    path: Path,
    *,
    hidden_size: int = 8,
    intermediate_size: int = 16,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    max_position_embeddings: int = 64,
) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        max_position_embeddings=max_position_embeddings,
        eos_token_id=2,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
    )
    random = np.random.default_rng(17)

    def weight(shape: tuple[int, ...], scale: float = 0.12) -> np.ndarray:
        return (random.standard_normal(shape) * scale).astype(np.float16)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": weight((config.vocab_size, config.hidden_size)),
        "model.norm.weight": np.ones(config.hidden_size, dtype=np.float16),
    }
    kv_width = config.hidden_size // config.num_attention_heads
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        tensors.update(
            {
                prefix + "input_layernorm.weight": np.ones(
                    config.hidden_size, dtype=np.float16
                ),
                prefix + "post_attention_layernorm.weight": np.ones(
                    config.hidden_size, dtype=np.float16
                ),
                prefix + "self_attn.q_proj.weight": weight(
                    (config.hidden_size, config.hidden_size)
                ),
                prefix + "self_attn.q_proj.bias": weight((config.hidden_size,), 0.02),
                prefix + "self_attn.k_proj.weight": weight(
                    (kv_width, config.hidden_size)
                ),
                prefix + "self_attn.k_proj.bias": weight((kv_width,), 0.02),
                prefix + "self_attn.v_proj.weight": weight(
                    (kv_width, config.hidden_size)
                ),
                prefix + "self_attn.v_proj.bias": weight((kv_width,), 0.02),
                prefix + "self_attn.o_proj.weight": weight(
                    (config.hidden_size, config.hidden_size)
                ),
                prefix + "mlp.gate_proj.weight": weight(
                    (config.intermediate_size, config.hidden_size)
                ),
                prefix + "mlp.up_proj.weight": weight(
                    (config.intermediate_size, config.hidden_size)
                ),
                prefix + "mlp.down_proj.weight": weight(
                    (config.hidden_size, config.intermediate_size)
                ),
            }
        )
    tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]
    write_engine(
        path,
        config,
        tensors,
        aliases={"lm_head.weight": "model.embed_tokens.weight"},
    )
    return config, tensors


def _rms_norm(x: np.ndarray, weight: np.ndarray, epsilon: float) -> np.ndarray:
    variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return x * (1.0 / np.sqrt(variance + epsilon)) * weight


def _rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    dimension = x.shape[-1]
    frequencies = 1.0 / (
        theta ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension)
    )
    angles = positions.astype(np.float32)[:, None] * frequencies[None, :]
    cosine = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[:, None, :]
    sine = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[:, None, :]
    half = dimension // 2
    rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cosine + rotated * sine


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def _reference(
    config: ModelConfig, weights: dict[str, np.ndarray], tokens: list[int]
) -> np.ndarray:
    hidden = weights["model.embed_tokens.weight"][tokens].astype(np.float32)
    head_dim = config.hidden_size // config.num_attention_heads
    positions = np.arange(len(tokens), dtype=np.int32)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        normalized = _rms_norm(
            hidden, weights[prefix + "input_layernorm.weight"], config.rms_norm_eps
        )
        query = (
            normalized @ weights[prefix + "self_attn.q_proj.weight"].T
            + weights[prefix + "self_attn.q_proj.bias"]
        ).reshape(len(tokens), config.num_attention_heads, head_dim)
        key = (
            normalized @ weights[prefix + "self_attn.k_proj.weight"].T
            + weights[prefix + "self_attn.k_proj.bias"]
        ).reshape(len(tokens), config.num_key_value_heads, head_dim)
        value = (
            normalized @ weights[prefix + "self_attn.v_proj.weight"].T
            + weights[prefix + "self_attn.v_proj.bias"]
        ).reshape(len(tokens), config.num_key_value_heads, head_dim)
        query = _rope(query, positions, config.rope_theta)
        key = _rope(key, positions, config.rope_theta)
        groups = config.num_attention_heads // config.num_key_value_heads
        key = np.repeat(key, groups, axis=1)
        value = np.repeat(value, groups, axis=1)
        q = query.transpose(1, 0, 2)
        k = key.transpose(1, 0, 2)
        v = value.transpose(1, 0, 2)
        scores = q @ k.transpose(0, 2, 1) * (head_dim**-0.5)
        mask = np.arange(len(tokens))[None, :] <= np.arange(len(tokens))[:, None]
        scores = np.where(mask[None, :, :], scores, -1e9)
        attention = (_softmax(scores) @ v).transpose(1, 0, 2).reshape(len(tokens), -1)
        hidden = hidden + attention @ weights[prefix + "self_attn.o_proj.weight"].T
        normalized = _rms_norm(
            hidden,
            weights[prefix + "post_attention_layernorm.weight"],
            config.rms_norm_eps,
        )
        gate = normalized @ weights[prefix + "mlp.gate_proj.weight"].T
        up = normalized @ weights[prefix + "mlp.up_proj.weight"].T
        silu = gate / (1.0 + np.exp(-gate))
        hidden = hidden + (silu * up) @ weights[prefix + "mlp.down_proj.weight"].T
    normalized = _rms_norm(hidden, weights["model.norm.weight"], config.rms_norm_eps)
    return normalized @ weights["lm_head.weight"].T


def test_full_qwen_prefill_and_cached_decode_match_numpy(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    config, weights = _fixture(path)
    model = MlxQwenModel(str(path), max_model_length=32)
    prompt = [1, 7, 11, 5]
    logits, cache = model.forward(prompt)
    actual = np.asarray(logits)
    expected = _reference(config, weights, prompt)
    np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-2)

    token = int(np.argmax(actual[-1]))
    decoded, decoded_cache = model.forward([token], cache)
    expected_decode = _reference(config, weights, prompt + [token])[-1]
    np.testing.assert_allclose(
        np.asarray(decoded[-1]), expected_decode, rtol=2e-2, atol=2e-2
    )
    assert decoded_cache.token_count == len(prompt) + 1


def test_physical_paged_store_crosses_boundaries_and_releases() -> None:
    store = MlxPagedKVStore(
        mx,
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        block_tokens=16,
    )
    key = mx.arange(18 * 4, dtype=mx.float16).reshape(18, 1, 4)
    value = key + 100
    table = (9, 3)
    touched = store.write_layer(0, table, 0, key[:15], value[:15])
    touched += store.write_layer(0, table, 15, key[15:], value[15:])
    store.materialize(touched)
    actual_key, actual_value = store.gather_layer(0, table, 18)
    np.testing.assert_array_equal(np.asarray(actual_key), np.asarray(key))
    np.testing.assert_array_equal(np.asarray(actual_value), np.asarray(value))
    ranged_key, ranged_value = store.gather_layer_range(0, table, 7, 18)
    np.testing.assert_array_equal(np.asarray(ranged_key), np.asarray(key[7:18]))
    np.testing.assert_array_equal(np.asarray(ranged_value), np.asarray(value[7:18]))
    assert store.allocated_blocks == 2
    with pytest.raises(RuntimeError, match="overwrite"):
        store.write_layer(0, table, 17, key[:1], value[:1])
    with pytest.raises(RuntimeError, match="unwritten"):
        store.gather_layer(1, table, 1)
    with pytest.raises(ValueError, match="shorter"):
        store.pack_layer(0, [(9,)], [18])
    store.release(table)
    assert store.allocated_blocks == 0


def test_chunked_paged_prefill_and_batched_decode_match_reference(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny-qwen.engine"
    config, weights = _fixture(path)
    model = MlxQwenModel(str(path), max_model_length=32, attention_tile_size=2)
    store = MlxPagedKVStore(
        mx,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=model.head_dim,
    )
    prompts = ([1, 7, 11, 5], [4, 3, 9])
    tables = [(0,), (1,)]
    first_logits = model.forward_paged_chunk(
        list(prompts[0][:2]), start_position=0, block_table=tables[0], store=store
    )
    assert first_logits.shape == (2, config.vocab_size)
    logits_a = model.forward_paged_chunk(
        list(prompts[0][2:]), start_position=2, block_table=tables[0], store=store
    )
    logits_b = model.forward_paged_chunk(
        list(prompts[1]), start_position=0, block_table=tables[1], store=store
    )
    np.testing.assert_allclose(
        np.asarray(logits_a[-1]),
        _reference(config, weights, list(prompts[0]))[-1],
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(logits_b[-1]),
        _reference(config, weights, list(prompts[1]))[-1],
        rtol=2e-2,
        atol=2e-2,
    )

    decode_tokens = [
        int(mx.argmax(logits_a[-1]).item()),
        int(mx.argmax(logits_b[-1]).item()),
    ]
    decoded = model.decode_paged_batch(
        decode_tokens,
        positions=[len(prompt) for prompt in prompts],
        block_tables=tables,
        store=store,
    )
    for index, prompt in enumerate(prompts):
        expected = _reference(config, weights, list(prompt) + [decode_tokens[index]])[
            -1
        ]
        np.testing.assert_allclose(
            np.asarray(decoded[index]), expected, rtol=2e-2, atol=2e-2
        )


def test_custom_metal_paged_path_matches_mlx_fallback(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    custom = MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=7,
        custom_metal=True,
        metal_paged_attention=True,
    )
    fallback = MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=7,
        custom_metal=False,
        metal_paged_attention=False,
    )
    prompts = ([1 + index % 30 for index in range(17)], [2, 5, 9, 4])
    custom_ids = [custom.submit(prompt, 5, []) for prompt in prompts]
    fallback_ids = [fallback.submit(prompt, 5, []) for prompt in prompts]
    while any(
        custom.scheduler.request(request_id).state is not SequenceState.COMPLETED
        for request_id in custom_ids
    ):
        custom.step()
    while any(
        fallback.scheduler.request(request_id).state is not SequenceState.COMPLETED
        for request_id in fallback_ids
    ):
        fallback.step()
    for custom_id, fallback_id in zip(custom_ids, fallback_ids):
        assert (
            custom.scheduler.request(custom_id).output
            == fallback.scheduler.request(fallback_id).output
        )
    assert custom.stats()["custom_metal"] is True
    assert custom.stats()["metal_paged_attention"] is True
    assert fallback.stats()["custom_metal"] is False
    assert custom.stats()["kv_materialized_blocks"] == 0
    assert fallback.stats()["kv_materialized_blocks"] == 0


def test_engine_prefill_is_chunked_and_decode_continues_between_chunks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    engine = MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=32,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=2,
    )
    first = engine.submit([1, 2, 3], 4, [])
    assert engine.step() == []
    first_event = engine.step()
    assert [event.request_id for event in first_event] == [first]

    second = engine.submit([3, 4, 5, 6, 7], 2, [])
    mixed = engine.step()
    assert [event.request_id for event in mixed] == [first]
    assert engine.scheduler.request(second).state is SequenceState.PREFILLING
    assert engine.step()[0].request_id == first
    final_prefill_step = engine.step()
    assert [event.request_id for event in final_prefill_step] == [first, second]

    engine.cancel(first)
    engine.cancel(second)
    assert engine.stats()["kv_cache"]["allocated_blocks"] == 0
    assert engine.stats()["kv_materialized_blocks"] == 0


def test_cancelling_partial_prefill_reclaims_physical_pages(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    engine = MlxEngine(
        path,
        max_num_sequences=1,
        max_model_length=32,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=2,
    )
    request_id = engine.submit([1, 2, 3, 4, 5], 2, [])
    assert engine.step() == []
    assert engine.stats()["kv_materialized_blocks"] == 1
    engine.cancel(request_id)
    assert engine.scheduler.request(request_id).state is SequenceState.CANCELLED
    assert engine.stats()["kv_cache"]["allocated_blocks"] == 0
    assert engine.stats()["kv_cache"]["reserved_blocks"] == 0
    assert engine.stats()["kv_materialized_blocks"] == 0


def test_fused_decode_batches_different_context_lengths(tmp_path: Path) -> None:
    path = tmp_path / "fast-tiny-qwen.engine"
    _fixture(
        path,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=1,
        max_position_embeddings=512,
    )
    engine = MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=512,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=128,
    )
    first = engine.submit([1 + index % 30 for index in range(300)], 10, [])
    second = engine.submit([1 + index % 30 for index in range(400)], 3, [])
    joint_steps = 0
    while (
        engine.scheduler.request(first).state is not SequenceState.COMPLETED
        or engine.scheduler.request(second).state is not SequenceState.COMPLETED
    ):
        events = engine.step()
        if {event.request_id for event in events} == {first, second}:
            joint_steps += 1
    assert joint_steps == 3
    assert len(engine.scheduler.request(first).output) == 10
    assert len(engine.scheduler.request(second).output) == 3
    expected = (
        list(engine.scheduler.request(first).output),
        list(engine.scheduler.request(second).output),
    )
    assert engine.stats()["kv_cache"]["allocated_blocks"] == 0
    assert engine.stats()["kv_materialized_blocks"] == 0

    fallback = MlxEngine(
        path,
        max_num_sequences=2,
        max_model_length=512,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=128,
        custom_metal=False,
        metal_paged_attention=False,
    )
    fallback_ids = (
        fallback.submit([1 + index % 30 for index in range(300)], 10, []),
        fallback.submit([1 + index % 30 for index in range(400)], 3, []),
    )
    while any(
        fallback.scheduler.request(request_id).state is not SequenceState.COMPLETED
        for request_id in fallback_ids
    ):
        fallback.step()
    assert (
        tuple(
            fallback.scheduler.request(request_id).output for request_id in fallback_ids
        )
        == expected
    )


@pytest.mark.parametrize("prompt_length", [15, 16, 17, 31, 32, 33])
def test_paged_model_block_boundary_matrix(tmp_path: Path, prompt_length: int) -> None:
    path = tmp_path / "tiny-qwen.engine"
    config, weights = _fixture(path)
    engine = MlxEngine(
        path,
        max_num_sequences=1,
        max_model_length=64,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=7,
    )
    prompt = [1 + (index % (config.vocab_size - 1)) for index in range(prompt_length)]
    expected = int(np.argmax(_reference(config, weights, prompt)[-1]))
    engine.kv_store.reset_peak()
    assert engine.generate(prompt, 1, []) == [expected]
    stats = engine.stats()
    assert stats["kv_peak_materialized_blocks"] == math.ceil(prompt_length / 16)
    assert stats["kv_cache"]["allocated_blocks"] == 0
    assert stats["kv_materialized_blocks"] == 0


def test_mlx_engine_api_lifecycle_and_cache_accounting(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    config, weights = _fixture(path)
    engine = MlxEngine(
        path, max_num_sequences=2, max_model_length=32, kv_cache_bytes=1 << 20
    )
    first = engine.submit([1, 2, 3], 4, [])
    first_events = engine.step()
    assert [event.request_id for event in first_events] == [first]
    second = engine.submit([4, 5], 3, [])
    mixed_events = engine.step()
    assert [event.request_id for event in mixed_events] == [first, second]
    engine.cancel(second)
    assert engine.scheduler.request(second).state is SequenceState.CANCELLED

    while engine.scheduler.request(first).state is not SequenceState.COMPLETED:
        engine.step()
    stats = engine.stats()
    assert stats["scheduler"]["completed"] == 1
    assert stats["scheduler"]["cancelled"] == 1
    assert stats["kv_cache"]["allocated_blocks"] == 0
    assert stats["kv_cache"]["reserved_blocks"] == 0

    expected: list[int] = []
    context = [6, 7]
    for _ in range(3):
        token = int(np.argmax(_reference(config, weights, context)[-1]))
        expected.append(token)
        context.append(token)
    assert engine.generate([6, 7], 3, []) == expected
    np.testing.assert_allclose(
        np.asarray(engine.debug_prefill_logits([6, 7])),
        _reference(config, weights, [6, 7])[-1],
        rtol=2e-2,
        atol=2e-2,
    )


def test_mlx_engine_rejects_unsafe_admission(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    # One block exactly for this model fixture.
    engine = MlxEngine(
        path, max_num_sequences=2, max_model_length=32, kv_cache_bytes=512
    )
    engine.submit([1], 15, [])
    with pytest.raises(ValueError, match="insufficient KV capacity"):
        engine.submit([2], 1, [])
    assert engine.stats()["kv_cache"]["allocation_failures"] == 1


def test_public_factory_selects_mlx_on_apple_silicon(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    engine = create_engine(
        path, backend="auto", max_model_length=32, kv_cache_bytes=1 << 20
    )
    assert isinstance(engine, MlxEngine)
    assert engine.build_info()["backend"] == "mlx"
    engine.close()
