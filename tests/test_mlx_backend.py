from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge_llm import create_engine
from forge_llm.backends.mlx import MlxQwenModel
from forge_llm.format import ModelConfig, write_engine
from forge_llm.mlx_engine import MlxEngine, SequenceState

mx = pytest.importorskip("mlx.core")


def _fixture(path: Path) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
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
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        tensors.update({
            prefix + "input_layernorm.weight": np.ones(config.hidden_size, dtype=np.float16),
            prefix + "post_attention_layernorm.weight": np.ones(config.hidden_size, dtype=np.float16),
            prefix + "self_attn.q_proj.weight": weight((config.hidden_size, config.hidden_size)),
            prefix + "self_attn.q_proj.bias": weight((config.hidden_size,), 0.02),
            prefix + "self_attn.k_proj.weight": weight((4, config.hidden_size)),
            prefix + "self_attn.k_proj.bias": weight((4,), 0.02),
            prefix + "self_attn.v_proj.weight": weight((4, config.hidden_size)),
            prefix + "self_attn.v_proj.bias": weight((4,), 0.02),
            prefix + "self_attn.o_proj.weight": weight((config.hidden_size, config.hidden_size)),
            prefix + "mlp.gate_proj.weight": weight((config.intermediate_size, config.hidden_size)),
            prefix + "mlp.up_proj.weight": weight((config.intermediate_size, config.hidden_size)),
            prefix + "mlp.down_proj.weight": weight((config.hidden_size, config.intermediate_size)),
        })
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
    frequencies = 1.0 / (theta ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension))
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


def _reference(config: ModelConfig, weights: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    hidden = weights["model.embed_tokens.weight"][tokens].astype(np.float32)
    head_dim = config.hidden_size // config.num_attention_heads
    positions = np.arange(len(tokens), dtype=np.int32)
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        normalized = _rms_norm(hidden, weights[prefix + "input_layernorm.weight"], config.rms_norm_eps)
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
        scores = q @ k.transpose(0, 2, 1) * (head_dim ** -0.5)
        mask = np.arange(len(tokens))[None, :] <= np.arange(len(tokens))[:, None]
        scores = np.where(mask[None, :, :], scores, -1e9)
        attention = (_softmax(scores) @ v).transpose(1, 0, 2).reshape(len(tokens), -1)
        hidden = hidden + attention @ weights[prefix + "self_attn.o_proj.weight"].T
        normalized = _rms_norm(
            hidden, weights[prefix + "post_attention_layernorm.weight"], config.rms_norm_eps
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
    np.testing.assert_allclose(np.asarray(decoded[-1]), expected_decode, rtol=2e-2, atol=2e-2)
    assert decoded_cache.token_count == len(prompt) + 1


def test_mlx_engine_api_lifecycle_and_cache_accounting(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    config, weights = _fixture(path)
    engine = MlxEngine(path, max_num_sequences=2, max_model_length=32, kv_cache_bytes=1 << 20)
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


def test_mlx_engine_rejects_unsafe_admission(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    # One block exactly for this model fixture.
    engine = MlxEngine(path, max_num_sequences=2, max_model_length=32, kv_cache_bytes=512)
    engine.submit([1], 15, [])
    with pytest.raises(ValueError, match="insufficient KV capacity"):
        engine.submit([2], 1, [])
    assert engine.stats()["kv_cache"]["allocation_failures"] == 1


def test_public_factory_selects_mlx_on_apple_silicon(tmp_path: Path) -> None:
    path = tmp_path / "tiny-qwen.engine"
    _fixture(path)
    engine = create_engine(path, backend="auto", max_model_length=32, kv_cache_bytes=1 << 20)
    assert isinstance(engine, MlxEngine)
    assert engine.build_info()["backend"] == "mlx"
    engine.close()
