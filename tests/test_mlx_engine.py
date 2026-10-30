"""The MLX engine against NumPy and kernel references (needs Apple silicon)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from conftest import tiny_artifact
from instinct.backends.metal_kernels import paged_decode_attention
from instinct.format import ModelConfig, write_engine
from instinct.mlx_engine import MlxEngine
from instinct.runtime import SequenceState
from instinct.sampling import SamplingParams


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


def _engine(path, **options):
    settings = dict(
        max_num_sequences=4,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=5,
        prefill_burst_size=8,
        page_size=4,
    )
    settings.update(options)
    return MlxEngine(path, **settings)


def _run(engine, prompts, count, **options):
    ids = [engine.submit(p, count, **options) for p in prompts]
    while engine.has_work():
        engine.step()
    return [engine.request(i).output for i in ids]


def _drained(engine):
    stats = engine.stats()
    return all(
        p["used_pages"] == 0 and p["reserved_pages"] == 0
        for p in stats["kv_pools"].values()
    )


@pytest.mark.parametrize("length", [1, 3, 4, 5, 9, 16, 17])
def test_prefill_logits_match_numpy_across_page_and_chunk_edges(tmp_path, length):
    path = tmp_path / "tiny.engine"
    config, weights = _fixture(path)
    prompt = [(i * 5 + 3) % config.vocab_size for i in range(length)]
    with _engine(path) as engine:
        actual = engine.prefill_logits(prompt)
        assert _drained(engine)
    np.testing.assert_allclose(
        actual, _reference(config, weights, prompt)[-1], atol=3e-3, rtol=3e-2
    )


def test_greedy_decode_matches_numpy(tmp_path):
    path = tmp_path / "tiny.engine"
    config, weights = _fixture(path)
    prompt, expected = [3, 1, 4, 1, 5, 9], []
    for _ in range(10):
        logits = _reference(config, weights, prompt + expected)[-1]
        expected.append(int(np.argmax(logits)))
    with _engine(path) as engine:
        assert engine.generate(prompt, 10) == expected
        assert _drained(engine)


@pytest.mark.parametrize("family", ["qwen2", "gemma3_text"])
def test_batched_requests_match_running_alone(tmp_path, family):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path, family)
    prompts = [
        [(i * 7 + j) % 32 for i in range(n)]
        for j, n in enumerate([1, 4, 9, 13, 21, 30])
    ]
    with _engine(path, max_num_sequences=3) as engine:
        alone = [engine.generate(p, 8) for p in prompts]
        together = _run(engine, prompts, 8)
        assert together == alone
        assert _drained(engine)


def test_sliding_window_pages_are_released_as_the_window_moves(tmp_path):
    path = tmp_path / "gemma.engine"
    tiny_artifact(path, "gemma3_text")  # window 4
    with _engine(path, max_num_sequences=1) as engine:
        engine.submit(list(range(1, 30)), 25)
        peak = 0
        while engine.has_work():
            engine.step()
            peak = max(peak, engine.cache.pools["sliding"].used)
        # Window 4 plus a burst of 8 fits in four pages of four rows.
        assert peak <= 4
        assert engine.cache.pools["full"].high_water >= 54 // 4
        assert _drained(engine)


def test_prefix_cache_reuses_finished_requests_and_checkpoints(tmp_path):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path)
    system = [(i * 3 + 1) % 32 for i in range(18)]
    with _engine(path, prefix_cache_bytes=1 << 18) as cached, _engine(path) as plain:
        first = cached.generate(system + [5, 6], 6, checkpoints=[len(system)])
        assert cached.stats()["prefix_cache"]["entries"] == 2
        # A follow-up that extends the finished conversation...
        follow = system + [5, 6] + first[:-1] + [7, 8, 9]
        assert cached.generate(follow, 6) == plain.generate(follow, 6)
        # ...and a sibling that shares only the checkpointed system prompt.
        sibling = system + [11, 12, 13]
        assert cached.generate(sibling, 6) == plain.generate(sibling, 6)
        stats = cached.stats()
        assert stats["prefix_cache"]["hits"] == 2
        assert stats["reused_tokens"] == len(system) + len(system) + 2 + len(first) - 1
        # Another namespace never sees those entries.
        cached.generate(sibling, 2, cache_namespace="other")
        assert cached.stats()["prefix_cache"]["hits"] == 2
        cached.clear_prefix_cache()
        assert _drained(cached)


def test_prefix_cache_eviction_makes_room_for_new_work(tmp_path):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path)
    # 14 pages of 128 bytes; each 24-token request needs 6.
    with _engine(path, kv_cache_bytes=14 * 128, prefix_cache_bytes=14 * 128) as engine:
        for start in range(4):
            engine.generate([(start + i) % 32 for i in range(20)], 4)
        assert engine.stats()["prefix_cache"]["evicted"] > 0
        engine.clear_prefix_cache()
        assert _drained(engine)


def test_cancel_and_errors_release_pages(tmp_path):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path)
    with _engine(path) as engine:
        request = engine.submit(list(range(1, 30)), 5)
        engine.step()
        assert engine.request(request).state is SequenceState.PREFILLING
        engine.cancel(request)
        assert engine.request(request).state is SequenceState.CANCELLED
        assert _drained(engine)
        engine.forget(request)
        with pytest.raises(ValueError, match="vocabulary"):
            engine.submit([99], 1)
        with pytest.raises(ValueError, match="context"):
            engine.submit([1] * 60, 10)
    with pytest.raises(RuntimeError, match="closed"):
        engine.submit([1], 1)


def test_forced_tokens_replace_sampling(tmp_path):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path)
    with _engine(path) as engine:
        request = engine.submit([1, 2, 3], 4)
        engine.step()
        engine.force(request, [7, 8])
        while engine.has_work():
            engine.step()
        assert engine.request(request).output[1:3] == [7, 8]


def test_sampling_settings_per_request(tmp_path):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path)
    prompt = [4, 9, 2]
    seeded = SamplingParams(temperature=1.0, top_p=0.9, top_k=20, seed=7)
    with _engine(path) as engine:
        greedy = engine.generate(prompt, 8)
        assert (
            engine.generate(
                prompt, 8, sampling=SamplingParams(temperature=1.0, top_k=1)
            )
            == greedy
        )
        first = engine.generate(prompt, 8, sampling=seeded)
        # Same seed, now batched next to greedy requests: same tokens.
        mixed = _run(engine, [prompt, prompt], 8, sampling=seeded)
        assert mixed == [first, first]
    with pytest.raises(ValueError):
        SamplingParams(top_p=0)
    with pytest.raises(ValueError):
        SamplingParams(temperature=-1)


@pytest.mark.parametrize(
    "q_heads, kv_heads, head_dim, lengths, window, softcap",
    [
        (16, 2, 512, [1, 700, 2049], 0, 0.0),
        (16, 8, 256, [5, 1500, 4000], 1024, 0.0),
        (16, 4, 256, [64, 3333], 0, 0.0),
        (8, 1, 80, [3, 300], 0, 30.0),
        (2, 1, 4, [9], 4, 0.0),
    ],
)
def test_paged_decode_kernel_matches_dense_attention(
    q_heads, kv_heads, head_dim, lengths, window, softcap
):
    page = 16
    mx.random.seed(0)
    pages = [math.ceil(n / page) for n in lengths]
    order = np.random.default_rng(0).permutation(sum(pages) + 3)
    tables, cursor = [], 0
    for count in pages:
        tables.append(list(order[cursor : cursor + count]) + [0] * (max(pages) - count))
        cursor += count
    rows = (sum(pages) + 3) * page
    keys = mx.random.normal((rows, kv_heads, head_dim)).astype(mx.bfloat16)
    values = mx.random.normal((rows, kv_heads, head_dim)).astype(mx.bfloat16)
    queries = mx.random.normal((len(lengths), q_heads, head_dim)).astype(mx.bfloat16)
    scale = head_dim**-0.5
    out = paged_decode_attention(
        mx,
        queries,
        keys,
        values,
        mx.array(np.array(tables, dtype=np.int32)),
        mx.array(lengths, dtype=mx.int32),
        page_size=page,
        longest=max(lengths),
        scale=scale,
        window=window,
        softcap=softcap,
    )
    for b, n in enumerate(lengths):
        positions = range(max(0, n - window) if window else 0, n)
        index = mx.array(
            [int(tables[b][t // page]) * page + t % page for t in positions]
        )
        k = mx.repeat(
            keys[index].astype(mx.float32).transpose(1, 0, 2), q_heads // kv_heads, 0
        )
        v = mx.repeat(
            values[index].astype(mx.float32).transpose(1, 0, 2), q_heads // kv_heads, 0
        )
        scores = (queries[b].astype(mx.float32)[:, None, :] @ k.transpose(0, 2, 1))[
            :, 0
        ] * scale
        if softcap:
            scores = mx.tanh(scores / softcap) * softcap
        expected = (mx.softmax(scores, axis=-1)[:, None, :] @ v)[:, 0]
        error = float(mx.abs(out[b].astype(mx.float32) - expected).max())
        assert error < 2e-2, error


def test_gated_delta_kernel_matches_the_recurrence():
    from instinct.backends.metal_kernels import gated_delta

    rng = np.random.default_rng(3)
    batch, steps, k_heads, v_heads, k_dim, v_dim = 2, 7, 2, 4, 32, 16
    q = rng.normal(size=(batch, steps, k_heads, k_dim)).astype(np.float32) * 0.3
    k = rng.normal(size=(batch, steps, k_heads, k_dim)).astype(np.float32) * 0.3
    v = rng.normal(size=(batch, steps, v_heads, v_dim)).astype(np.float32)
    g = rng.uniform(0.5, 1.0, size=(batch, steps, v_heads)).astype(np.float32)
    beta = rng.uniform(0.0, 1.0, size=(batch, steps, v_heads)).astype(np.float32)
    state = rng.normal(size=(batch, v_heads, v_dim, k_dim)).astype(np.float32) * 0.1
    out, final = gated_delta(mx, *(mx.array(a) for a in (q, k, v, g, beta, state)))
    expected = np.zeros_like(v)
    s = state.copy()
    for t in range(steps):
        for h in range(v_heads):
            kh = h // (v_heads // k_heads)
            s[:, h] *= g[:, t, h, None, None]
            memory = np.einsum("bvk,bk->bv", s[:, h], k[:, t, kh])
            delta = (v[:, t, h] - memory) * beta[:, t, h, None]
            s[:, h] += delta[:, :, None] * k[:, t, kh, None, :]
            expected[:, t, h] = np.einsum("bvk,bk->bv", s[:, h], q[:, t, kh])
    np.testing.assert_allclose(np.array(out), expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(np.array(final), s, atol=1e-4, rtol=1e-4)
