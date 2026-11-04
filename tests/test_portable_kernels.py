"""The plain-MLX versions the engine uses off Metal (NVIDIA through MLX's CUDA
backend, or the CPU) against references, the Metal kernels and fused ops.

``INSTINCT_TEST_DEVICE=cpu`` runs these on MLX's CPU backend, which takes the
same path as CUDA: no Metal kernels, every fused op checked first.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from conftest import tiny_artifact
from instinct.backends import portable
from instinct.mlx_engine import MlxEngine


@pytest.fixture
def kernels(monkeypatch):
    """Set INSTINCT_KERNELS for one test, with fresh fused-op checks."""

    def use(value: str) -> None:
        monkeypatch.setenv("INSTINCT_KERNELS", value)
        portable._fused_ok.clear()

    yield use
    portable._fused_ok.clear()


def _recurrence(q, k, v, g, beta, state):
    """The gated delta rule one token at a time, in float64."""
    q, k, v, g, beta = (np.asarray(a, dtype=np.float64) for a in (q, k, v, g, beta))
    s = np.array(state, dtype=np.float64)
    batch, steps, k_heads, _ = q.shape
    v_heads = v.shape[2]
    out = np.zeros(v.shape)
    for t in range(steps):
        for h in range(v_heads):
            kh = h // (v_heads // k_heads)
            s[:, h] *= g[:, t, h, None, None]
            memory = np.einsum("bvk,bk->bv", s[:, h], k[:, t, kh])
            write = (v[:, t, h] - memory) * beta[:, t, h, None]
            s[:, h] += write[:, :, None] * k[:, t, kh, None, :]
            out[:, t, h] = np.einsum("bvk,bk->bv", s[:, h], q[:, t, kh])
    return out, s


def _delta_inputs(steps, seed=3, k_dim=32, strong_decay=False, correlated=False):
    rng = np.random.default_rng(seed)
    batch, k_heads, v_heads, v_dim = 2, 2, 4, 16
    # Unit-length keys and scaled queries, as Qwen3.5's layers make them.
    k = rng.normal(size=(batch, steps, k_heads, k_dim))
    if correlated:
        # Real keys point in similar directions and writes are near full
        # strength. A power series for the chunk solve overflowed on these.
        k = rng.normal(size=(batch, 1, k_heads, k_dim)) + 0.2 * k
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q = rng.normal(size=(batch, steps, k_heads, k_dim)) / k_dim**0.5
    v = rng.normal(size=(batch, steps, v_heads, v_dim))
    low = 0.01 if strong_decay else 0.95 if correlated else 0.5
    g = rng.uniform(low, 1.0, size=(batch, steps, v_heads))
    beta = rng.uniform(0.9 if correlated else 0.0, 1.0, size=(batch, steps, v_heads))
    state = rng.normal(size=(batch, v_heads, v_dim, k_dim)) * 0.1
    return [a.astype(np.float32) for a in (q, k, v, g, beta, state)]


@pytest.mark.parametrize("steps", [1, 5, 64, 130])
@pytest.mark.parametrize("shape", ["random", "strong_decay", "correlated"])
def test_chunked_gated_delta_matches_the_recurrence(kernels, steps, shape):
    kernels("portable")
    inputs = _delta_inputs(
        steps,
        strong_decay=shape == "strong_decay",
        correlated=shape == "correlated",
    )
    out, final = portable.gated_delta(mx, *(mx.array(a) for a in inputs))
    expected, state = _recurrence(*inputs)
    np.testing.assert_allclose(np.array(out), expected, atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(np.array(final), state, atol=2e-4, rtol=2e-3)


def test_chunked_gated_delta_matches_the_metal_kernel(kernels):
    if not portable.metal_kernels(mx):
        pytest.skip("needs an Apple GPU")
    from instinct.backends.metal_kernels import gated_delta

    inputs = [mx.array(a) for a in _delta_inputs(200, k_dim=128)]
    inputs[2] = inputs[2].astype(mx.bfloat16)
    metal = gated_delta(mx, *inputs)
    kernels("portable")
    plain = portable.gated_delta(mx, *inputs)
    for a, b in zip(metal, plain):
        error = float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())
        assert error < 2e-2, error


def _close(a, b, tolerance):
    a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
    np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_plain_rope_matches_the_fused_op(dtype):
    dtype = getattr(mx, dtype)
    key = mx.random.key(11)
    freqs = mx.array([1.0, 4.0, 90.0, 1e4, mx.inf, mx.inf])
    cases = [
        # Decode: [batch, heads, 1, dim], one position per sequence.
        (mx.random.normal((3, 4, 1, 24), key=key), mx.array([0, 9, 120000])),
        # Prefill: [1, heads, tokens, dim] from one starting position.
        (mx.random.normal((1, 4, 7, 24), key=key), 33),
    ]
    for x, offset in cases:
        x = x.astype(dtype)
        for dims, base, f in ((24, 10000.0, None), (12, None, freqs), (16, 1e6, None)):
            fused = mx.fast.rope(
                x, dims, traditional=False, base=base, scale=1.0, offset=offset, freqs=f
            )
            plain = portable._rope_plain(mx, x, dims, base=base, freqs=f, offset=offset)
            _close(plain, fused, 2e-2)


def test_plain_rms_norm_matches_the_fused_op():
    x = mx.random.normal((5, 3, 96), key=mx.random.key(12)).astype(mx.bfloat16)
    weight = mx.random.normal((96,), key=mx.random.key(13)).astype(mx.bfloat16)
    for w in (weight, None):
        fused = mx.fast.rms_norm(x, w, 1e-6)
        _close(portable._rms_norm_plain(mx, x, w, 1e-6), fused, 2e-2)


@pytest.mark.parametrize("dim", [64, 256, 512])
def test_plain_attention_matches_the_fused_op(dim):
    key = mx.random.key(14)
    q = mx.random.normal((2, 8, 5, dim), key=key).astype(mx.bfloat16)
    k = mx.random.normal((2, 2, 11, dim), key=key).astype(mx.bfloat16)
    v = mx.random.normal((2, 2, 11, dim), key=key).astype(mx.bfloat16)
    boolean = mx.random.uniform(shape=(2, 1, 5, 11), key=key) > 0.4
    boolean[..., -1] = True
    for mask in ("causal", boolean, None):
        fused = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=dim**-0.5, mask=mask
        )
        plain = portable._attention_plain(mx, q, k, v, scale=dim**-0.5, mask=mask)
        _close(plain, fused, 3e-2)


def test_fused_op_checks_pass_on_this_device(kernels):
    kernels("portable")
    report = portable.report(mx)
    assert report["decode_attention"] == report["gated_delta"] == "portable"
    assert {report[n] for n in ("rope", "rms_norm", "attention", "compile")} == {
        "fused"
    }
    kernels("plain")
    assert portable.report(mx)["rope"] == "plain"


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


def _generate_all(path, **options):
    """Greedy replies and the logits behind a prefill and a decode step."""
    prompts = [[(i * 7 + j) % 32 for i in range(n)] for j, n in enumerate([3, 9, 21])]
    with _engine(path, **options) as engine:
        ids = [engine.submit(p, 6) for p in prompts]
        while engine.has_work():
            engine.step()
        replies = [engine.request(i).output for i in ids]
        logits = engine.prefill_logits(prompts[2] + replies[2][:3])
        # A follow-up that reuses the cached prompt and decodes from it.
        follow = engine.generate(prompts[2] + replies[2][:-1] + [5, 6], 4)
        assert engine.stats()["reused_tokens"] > 0
        engine.clear_prefix_cache()
        pools = engine.stats()["kv_pools"].values()
        assert all(p["used_pages"] == p["reserved_pages"] == 0 for p in pools)
    return replies, follow, logits


@pytest.mark.parametrize(
    "family,bits",
    [("qwen2", 16), ("gemma3_text", 16), ("gemma3_text", 8)],
)
def test_engine_agrees_without_metal_kernels(tmp_path, kernels, family, bits):
    """Same replies whichever path runs: paged decode through the dense
    gather and plain attention, sliding windows and 8-bit K/V included."""
    path = tmp_path / "tiny.engine"
    tiny_artifact(path, family)
    options = dict(kv_bits=bits, prefix_cache_bytes=1 << 18)
    results = {}
    for mode in ("auto", "portable", "plain"):
        kernels(mode)
        results[mode] = _generate_all(path, **options)
    replies, follow, logits = results["auto"]
    for mode in ("portable", "plain"):
        assert results[mode][0] == replies and results[mode][1] == follow
        # Plain RoPE and norms round a little differently, which can move an
        # 8-bit K/V value by one step.
        np.testing.assert_allclose(results[mode][2], logits, atol=2e-2, rtol=2e-2)
