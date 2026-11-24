"""How the engine picks the CUDA kernels, without an NVIDIA GPU.

Stand-in kernels play the CUDA ones: a kernel that passes its first-use
check is used, and one that fails it (an error or a wrong answer) falls back
to the portable version, with the same replies either way.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from conftest import tiny_artifact  # noqa: E402
from instinct.backends import cuda_kernels as ck  # noqa: E402
from instinct.backends import portable  # noqa: E402
from instinct.mlx_engine import MlxEngine  # noqa: E402


@pytest.fixture
def cuda(monkeypatch):
    """Pretend the default device is an NVIDIA GPU, with stand-in kernels:
    ``use(decode, delta)`` sets what ``_run_decode`` and ``_run_gated_delta``
    do; the default computes the right answer from plain MLX."""
    monkeypatch.setattr(
        portable,
        "custom_kernels",
        lambda mx_: "cuda" if portable.mode() == "auto" else None,
    )
    ck.reset()
    calls = {"decode": 0, "delta": 0}

    def correct_decode(
        mx_,
        queries,
        keys,
        values,
        tables,
        lengths,
        *,
        page_size,
        longest,
        scale,
        window,
        softcap,
        scales,
    ):
        calls["decode"] += 1
        span = min(longest, window) if window else longest
        rows, valid = portable.decode_rows(
            mx_, tables, lengths, page_size=page_size, span=span
        )
        flat = rows.reshape(-1)
        batch = int(queries.shape[0])
        heads, dim = int(keys.shape[1]), int(keys.shape[2])

        def dense(store, scale_):
            gathered = store[flat].astype(mx_.float32)
            if scale_ is not None:
                gathered = gathered * scale_[flat][..., None]
            gathered = gathered.astype(queries.dtype)
            return gathered.reshape(batch, span, heads, dim).transpose(0, 2, 1, 3)

        out = portable._attention_plain(
            mx_,
            queries[:, :, None, :],
            dense(keys, scales[0] if scales else None),
            dense(values, scales[1] if scales else None),
            scale=scale,
            mask=valid[:, None, None, :],
            softcap=softcap,
        )
        return out[:, :, 0, :]

    def correct_delta(mx_, *args):
        calls["delta"] += 1
        return portable._gated_delta_plain(mx_, *args)

    def use(decode=correct_decode, delta=correct_delta):
        ck.reset()
        monkeypatch.setattr(ck, "_run_decode", decode)
        monkeypatch.setattr(ck, "_run_gated_delta", delta)

    use()
    yield use, calls
    ck.reset()


def _replies(path, bits):
    prompts = [[(i * 7 + j) % 32 for i in range(n)] for j, n in enumerate([3, 9, 21])]
    engine = MlxEngine(
        path,
        max_num_sequences=4,
        kv_cache_bytes=1 << 20,
        prefill_chunk_size=5,
        prefill_burst_size=8,
        page_size=4,
        kv_bits=bits,
    )
    with engine:
        ids = [engine.submit(p, 6) for p in prompts]
        while engine.has_work():
            engine.step()
        return [engine.request(i).output for i in ids], engine.stats()["kernels"]


@pytest.mark.parametrize("family,bits", [("qwen2", 16), ("gemma3_text", 8)])
def test_engine_uses_cuda_kernels_that_pass_and_drops_ones_that_fail(
    tmp_path, monkeypatch, cuda, family, bits
):
    path = tmp_path / "tiny.engine"
    tiny_artifact(path, family)
    monkeypatch.setenv("INSTINCT_KERNELS", "portable")
    expected, kernels = _replies(path, bits)
    assert kernels["decode_attention"] == "portable"
    monkeypatch.setenv("INSTINCT_KERNELS", "auto")
    use, calls = cuda

    got, kernels = _replies(path, bits)
    assert got == expected and calls["decode"] > 0
    assert kernels["decode_attention"] == "cuda" and not ck.fallbacks()

    def broken(*args, **kwargs):
        raise RuntimeError("nvrtc: error")

    for bad in (broken, lambda mx_, queries, *a, **k: mx_.zeros_like(queries)):
        use(decode=bad)
        got, kernels = _replies(path, bits)
        assert got == expected
        assert kernels["decode_attention"] == "portable"
        assert "decode_attention" in ck.fallbacks()


def test_gated_delta_falls_back_when_its_kernel_is_wrong(cuda):
    use, calls = cuda
    rng = mx.random.key(0)
    q = mx.random.normal((1, 3, 2, 32), key=rng).astype(mx.bfloat16)
    k = q / mx.sqrt(mx.sum(q.astype(mx.float32) ** 2, axis=-1, keepdims=True)).astype(
        mx.bfloat16
    )
    v = mx.random.normal((1, 3, 4, 8), key=rng).astype(mx.bfloat16)
    g = mx.full((1, 3, 4), 0.9)
    beta = mx.full((1, 3, 4), 0.5)
    state = mx.zeros((1, 4, 8, 32))
    expected = portable._gated_delta_plain(mx, q, k, v, g, beta, state)
    out = portable.gated_delta(mx, q, k, v, g, beta, state)
    assert calls["delta"] >= 2 and mx.allclose(out[0], expected[0]).item()
    use(delta=lambda mx_, q_, k_, v_, g_, b_, s_: (mx_.zeros_like(v_), s_))
    out = portable.gated_delta(mx, q, k, v, g, beta, state)
    assert mx.allclose(out[0], expected[0]).item()
    assert "gated_delta" in ck.fallbacks()
    # A key dim the kernel can't take goes straight to the plain version.
    ck.reset()
    odd = portable.gated_delta(
        mx, q[..., :20], k[..., :20], v, g, beta, state[..., :20]
    )
    assert odd[0].shape == v.shape and not ck.fallbacks()


def test_the_check_catches_a_kernel_that_ignores_the_window(cuda):
    use, _ = cuda

    def windowless(mx_, *args, window, **kwargs):
        return correct(mx_, *args, window=0, **kwargs)

    correct = ck._run_decode
    use(decode=correct)
    args = dict(
        head_dim=16, q_heads=4, kv_heads=2, page_size=8, softcap=0.0, quantized=False
    )
    assert ck._check_decode(mx, mx.bfloat16, mx.bfloat16, window=64, **args)
    use(decode=windowless)
    assert not ck._check_decode(mx, mx.bfloat16, mx.bfloat16, window=64, **args)
