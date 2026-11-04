"""The engine's GPU work in plain MLX ops, for devices without Metal.

``metal_kernels`` holds hand-written Metal kernels for paged decode attention
and Qwen3.5's gated delta rule. They only run on Apple GPUs. Everywhere else
(NVIDIA through MLX's CUDA backend, or the CPU) the engine uses the versions
here, which compute the same thing from ordinary MLX ops.

MLX's fused ops (RoPE with per-sequence offsets or explicit frequencies,
attention at head dim 512, RMS norm without a weight, ``mx.compile``) may be
missing for some shape on a backend, and a missing op only fails when the
graph is evaluated, mid-step. So off Metal each one is tried once on tiny
inputs against a plain version, and a plain version replaces any that fails
or disagrees. ``report`` says which were used.

``INSTINCT_KERNELS`` overrides the choice: ``portable`` skips the Metal
kernels, and ``plain`` also replaces every fused op (for checking a device
or the plain versions themselves).
"""

from __future__ import annotations

import os
from typing import Any

CHUNK = 64  # Tokens per block of the chunked gated delta rule.

_fused_ok: dict[str, bool] = {}
_compile_disabled = False


def mode() -> str:
    value = os.environ.get("INSTINCT_KERNELS", "auto").strip().lower() or "auto"
    if value not in {"auto", "portable", "plain"}:
        raise ValueError("INSTINCT_KERNELS must be auto, portable or plain")
    return value


def metal_kernels(mx: Any) -> bool:
    """Whether the custom Metal kernels can run on the default device."""
    if mode() != "auto":
        return False
    metal = getattr(mx, "metal", None)
    try:
        return bool(metal and metal.is_available()) and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError):
        return False


def report(mx: Any) -> dict[str, str]:
    """Which implementation each piece of work uses on this device."""
    custom = "metal" if metal_kernels(mx) else "portable"
    fused = {name: "fused" if _fused(mx, name) else "plain" for name in _PROBES}
    return {"decode_attention": custom, "gated_delta": custom, **fused}


def _fused(mx: Any, name: str) -> bool:
    if mode() == "plain":
        ok = False
    elif metal_kernels(mx):
        ok = True
    else:
        if name not in _fused_ok:
            try:
                _fused_ok[name] = bool(_PROBES[name](mx))
            except Exception:  # noqa: BLE001 - any failure means use the plain op.
                _fused_ok[name] = False
        ok = _fused_ok[name]
    if name == "compile":
        _set_compile(mx, ok)
    return ok


def _set_compile(mx: Any, ok: bool) -> None:
    """Turn ``mx.compile`` off when it can't be used, and back on only if this
    module was what turned it off."""
    global _compile_disabled
    if not ok and not _compile_disabled:
        mx.disable_compile()
        _compile_disabled = True
    elif ok and _compile_disabled:
        mx.enable_compile()
        _compile_disabled = False


# --- RoPE ---------------------------------------------------------------------


def rope(
    mx: Any,
    x: Any,
    dims: int,
    *,
    base: float | None,
    freqs: Any,
    offset: Any,
) -> Any:
    """Rotate ``[batch, heads, tokens, head_dim]`` (non-traditional layout).

    ``offset`` is the first position, an int or one per batch row; ``freqs``,
    when given, divides the positions instead of powers of ``base``.
    """
    if _fused(mx, "rope"):
        return mx.fast.rope(
            x,
            dims,
            traditional=False,
            base=base,
            scale=1.0,
            offset=offset,
            freqs=freqs,
        )
    return _rope_plain(mx, x, dims, base=base, freqs=freqs, offset=offset)


def _rope_plain(mx, x, dims, *, base, freqs, offset):
    f32 = mx.float32
    half = dims // 2
    if freqs is None:
        inverse = base ** (-mx.arange(0, half, dtype=f32) * 2 / dims)
    else:
        inverse = 1.0 / freqs.astype(f32)
    steps = mx.arange(int(x.shape[-2]), dtype=f32)
    if isinstance(offset, int):
        angles = ((steps + offset)[:, None] * inverse)[None, None]
    else:
        positions = offset.astype(f32)[:, None] + steps[None, :]
        angles = (positions[..., None] * inverse)[:, None]
    cos, sin = mx.cos(angles), mx.sin(angles)
    values = x.astype(f32)
    first, second = values[..., :half], values[..., half:dims]
    rotated = [first * cos - second * sin, second * cos + first * sin]
    return mx.concatenate([*rotated, values[..., dims:]], axis=-1).astype(x.dtype)


# --- RMS norm -----------------------------------------------------------------


def rms_norm(mx: Any, x: Any, weight: Any, eps: float) -> Any:
    if _fused(mx, "rms_norm"):
        return mx.fast.rms_norm(x, weight, eps)
    return _rms_norm_plain(mx, x, weight, eps)


def _rms_norm_plain(mx, x, weight, eps):
    values = x.astype(mx.float32)
    out = values * mx.rsqrt(mx.mean(values * values, axis=-1, keepdims=True) + eps)
    if weight is not None:
        out = out * weight.astype(mx.float32)
    return out.astype(x.dtype)


# --- attention ----------------------------------------------------------------


def attention(
    mx: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    scale: float,
    mask: Any = None,
    softcap: float = 0.0,
) -> Any:
    """``[batch, heads, queries, dim]`` attention; k and v may have fewer
    heads. ``mask`` is "causal" (aligned to the newest key), a boolean array
    (True attends) or None."""
    if not softcap and _fused(mx, "attention"):
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    return _attention_plain(mx, q, k, v, scale=scale, mask=mask, softcap=softcap)


def _attention_plain(mx, q, k, v, *, scale, mask, softcap=0.0):
    f32 = mx.float32
    groups = int(q.shape[1]) // int(k.shape[1])
    if groups > 1:
        k = mx.repeat(k, groups, axis=1)
        v = mx.repeat(v, groups, axis=1)
    scores = (q.astype(f32) * scale) @ k.astype(f32).swapaxes(-1, -2)
    if softcap:
        scores = mx.tanh(scores / softcap) * softcap
    if isinstance(mask, str):
        queries, keys = int(q.shape[2]), int(k.shape[2])
        rows = mx.arange(queries)[:, None] + (keys - queries)
        mask = mx.arange(keys)[None, :] <= rows
    if mask is not None:
        scores = mx.where(mask, scores, -mx.inf)
    return (mx.softmax(scores, axis=-1) @ v.astype(f32)).astype(q.dtype)


# --- paged decode attention ---------------------------------------------------


def decode_rows(mx: Any, tables: Any, lengths: Any, *, page_size: int, span: int):
    """Cache rows of each sequence's last ``span`` keys, right-aligned so the
    newest key is last, and which of them exist. ``tables`` is
    ``[batch, pages]`` (unused or released pages are negative) and
    ``lengths`` counts each sequence's keys."""
    positions = lengths[:, None] - span + mx.arange(span)[None, :]
    valid = positions >= 0
    clipped = mx.maximum(positions, 0)
    pages = mx.take_along_axis(tables, clipped // page_size, axis=1)
    valid = valid & (pages >= 0)
    rows = mx.where(valid, pages * page_size + clipped % page_size, 0)
    return rows.astype(mx.int32), valid


# --- gated delta rule ---------------------------------------------------------


def gated_delta(
    mx: Any, q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any
) -> tuple[Any, Any]:
    """Qwen3.5's gated delta rule over ``[batch, steps]`` tokens.

    Same contract as ``metal_kernels.gated_delta``: ``q``/``k`` are
    ``[batch, steps, k_heads, k_dim]``, ``v`` is ``[batch, steps, v_heads,
    v_dim]``, ``g`` (decay, in (0, 1]) and ``beta`` are ``[batch, steps,
    v_heads]``, ``state`` is float32 ``[batch, v_heads, v_dim, k_dim]``.

    Per token: ``S = g S; u = beta (v - S k); S += u k^T; out = S q``. One
    token is done directly. Longer runs go CHUNK tokens at a time: within a
    chunk the writes ``u`` solve a lower-triangular system that only the
    chunk's own keys and decays enter, so every chunk's system is solved at
    once with batched matmuls, and only applying the incoming state is
    sequential across chunks.
    """
    if metal_kernels(mx):
        from .metal_kernels import gated_delta as kernel

        return kernel(mx, q, k, v, g, beta, state)
    f32 = mx.float32
    batch, steps, k_heads, _ = (int(n) for n in q.shape)
    v_heads = int(v.shape[2])
    if v_heads % k_heads:
        raise ValueError("value heads must be a whole multiple of key heads")
    repeat = v_heads // k_heads
    # Head-major float32: [batch, v_heads, steps, dim].
    q = mx.repeat(q, repeat, axis=2).astype(f32).transpose(0, 2, 1, 3)
    k = mx.repeat(k, repeat, axis=2).astype(f32).transpose(0, 2, 1, 3)
    values = v.astype(f32).transpose(0, 2, 1, 3)
    g = g.astype(f32).transpose(0, 2, 1)
    beta = beta.astype(f32).transpose(0, 2, 1)
    if steps == 1:
        s = state * g[..., None]
        memory = (s @ k.swapaxes(-1, -2))[..., 0]
        write = (values[:, :, 0] - memory) * beta
        s = s + write[..., None] * k
        out = (s @ q.swapaxes(-1, -2))[..., 0][:, :, None]
        return out.transpose(0, 2, 1, 3).astype(v.dtype), s

    pad = -steps % CHUNK
    if pad:
        # Padding tokens write nothing (beta 0) and don't decay (g 1).
        widths = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, values = (mx.pad(a, widths) for a in (q, k, values))
        g = mx.pad(g, [(0, 0), (0, 0), (0, pad)], constant_values=1.0)
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
    chunks = (steps + pad) // CHUNK

    def blocks(a):
        return a.reshape(batch, v_heads, chunks, CHUNK, *a.shape[3:])

    q, k, values, beta = blocks(q), blocks(k), blocks(values), blocks(beta)
    log_decay = mx.cumsum(blocks(mx.log(mx.maximum(g, 1e-30))), axis=-1)
    decay = mx.exp(log_decay)  # From the chunk start through each token.
    gaps = log_decay[..., :, None] - log_decay[..., None, :]
    causal = mx.arange(CHUNK)[:, None] >= mx.arange(CHUNK)[None, :]
    between = mx.exp(mx.where(causal, gaps, -mx.inf))  # [t, s], s <= t
    strict = between * (1 - mx.eye(CHUNK, dtype=f32))
    # (I + L) u = beta (v - decay S k), L[t, s] = beta_t gap(t, s) k_t.k_s.
    lower = beta[..., None] * strict * (k @ k.swapaxes(-1, -2))
    inverse = _inverse_unit_lower(mx, lower)
    from_values = inverse @ (beta[..., None] * values)
    from_state = inverse @ ((beta * decay)[..., None] * k)
    attend = between * (q @ k.swapaxes(-1, -2))
    tail = mx.exp(log_decay[..., -1:] - log_decay)  # To the chunk end.
    s = state
    outputs = []
    for c in range(chunks):
        write = from_values[:, :, c] - from_state[:, :, c] @ s.swapaxes(-1, -2)
        outputs.append(
            decay[:, :, c, :, None] * (q[:, :, c] @ s.swapaxes(-1, -2))
            + attend[:, :, c] @ write
        )
        s = (
            decay[:, :, c, -1, None, None] * s
            + (write * tail[:, :, c, :, None]).swapaxes(-1, -2) @ k[:, :, c]
        )
    out = mx.concatenate(outputs, axis=2)[:, :, :steps]
    return out.transpose(0, 2, 1, 3).astype(v.dtype), s


def _inverse_unit_lower(mx: Any, lower: Any) -> Any:
    """``(I + lower)^-1`` for strictly lower-triangular ``[..., n, n]`` (n a
    power of two), by doubling the inverted diagonal blocks:
    ``[[A, 0], [X, D]]^-1 = [[A^-1, 0], [-D^-1 X A^-1, D^-1]]``.

    Every level holds exact inverses, which stay small here. The series
    ``I - L + L^2 - ...`` is exact too, but with correlated keys its
    middle powers overflow float32 before they cancel."""
    size = int(lower.shape[-1])
    lead = tuple(int(n) for n in lower.shape[:-2])
    axis = len(lead)
    blocks = mx.ones((*lead, size, 1, 1), dtype=lower.dtype)
    width = 1
    while width < size:
        pairs = size // (2 * width)
        grid = lower.reshape(*lead, pairs, 2, width, pairs, 2, width)
        # Each pair's corner block: rows from its second half, columns from
        # its first.
        diagonal = mx.diagonal(grid, axis1=axis, axis2=axis + 3)
        corner = mx.moveaxis(diagonal[..., 1, :, 0, :, :], -1, axis)
        first, second = blocks[..., 0::2, :, :], blocks[..., 1::2, :, :]
        top = mx.concatenate([first, mx.zeros_like(first)], axis=-1)
        bottom = mx.concatenate([-(second @ corner @ first), second], axis=-1)
        blocks = mx.concatenate([top, bottom], axis=-2)
        width *= 2
    return blocks[..., 0, :, :]


# --- probes -------------------------------------------------------------------


def _close(mx: Any, a: Any, b: Any, tolerance: float) -> bool:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    mx.eval(a, b)
    return bool(mx.all(mx.abs(a - b) <= tolerance * (1 + mx.abs(b))).item())


def _probe_rope(mx: Any) -> bool:
    key = mx.random.key(0)
    freqs = mx.array([1.0, 3.0, 50.0, 400.0, 2000.0, 9000.0, mx.inf, mx.inf])
    for dtype in (mx.float16, mx.bfloat16):
        decode = mx.random.normal((3, 2, 1, 16), key=key).astype(dtype)
        offsets = mx.array([0, 5, 70000], mx.int32)
        prefill = mx.random.normal((1, 2, 5, 16), key=key).astype(dtype)
        for x, offset in ((decode, offsets), (prefill, 9)):
            for base, f in ((10000.0, None), (None, freqs)):
                fast = mx.fast.rope(
                    x,
                    16,
                    traditional=False,
                    base=base,
                    scale=1.0,
                    offset=offset,
                    freqs=f,
                )
                plain = _rope_plain(mx, x, 16, base=base, freqs=f, offset=offset)
                if not _close(mx, fast, plain, 2e-2):
                    return False
    return True


def _probe_rms_norm(mx: Any) -> bool:
    x = mx.random.normal((3, 2, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    weight = mx.random.normal((64,), key=mx.random.key(2)).astype(mx.bfloat16)
    return all(
        _close(mx, mx.fast.rms_norm(x, w, 1e-6), _rms_norm_plain(mx, x, w, 1e-6), 2e-2)
        for w in (weight, None)
    )


def _probe_attention(mx: Any) -> bool:
    key = mx.random.key(3)
    for dim in (128, 256, 512):
        q = mx.random.normal((2, 4, 3, dim), key=key).astype(mx.bfloat16)
        k = mx.random.normal((2, 2, 7, dim), key=key).astype(mx.bfloat16)
        v = mx.random.normal((2, 2, 7, dim), key=key).astype(mx.bfloat16)
        boolean = mx.random.uniform(shape=(2, 1, 3, 7), key=key) > 0.3
        boolean[..., -1] = True
        for mask in ("causal", boolean):
            fast = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=dim**-0.5, mask=mask
            )
            plain = _attention_plain(mx, q, k, v, scale=dim**-0.5, mask=mask)
            if not _close(mx, fast, plain, 3e-2):
                return False
    return True


def _probe_compile(mx: Any) -> bool:
    step = mx.compile(lambda a, b: mx.tanh(a) * b + 1.0)
    a = mx.random.normal((64,), key=mx.random.key(4)).astype(mx.bfloat16)
    b = mx.random.normal((64,), key=mx.random.key(5)).astype(mx.bfloat16)
    return _close(mx, step(a, b), mx.tanh(a) * b + 1.0, 2e-2)


_PROBES = {
    "rope": _probe_rope,
    "rms_norm": _probe_rms_norm,
    "attention": _probe_attention,
    "compile": _probe_compile,
}
