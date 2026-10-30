"""Custom Metal kernels for the MLX engine."""

from __future__ import annotations

import math
from functools import cache
from typing import Any

# Paged decode attention, split-K ("flash decoding"). Pass one runs a
# threadgroup per (sequence, kv head, split of the key range): its simdgroups
# serve the G query heads that share the kv head, R simdgroups per head taking
# alternate keys, so one key row feeds all G heads from cache. Each lane holds
# E consecutive elements of the head; FULL marks head dims that fill every
# lane, so the per-lane loops unroll without bounds checks. With one split the
# kernel normalizes and writes the output itself; otherwise each split writes
# an unnormalized partial with its running max and sum, and pass two merges
# them. Keys and values are read straight from the page arena.
_DECODE_BODY = r"""
    const uint b = threadgroup_position_in_grid.z;
    const uint kvh = threadgroup_position_in_grid.y;
    const uint split = threadgroup_position_in_grid.x;
    const uint splits = threadgroups_per_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint g = sg % G;
    const uint r = sg / G;
    const uint h = kvh * G + g;
    constexpr uint E = (D + 31) / 32;
    const uint base = lane * E;
#define IN_HEAD(i) (FULL || base + (i) < D)

    const int length = lengths[b];
    const int first = WINDOW > 0 ? max(0, length - WINDOW) : 0;
    const int per_split = (length - first + int(splits) - 1) / int(splits);
    const int begin = first + int(split) * per_split;
    const int end = min(length, begin + per_split);

    float q[E], o[E];
    auto query = queries + (b * HQ + h) * D + base;
    for (uint i = 0; i < E; ++i) {
        q[i] = IN_HEAD(i) ? float(query[i]) * params[0] : 0.0f;
        o[i] = 0.0f;
    }
    float m = -INFINITY, l = 0.0f;
    auto table = tables + b * tables_shape[1];
    for (int t = begin + int(r); t < end; t += R) {
        const int row = table[t / PAGE] * PAGE + t % PAGE;
        auto key = keys + (row * HKV + kvh) * D + base;
        auto value = values + (row * HKV + kvh) * D + base;
        float s = 0.0f;
        for (uint i = 0; i < E; ++i) if (IN_HEAD(i)) s += q[i] * float(key[i]);
        s = simd_sum(s);
        if (params[1] > 0.0f) s = metal::precise::tanh(s / params[1]) * params[1];
        const float next = max(m, s);
        const float factor = metal::fast::exp(m - next);
        const float p = metal::fast::exp(s - next);
        l = l * factor + p;
        for (uint i = 0; i < E; ++i) if (IN_HEAD(i)) o[i] = o[i] * factor + p * float(value[i]);
        m = next;
    }

    threadgroup float maxima[R * G];
    threadgroup float sums[G];
    threadgroup float acc[G * D];
    if (lane == 0) maxima[r * G + g] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float top = -INFINITY;
    for (uint i = 0; i < R; ++i) top = max(top, maxima[i * G + g]);
    const float weight = m == -INFINITY ? 0.0f : metal::fast::exp(m - top);
    for (uint step = 0; step < R; ++step) {
        if (r == step) {
            for (uint i = 0; i < E; ++i) {
                if (!IN_HEAD(i)) continue;
                const uint at = g * D + base + i;
                acc[at] = (step == 0 ? 0.0f : acc[at]) + o[i] * weight;
            }
            if (lane == 0) sums[g] = (step == 0 ? 0.0f : sums[g]) + l * weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r == 0) {
        WRITE
    }
"""

_SINGLE_WRITE = r"""
        const float inverse = sums[g] > 0.0f ? 1.0f / sums[g] : 0.0f;
        for (uint i = 0; i < E; ++i)
            if (IN_HEAD(i)) out[(b * HQ + h) * D + base + i] = T(acc[g * D + base + i] * inverse);
"""

_SPLIT_WRITE = r"""
        const uint slot = (b * HQ + h) * splits + split;
        for (uint i = 0; i < E; ++i)
            if (IN_HEAD(i)) partial_o[slot * D + base + i] = acc[g * D + base + i];
        if (lane == 0) {
            partial_m[slot] = top;
            partial_l[slot] = sums[g];
        }
"""

_DECODE_MERGE = r"""
    const uint bh = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint splits = partial_m_shape[2];
    constexpr uint E = (D + 31) / 32;
    const uint base = lane * E;
#define IN_HEAD(i) (FULL || base + (i) < D)
    float top = -INFINITY;
    for (uint s = 0; s < splits; ++s) top = max(top, partial_m[bh * splits + s]);
    float total = 0.0f, o[E];
    for (uint i = 0; i < E; ++i) o[i] = 0.0f;
    for (uint s = 0; s < splits; ++s) {
        const float mm = partial_m[bh * splits + s];
        if (mm == -INFINITY) continue;
        const float w = metal::fast::exp(mm - top);
        total += partial_l[bh * splits + s] * w;
        auto source = partial_o + (bh * splits + s) * D + base;
        for (uint i = 0; i < E; ++i) if (IN_HEAD(i)) o[i] += source[i] * w;
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    for (uint i = 0; i < E; ++i) if (IN_HEAD(i)) out[bh * D + base + i] = T(o[i] * inverse);
"""

KEYS_PER_SPLIT = 512
MAX_SPLITS = 128


@cache
def _kernels(mx: Any) -> tuple[Any, Any, Any]:
    inputs = ["queries", "keys", "values", "tables", "lengths", "params"]
    single = mx.fast.metal_kernel(
        name="instinct_paged_decode",
        input_names=inputs,
        output_names=["out"],
        source=_DECODE_BODY.replace("WRITE", _SINGLE_WRITE),
    )
    split = mx.fast.metal_kernel(
        name="instinct_paged_decode_split",
        input_names=inputs,
        output_names=["partial_o", "partial_m", "partial_l"],
        source=_DECODE_BODY.replace("WRITE", _SPLIT_WRITE),
    )
    merge = mx.fast.metal_kernel(
        name="instinct_paged_decode_merge",
        input_names=["partial_o", "partial_m", "partial_l"],
        output_names=["out"],
        source=_DECODE_MERGE,
    )
    return single, split, merge


def paged_decode_attention(
    mx: Any,
    queries: Any,
    keys: Any,
    values: Any,
    tables: Any,
    lengths: Any,
    *,
    page_size: int,
    longest: int,
    scale: float,
    window: int = 0,
    softcap: float = 0.0,
) -> Any:
    """Attention for one new token per sequence over its paged keys.

    ``queries`` is ``[batch, q_heads, head_dim]``; ``keys`` and ``values`` are
    a layer's whole arena, ``[rows, kv_heads, head_dim]``; ``tables`` maps
    each sequence's logical pages to physical ones and ``lengths`` counts its
    keys, the new one included. ``longest`` (the largest length) sizes the
    split. Returns ``[batch, q_heads, head_dim]``.
    """
    batch, q_heads, head_dim = (int(n) for n in queries.shape)
    kv_heads = int(keys.shape[1])
    if q_heads % kv_heads:
        raise ValueError("query heads must be a multiple of kv heads")
    group = q_heads // kv_heads
    repeat = max(1, min(4, 32 // group))
    span = min(longest, window) if window else longest
    splits = max(1, min(MAX_SPLITS, math.ceil(span / KEYS_PER_SPLIT)))
    single, split, merge = _kernels(mx)
    template = [
        ("T", queries.dtype),
        ("D", head_dim),
        ("FULL", head_dim % 32 == 0),
        ("HQ", q_heads),
        ("HKV", kv_heads),
        ("G", group),
        ("R", repeat),
        ("PAGE", page_size),
        ("WINDOW", window),
    ]
    inputs = [
        queries,
        keys,
        values,
        tables,
        lengths,
        mx.array([scale, softcap], mx.float32),
    ]
    grid = (splits * 32 * group * repeat, kv_heads, batch)
    threads = (32 * group * repeat, 1, 1)
    if splits == 1:
        (out,) = single(
            inputs=inputs,
            template=template,
            grid=grid,
            threadgroup=threads,
            output_shapes=[(batch, q_heads, head_dim)],
            output_dtypes=[queries.dtype],
        )
        return out
    partial_o, partial_m, partial_l = split(
        inputs=inputs,
        template=template,
        grid=grid,
        threadgroup=threads,
        output_shapes=[
            (batch, q_heads, splits, head_dim),
            (batch, q_heads, splits),
            (batch, q_heads, splits),
        ],
        output_dtypes=[mx.float32] * 3,
    )
    (out,) = merge(
        inputs=[partial_o, partial_m, partial_l],
        template=[("T", queries.dtype), ("D", head_dim), ("FULL", head_dim % 32 == 0)],
        grid=(batch * q_heads * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(batch, q_heads, head_dim)],
        output_dtypes=[queries.dtype],
    )
    return out


# Gated delta rule (Qwen3.5's linear attention), one recurrence step per
# token. A threadgroup row of 32 lanes owns one value row of one head's state
# matrix, D_K / 32 entries per lane, and walks the sequence's tokens in order:
# decay the row, read the key's memory, correct it toward the value, then read
# out with the query. Sequences in a batch are independent rows of the grid.
_GATED_DELTA = r"""
    const uint n = thread_position_in_grid.z;
    const uint b = n / HV;
    const uint hv = n % HV;
    const uint hk = hv / (HV / HK);
    const uint dv = thread_position_in_grid.y;
    const uint lane = thread_position_in_threadgroup.x;
    constexpr uint E = DK / 32;
    const uint steps = q_shape[1];

    auto q_ = q + (b * steps * HK + hk) * DK + lane * E;
    auto k_ = k + (b * steps * HK + hk) * DK + lane * E;
    auto v_ = v + (b * steps * HV + hv) * DV + dv;
    auto g_ = g + b * steps * HV + hv;
    auto beta_ = beta + b * steps * HV + hv;
    auto y_ = y + (b * steps * HV + hv) * DV + dv;
    auto s_in = state_in + ((b * HV + hv) * DV + dv) * DK + lane * E;
    auto s_out = state_out + ((b * HV + hv) * DV + dv) * DK + lane * E;

    float s[E];
    for (uint i = 0; i < E; ++i) s[i] = float(s_in[i]);
    for (uint t = 0; t < steps; ++t) {
        const float decay = float(g_[t * HV]);
        float memory = 0.0f;
        for (uint i = 0; i < E; ++i) {
            s[i] *= decay;
            memory += s[i] * float(k_[t * HK * DK + i]);
        }
        memory = simd_sum(memory);
        const float delta = (float(v_[t * HV * DV]) - memory) * float(beta_[t * HV]);
        float out = 0.0f;
        for (uint i = 0; i < E; ++i) {
            s[i] += float(k_[t * HK * DK + i]) * delta;
            out += s[i] * float(q_[t * HK * DK + i]);
        }
        out = simd_sum(out);
        if (lane == 0) y_[t * HV * DV] = T(out);
    }
    for (uint i = 0; i < E; ++i) s_out[i] = s[i];
"""


@cache
def _gated_delta_kernel(mx: Any) -> Any:
    return mx.fast.metal_kernel(
        name="instinct_gated_delta",
        input_names=["q", "k", "v", "g", "beta", "state_in"],
        output_names=["y", "state_out"],
        source=_GATED_DELTA,
    )


def gated_delta(
    mx: Any, q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any
) -> tuple[Any, Any]:
    """Run the gated delta rule over ``[batch, steps]`` tokens.

    ``q``/``k`` are ``[batch, steps, k_heads, k_dim]``, ``v`` is
    ``[batch, steps, v_heads, v_dim]``, ``g`` (decay) and ``beta`` (write
    strength) are ``[batch, steps, v_heads]`` and ``state`` is the float32
    ``[batch, v_heads, v_dim, k_dim]`` memory. Returns the outputs, shaped
    like ``v``, and the new state.
    """
    batch, _, k_heads, k_dim = (int(n) for n in q.shape)
    v_heads, v_dim = int(v.shape[2]), int(v.shape[3])
    if k_dim % 32 or v_heads % k_heads:
        raise ValueError(
            "gated delta needs a key dim divisible by 32 and whole head groups"
        )
    return _gated_delta_kernel(mx)(
        inputs=[q, k, v, g, beta, state],
        template=[
            ("T", v.dtype),
            ("DK", k_dim),
            ("DV", v_dim),
            ("HK", k_heads),
            ("HV", v_heads),
        ],
        grid=(32, v_dim, batch * v_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[tuple(v.shape), tuple(state.shape)],
        output_dtypes=[v.dtype, mx.float32],
    )
