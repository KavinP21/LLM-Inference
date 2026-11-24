"""Custom CUDA kernels for the MLX engine on NVIDIA GPUs.

These are the CUDA counterparts of ``metal_kernels``: paged decode attention
(split-K "flash decoding" read straight from the page arena) and Qwen3.5's
gated delta rule. They're compiled at run time through ``mx.fast.cuda_kernel``
(NVRTC), so they need no CUDA toolkit and no build step.

Each kernel is compiled per shape (head dim, head counts, page size, window,
K/V width). The first time a shape is used, the kernel runs once on small
random inputs with that exact shape and is compared with the plain-MLX
version in ``portable``. A kernel that fails to compile, fails to launch or
disagrees is never used for that shape again: the caller gets ``None`` and
falls back to ``portable``, and ``fallbacks()`` says why. So a GPU, driver or
MLX version these kernels don't suit costs speed, not correctness.
"""

from __future__ import annotations

import logging
import math
from functools import cache
from typing import Any

log = logging.getLogger(__name__)

# Shared by every kernel: dtype conversions that don't depend on the CUDA
# headers' implicit operators, and a warp-wide sum.
HEADER = r"""
#define INSTINCT_NEG_INF __int_as_float(0xff800000)

__device__ __forceinline__ float instinct_float(float x) { return x; }
__device__ __forceinline__ float instinct_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float instinct_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float instinct_float(int8_t x) { return static_cast<float>(x); }

template <typename T> __device__ __forceinline__ T instinct_cast(float x);
template <> __device__ __forceinline__ float instinct_cast<float>(float x) { return x; }
template <> __device__ __forceinline__ __half instinct_cast<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 instinct_cast<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float instinct_warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
"""

# Paged decode attention, split-K. A block runs per (split of the key range,
# kv head, sequence): its warps serve the G query heads that share the kv
# head, R warps per head taking alternate keys, so each key row is read once
# for all G heads. Each lane holds E consecutive elements of the head; FULL
# marks head dims that fill every lane. With one split the block normalizes
# and writes the output itself; otherwise each split writes an unnormalized
# partial with its running max and sum, and the merge kernel combines them.
DECODE_BODY = r"""
    const int b = blockIdx.z;
    const int kvh = blockIdx.y;
    const int split = blockIdx.x;
    const int splits = gridDim.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int g = warp % G;
    const int r = warp / G;
    const int h = kvh * G + g;
    constexpr int E = (D + 31) / 32;
    const int base = lane * E;
#define IN_HEAD(i) (FULL || base + (i) < D)

    const int length = lengths[b];
    const int first = WINDOW > 0 ? max(0, length - WINDOW) : 0;
    const int per_split = (length - first + splits - 1) / splits;
    const int begin = first + split * per_split;
    const int end = min(length, begin + per_split);

    float q[E], o[E];
    const auto* query = queries + ((long long)b * HQ + h) * D + base;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        q[i] = IN_HEAD(i) ? instinct_float(query[i]) * params[0] : 0.0f;
        o[i] = 0.0f;
    }
    float m = INSTINCT_NEG_INF, l = 0.0f;
    const auto* table = tables + (long long)b * tables_shape[1];
    for (int t = begin + r; t < end; t += R) {
        const long long row = (long long)table[t / PAGE] * PAGE + t % PAGE;
        const auto* key = keys + (row * HKV + kvh) * D + base;
        const auto* value = values + (row * HKV + kvh) * D + base;
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < E; ++i) if (IN_HEAD(i)) s += q[i] * instinct_float(key[i]);
        s = instinct_warp_sum(s) KEY_SCALE;
        if (params[1] > 0.0f) s = tanhf(s / params[1]) * params[1];
        const float next = fmaxf(m, s);
        const float factor = __expf(m - next);
        const float p = __expf(s - next);
        l = l * factor + p;
        const float pv = p VALUE_SCALE;
#pragma unroll
        for (int i = 0; i < E; ++i)
            if (IN_HEAD(i)) o[i] = o[i] * factor + pv * instinct_float(value[i]);
        m = next;
    }

    __shared__ float maxima[R * G];
    __shared__ float sums[G];
    __shared__ float acc[G * D];
    if (lane == 0) maxima[r * G + g] = m;
    __syncthreads();
    float top = INSTINCT_NEG_INF;
    for (int i = 0; i < R; ++i) top = fmaxf(top, maxima[i * G + g]);
    const float weight = m == INSTINCT_NEG_INF ? 0.0f : __expf(m - top);
    for (int step = 0; step < R; ++step) {
        if (r == step) {
#pragma unroll
            for (int i = 0; i < E; ++i) {
                if (!IN_HEAD(i)) continue;
                const int at = g * D + base + i;
                acc[at] = (step == 0 ? 0.0f : acc[at]) + o[i] * weight;
            }
            if (lane == 0) sums[g] = (step == 0 ? 0.0f : sums[g]) + l * weight;
        }
        __syncthreads();
    }
    if (r == 0) {
        WRITE
    }
#undef IN_HEAD
"""

SINGLE_WRITE = r"""
        const float inverse = sums[g] > 0.0f ? 1.0f / sums[g] : 0.0f;
        for (int i = 0; i < E; ++i)
            if (IN_HEAD(i))
                out[((long long)b * HQ + h) * D + base + i] =
                    instinct_cast<T>(acc[g * D + base + i] * inverse);
"""

SPLIT_WRITE = r"""
        const long long slot = ((long long)b * HQ + h) * splits + split;
        for (int i = 0; i < E; ++i)
            if (IN_HEAD(i)) partial_o[slot * D + base + i] = acc[g * D + base + i];
        if (lane == 0) {
            partial_m[slot] = top;
            partial_l[slot] = sums[g];
        }
"""

DECODE_MERGE = r"""
    const long long bh = blockIdx.x;
    const int lane = threadIdx.x;
    const int splits = partial_m_shape[2];
    constexpr int E = (D + 31) / 32;
    const int base = lane * E;
#define IN_HEAD(i) (FULL || base + (i) < D)
    float top = INSTINCT_NEG_INF;
    for (int s = 0; s < splits; ++s) top = fmaxf(top, partial_m[bh * splits + s]);
    float total = 0.0f, o[E];
#pragma unroll
    for (int i = 0; i < E; ++i) o[i] = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float mm = partial_m[bh * splits + s];
        if (mm == INSTINCT_NEG_INF) continue;
        const float w = __expf(mm - top);
        total += partial_l[bh * splits + s] * w;
        const float* source = partial_o + (bh * splits + s) * D + base;
#pragma unroll
        for (int i = 0; i < E; ++i) if (IN_HEAD(i)) o[i] += source[i] * w;
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
    for (int i = 0; i < E; ++i)
        if (IN_HEAD(i)) out[bh * D + base + i] = instinct_cast<T>(o[i] * inverse);
#undef IN_HEAD
"""

KEYS_PER_SPLIT = 512
MAX_SPLITS = 128

DECODE_INPUTS = ("queries", "keys", "values", "tables", "lengths", "params")
QUANTIZED_INPUTS = ("key_scales", "value_scales")


def decode_source(quantized: bool, write: str) -> str:
    """The decode kernel's body, for 16-bit or 8-bit K/V."""
    body = DECODE_BODY
    if quantized:
        # 8-bit K/V: one scale per token and head, applied to the score and
        # to the probability that weights the value row.
        body = body.replace(
            "KEY_SCALE", "* instinct_float(key_scales[row * HKV + kvh])"
        ).replace("VALUE_SCALE", "* instinct_float(value_scales[row * HKV + kvh])")
    else:
        body = body.replace("KEY_SCALE", "").replace("VALUE_SCALE", "")
    return body.replace("WRITE", write)


@cache
def _decode_kernels(mx: Any, quantized: bool) -> tuple[Any, Any, Any]:
    inputs = list(DECODE_INPUTS) + (list(QUANTIZED_INPUTS) if quantized else [])
    suffix = "_q8" if quantized else ""
    single = mx.fast.cuda_kernel(
        name="instinct_paged_decode" + suffix,
        input_names=inputs,
        output_names=["out"],
        source=decode_source(quantized, SINGLE_WRITE),
        header=HEADER,
    )
    split = mx.fast.cuda_kernel(
        name="instinct_paged_decode_split" + suffix,
        input_names=inputs,
        output_names=["partial_o", "partial_m", "partial_l"],
        source=decode_source(quantized, SPLIT_WRITE),
        header=HEADER,
    )
    merge = mx.fast.cuda_kernel(
        name="instinct_paged_decode_merge",
        input_names=["partial_o", "partial_m", "partial_l"],
        output_names=["out"],
        source=DECODE_MERGE,
        header=HEADER,
    )
    return single, split, merge


def decode_layout(q_heads: int, kv_heads: int, longest: int, window: int):
    """Query heads per kv head, warps per query head and key-range splits."""
    if q_heads % kv_heads:
        raise ValueError("query heads must be a multiple of kv heads")
    group = q_heads // kv_heads
    if group > 32:
        raise ValueError("at most 32 query heads per kv head")
    repeat = max(1, min(4, 32 // group))
    span = min(longest, window) if window else longest
    splits = max(1, min(MAX_SPLITS, math.ceil(span / KEYS_PER_SPLIT)))
    return group, repeat, splits


def _run_decode(
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
    window: int,
    softcap: float,
    scales: tuple[Any, Any] | None,
) -> Any:
    batch, q_heads, head_dim = (int(n) for n in queries.shape)
    kv_heads = int(keys.shape[1])
    group, repeat, splits = decode_layout(q_heads, kv_heads, longest, window)
    single, split, merge = _decode_kernels(mx, scales is not None)
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
        *(scales or ()),
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
    scales: tuple[Any, Any] | None = None,
) -> Any | None:
    """Same contract as ``metal_kernels.paged_decode_attention``, or None
    when this shape's kernel failed its check (use the portable path)."""
    q_heads, head_dim = int(queries.shape[1]), int(queries.shape[2])
    kv_heads = int(keys.shape[1])
    key = (
        "decode_attention",
        str(queries.dtype),
        str(keys.dtype),
        head_dim,
        q_heads,
        kv_heads,
        page_size,
        window,
        softcap > 0,
    )
    check = lambda: _check_decode(  # noqa: E731
        mx,
        queries.dtype,
        keys.dtype,
        head_dim=head_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        page_size=page_size,
        window=window,
        softcap=softcap,
        quantized=scales is not None,
    )
    if not _verified(key, check):
        return None
    return _run_decode(
        mx,
        queries,
        keys,
        values,
        tables,
        lengths,
        page_size=page_size,
        longest=longest,
        scale=scale,
        window=window,
        softcap=softcap,
        scales=scales,
    )


# Gated delta rule (Qwen3.5's linear attention), one recurrence step per
# token. A warp owns one value row of one head's state matrix, D_K / 32
# entries per lane, and walks the sequence's tokens in order: decay the row,
# read the key's memory, correct it toward the value, then read out with the
# query. Sequences in a batch are independent rows of the grid.
GATED_DELTA = r"""
    const int n = blockIdx.z;
    const int b = n / HV;
    const int hv = n % HV;
    const int hk = hv / (HV / HK);
    const int dv = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    constexpr int E = DK / 32;
    const int steps = q_shape[1];
    if (dv >= DV) return;  // Whole warps only: a warp shares one dv.

    const auto* q_ = q + ((long long)b * steps * HK + hk) * DK + lane * E;
    const auto* k_ = k + ((long long)b * steps * HK + hk) * DK + lane * E;
    const auto* v_ = v + ((long long)b * steps * HV + hv) * DV + dv;
    const auto* g_ = g + (long long)b * steps * HV + hv;
    const auto* beta_ = beta + (long long)b * steps * HV + hv;
    auto* y_ = y + ((long long)b * steps * HV + hv) * DV + dv;
    const float* s_in = state_in + (((long long)b * HV + hv) * DV + dv) * DK + lane * E;
    float* s_out = state_out + (((long long)b * HV + hv) * DV + dv) * DK + lane * E;

    float s[E];
#pragma unroll
    for (int i = 0; i < E; ++i) s[i] = s_in[i];
    for (int t = 0; t < steps; ++t) {
        const float decay = instinct_float(g_[(long long)t * HV]);
        float memory = 0.0f;
#pragma unroll
        for (int i = 0; i < E; ++i) {
            s[i] *= decay;
            memory += s[i] * instinct_float(k_[(long long)t * HK * DK + i]);
        }
        memory = instinct_warp_sum(memory);
        const float delta = (instinct_float(v_[(long long)t * HV * DV]) - memory)
            * instinct_float(beta_[(long long)t * HV]);
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < E; ++i) {
            s[i] += instinct_float(k_[(long long)t * HK * DK + i]) * delta;
            out += s[i] * instinct_float(q_[(long long)t * HK * DK + i]);
        }
        out = instinct_warp_sum(out);
        if (lane == 0) y_[(long long)t * HV * DV] = instinct_cast<T>(out);
    }
#pragma unroll
    for (int i = 0; i < E; ++i) s_out[i] = s[i];
"""

GATED_DELTA_INPUTS = ("q", "k", "v", "g", "beta", "state_in")


@cache
def _gated_delta_kernel(mx: Any) -> Any:
    return mx.fast.cuda_kernel(
        name="instinct_gated_delta",
        input_names=list(GATED_DELTA_INPUTS),
        output_names=["y", "state_out"],
        source=GATED_DELTA,
        header=HEADER,
    )


def _run_gated_delta(mx, q, k, v, g, beta, state):
    batch, _, k_heads, k_dim = (int(n) for n in q.shape)
    v_heads, v_dim = int(v.shape[2]), int(v.shape[3])
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


def gated_delta(
    mx: Any, q: Any, k: Any, v: Any, g: Any, beta: Any, state: Any
) -> tuple[Any, Any] | None:
    """Same contract as ``metal_kernels.gated_delta``, or None when the
    kernel doesn't fit this shape or failed its check."""
    k_heads, k_dim = int(q.shape[2]), int(q.shape[3])
    v_heads, v_dim = int(v.shape[2]), int(v.shape[3])
    if k_dim % 32 or v_heads % k_heads:
        return None
    key = (
        "gated_delta",
        str(q.dtype),
        str(v.dtype),
        str(g.dtype),
        k_dim,
        v_dim,
        k_heads,
        v_heads,
    )
    check = lambda: _check_gated_delta(  # noqa: E731
        mx,
        (q.dtype, v.dtype, g.dtype),
        k_dim=k_dim,
        v_dim=v_dim,
        k_heads=k_heads,
        v_heads=v_heads,
    )
    if not _verified(key, check):
        return None
    return _run_gated_delta(mx, q, k, v, g, beta, state)


# --- checks -------------------------------------------------------------------

_checked: dict[tuple, bool] = {}
_failures: dict[str, str] = {}


def fallbacks() -> dict[str, str]:
    """Kernels that failed a check, with why; they use the portable path."""
    return dict(_failures)


def reset() -> None:
    """Forget every check (for tests)."""
    _checked.clear()
    _failures.clear()


def _verified(key: tuple, check: Any) -> bool:
    if key not in _checked:
        try:
            ok = bool(check())
            reason = "" if ok else "disagreed with the portable version"
        except Exception as exc:  # noqa: BLE001 - any failure means fall back.
            ok, reason = False, f"{type(exc).__name__}: {exc}"[:300]
        _checked[key] = ok
        if not ok:
            name = key[0]
            _failures[name] = f"{reason} (shape {key[1:]})"
            log.warning(
                "CUDA kernel %s failed its check for %s: %s; using the portable "
                "version for that shape",
                name,
                key[1:],
                reason,
            )
    return _checked[key]


def _close(mx: Any, a: Any, b: Any, tolerance: float) -> bool:
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    mx.eval(a, b)
    finite = bool(mx.all(mx.isfinite(a)).item())
    return finite and bool(mx.all(mx.abs(a - b) <= tolerance * (1 + mx.abs(b))).item())


def _check_decode(
    mx: Any,
    dtype: Any,
    storage: Any,
    *,
    head_dim: int,
    q_heads: int,
    kv_heads: int,
    page_size: int,
    window: int,
    softcap: float,
    quantized: bool,
) -> bool:
    """Run the decode kernels at this shape on random pages against gathered
    plain attention: once with every sequence short (one split), once with
    one long enough to split (and to pass the window, if any)."""
    from . import portable

    long = (window or KEYS_PER_SPLIT) + KEYS_PER_SPLIT // 2 + 3
    for lengths in ([min(long, 300), 5], [long, 37]):
        pages = [math.ceil(n / page_size) for n in lengths]
        total = sum(pages) + 1  # One spare page nobody uses.
        rng = mx.random.key(len(lengths) + lengths[0])
        k1, k2, k3, k4 = mx.random.split(rng, 4)
        rows = total * page_size
        keys = mx.random.normal((rows, kv_heads, head_dim), key=k1)
        values = mx.random.normal((rows, kv_heads, head_dim), key=k2)
        queries = mx.random.normal((len(lengths), q_heads, head_dim), key=k3)
        queries = queries.astype(dtype)
        # Scatter each sequence's pages across the arena, in shuffled order.
        order = mx.random.permutation(total, key=k4).tolist()
        tables = []
        for count in pages:
            tables.append(order[:count] + [-1] * (max(pages) - count))
            order = order[count:]
        table = mx.array(tables, dtype=mx.int32)
        lengths_ = mx.array(lengths, dtype=mx.int32)
        if quantized:
            from ..paged_kv import quantize

            keys, key_scales = quantize(mx, keys.astype(dtype))
            values, value_scales = quantize(mx, values.astype(dtype))
            keys, values = keys.astype(storage), values.astype(storage)
            scales = (key_scales, value_scales)
            dense_keys = keys.astype(mx.float32) * key_scales[..., None]
            dense_values = values.astype(mx.float32) * value_scales[..., None]
        else:
            keys, values = keys.astype(storage), values.astype(storage)
            scales = None
            dense_keys, dense_values = keys, values
        scale = head_dim**-0.5
        out = _run_decode(
            mx,
            queries,
            keys,
            values,
            mx.maximum(table, 0),
            lengths_,
            page_size=page_size,
            longest=max(lengths),
            scale=scale,
            window=window,
            softcap=softcap,
            scales=scales,
        )
        span = min(max(lengths), window or max(lengths))
        positions, valid = portable.decode_rows(
            mx, table, lengths_, page_size=page_size, span=span
        )
        flat = positions.reshape(-1)
        batch = len(lengths)

        def dense(store):
            gathered = store[flat].astype(dtype)
            return gathered.reshape(batch, span, kv_heads, head_dim).transpose(
                0, 2, 1, 3
            )

        expected = portable._attention_plain(
            mx,
            queries[:, :, None, :],
            dense(dense_keys),
            dense(dense_values),
            scale=scale,
            mask=valid[:, None, None, :],
            softcap=softcap,
        )[:, :, 0, :]
        if not _close(mx, out, expected, 3e-2):
            return False
    return True


def _check_gated_delta(
    mx: Any,
    dtypes: tuple[Any, Any, Any],
    *,
    k_dim: int,
    v_dim: int,
    k_heads: int,
    v_heads: int,
) -> bool:
    """Run the recurrence at this shape against the plain version, for one
    token (decode) and a short run (prefill)."""
    from . import portable

    q_dtype, v_dtype, g_dtype = dtypes
    for steps in (1, 19):
        key = mx.random.key(steps)
        kq, kk, kv, kg, kb, ks = mx.random.split(key, 6)
        k = mx.random.normal((2, steps, k_heads, k_dim), key=kk)
        k = k * mx.rsqrt(mx.sum(k * k, axis=-1, keepdims=True))
        q = mx.random.normal((2, steps, k_heads, k_dim), key=kq) * k_dim**-0.5
        v = mx.random.normal((2, steps, v_heads, v_dim), key=kv)
        g = mx.random.uniform(0.6, 1.0, (2, steps, v_heads), key=kg)
        beta = mx.random.uniform(0.0, 1.0, (2, steps, v_heads), key=kb)
        state = 0.1 * mx.random.normal((2, v_heads, v_dim, k_dim), key=ks)
        args = (
            q.astype(q_dtype),
            k.astype(q_dtype),
            v.astype(v_dtype),
            g.astype(g_dtype),
            beta.astype(g_dtype),
            state.astype(mx.float32),
        )
        out, new_state = _run_gated_delta(mx, *args)
        expected, expected_state = portable._gated_delta_plain(mx, *args)
        if not (
            _close(mx, out, expected, 3e-2)
            and _close(mx, new_state, expected_state, 3e-2)
        ):
            return False
    return True
