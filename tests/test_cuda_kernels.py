"""The CUDA kernels, without an NVIDIA GPU.

- Their source runs on the CPU under ``cuda_emulator`` (one OS thread per
  CUDA thread) and is compared with float64 references, at the shapes Gemma 4
  and Qwen3.5 use.
- Their sources are checked to be complete CUDA: no Metal left, every
  placeholder filled, and shape arguments only where MLX will pass them.

``tools/nvidia/compile_kernels.py`` compiles the same sources with NVRTC, and
``test_cuda_dispatch.py`` checks how the engine chooses them.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import cuda_emulator as emu
import numpy as np
import pytest
from instinct.backends import cuda_kernels as ck

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compile_kernels", ROOT / "tools/nvidia/compile_kernels.py"
)
compile_kernels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compile_kernels)

needs_compiler = pytest.mark.skipif(
    emu.compiler() is None, reason="needs clang++ or g++"
)


def _source(name, source, inputs, outputs, template):
    text = compile_kernels.build_kernel(name, "", source, inputs, outputs, template)
    return text.replace(compile_kernels.DEFAULT_HEADER, "")


def _reference_decode(
    q, keys, values, tables, lengths, page, scale, window, softcap, scales
):
    batch, q_heads, _ = q.shape
    group = q_heads // keys.shape[1]
    out = np.zeros(q.shape)
    for b in range(batch):
        length = lengths[b]
        first = max(0, length - window) if window else 0
        positions = np.arange(first, length)
        rows = tables[b][positions // page] * page + positions % page
        for h in range(q_heads):
            kv = h // group
            k = keys[rows, kv].astype(np.float64)
            v = values[rows, kv].astype(np.float64)
            if scales is not None:
                k = k * scales[0][rows, kv, None]
                v = v * scales[1][rows, kv, None]
            s = (q[b, h].astype(np.float64) * scale) @ k.T
            if softcap:
                s = np.tanh(s / softcap) * softcap
            p = np.exp(s - s.max())
            out[b, h] = (p / p.sum()) @ v
    return out


# (dtype, K/V storage, head dim, query heads, kv heads, window, softcap)
DECODE_SHAPES = {
    "qwen-8bit": ("bfloat16", "int8", 256, 16, 4, 0, 0.0),
    "gemma-sliding": ("bfloat16", "bfloat16", 256, 16, 8, 1024, 0.0),
    "gemma-full": ("bfloat16", "bfloat16", 512, 16, 2, 0, 0.0),
    "odd-head-softcap": ("float16", "float16", 80, 6, 3, 0, 30.0),
}


@needs_compiler
@pytest.mark.parametrize("shape", list(DECODE_SHAPES))
def test_decode_kernels_match_attention_over_the_pages(tmp_path, shape):
    dtype, storage, dim, q_heads, kv_heads, window, softcap = DECODE_SHAPES[shape]
    quantized = storage == "int8"
    page = 64
    group, repeat, _ = ck.decode_layout(q_heads, kv_heads, 1, window)
    threads = 32 * group * repeat
    template = [
        ("T", dtype),
        ("D", dim),
        ("FULL", dim % 32 == 0),
        ("HQ", q_heads),
        ("HKV", kv_heads),
        ("G", group),
        ("R", repeat),
        ("PAGE", page),
        ("WINDOW", window),
    ]
    inputs = [
        ("queries", dtype),
        ("keys", storage),
        ("values", storage),
        ("tables", "int32"),
        ("lengths", "int32"),
        ("params", "float32"),
    ] + ([("key_scales", "float16"), ("value_scales", "float16")] if quantized else [])
    sources = [
        _source(
            "single",
            ck.decode_source(quantized, ck.SINGLE_WRITE),
            inputs,
            [("out", dtype)],
            template,
        ),
        _source(
            "split",
            ck.decode_source(quantized, ck.SPLIT_WRITE),
            inputs,
            [
                ("partial_o", "float32"),
                ("partial_m", "float32"),
                ("partial_l", "float32"),
            ],
            template,
        ),
        _source(
            "merge",
            ck.DECODE_MERGE,
            [
                ("partial_o", "float32"),
                ("partial_m", "float32"),
                ("partial_l", "float32"),
            ],
            [("out", dtype)],
            [("T", dtype), ("D", dim), ("FULL", dim % 32 == 0)],
        ),
    ]
    rng = np.random.default_rng(len(shape))
    long = (window or ck.KEYS_PER_SPLIT) + ck.KEYS_PER_SPLIT // 2 + 3
    for case, lengths in enumerate(([300, 5], [long, 37])):
        pages = [math.ceil(n / page) for n in lengths]
        total = sum(pages) + 1
        order = rng.permutation(total)
        tables = np.zeros((2, max(pages)), np.int32)
        tables[0, : pages[0]] = order[: pages[0]]
        tables[1, : pages[1]] = order[pages[0] : pages[0] + pages[1]]
        rows = total * page

        def floats(shape_, kind):
            values = rng.normal(size=shape_).astype(np.float32)
            if kind == "bfloat16":
                bits = emu.to_bf16(values)
                return bits, emu.from_bf16(bits).reshape(shape_)
            values = values.astype(np.float16)
            return values, values.astype(np.float32)

        q_raw, q = floats((2, q_heads, dim), dtype)
        scales = None
        if quantized:
            k_raw = rng.integers(-127, 128, (rows, kv_heads, dim)).astype(np.int8)
            v_raw = rng.integers(-127, 128, (rows, kv_heads, dim)).astype(np.int8)
            ks = rng.uniform(0.002, 0.02, (rows, kv_heads)).astype(np.float16)
            vs = rng.uniform(0.002, 0.02, (rows, kv_heads)).astype(np.float16)
            keys, values, scales = (
                k_raw,
                v_raw,
                (ks.astype(np.float32), vs.astype(np.float32)),
            )
        else:
            k_raw, keys = floats((rows, kv_heads, dim), storage)
            v_raw, values = floats((rows, kv_heads, dim), storage)
        scale = dim**-0.5
        buffers = {
            "queries": emu.Buffer(dtype, q_raw),
            "keys": emu.Buffer(storage, k_raw),
            "values": emu.Buffer(storage, v_raw),
            "tables": emu.Buffer("int32", tables),
            "lengths": emu.Buffer("int32", np.array(lengths, np.int32)),
            "params": emu.Buffer("float32", np.array([scale, softcap], np.float32)),
        }
        if quantized:
            buffers["key_scales"] = emu.Buffer("float16", ks)
            buffers["value_scales"] = emu.Buffer("float16", vs)
        names = [name for name, _ in inputs]
        _, _, splits = ck.decode_layout(q_heads, kv_heads, max(lengths), window)
        grid = (splits * threads, kv_heads, 2)
        shape_arg = f"shape:2,{max(pages)}"
        with_shape = names[:4] + [shape_arg] + names[4:]
        if splits == 1:
            buffers["out"] = emu.Buffer(dtype, size=2 * q_heads * dim)
            launches = [
                emu.Launch("single", grid, (threads, 1, 1), with_shape + ["out"])
            ]
        else:
            slots = 2 * q_heads * splits
            buffers["partial_o"] = emu.Buffer("float32", size=slots * dim)
            buffers["partial_m"] = emu.Buffer("float32", size=slots)
            buffers["partial_l"] = emu.Buffer("float32", size=slots)
            buffers["out"] = emu.Buffer(dtype, size=2 * q_heads * dim)
            launches = [
                emu.Launch(
                    "split",
                    grid,
                    (threads, 1, 1),
                    with_shape + ["partial_o", "partial_m", "partial_l"],
                ),
                emu.Launch(
                    "merge",
                    (2 * q_heads * 32, 1, 1),
                    (32, 1, 1),
                    [
                        "partial_o",
                        "partial_m",
                        f"shape:2,{q_heads},{splits}",
                        "partial_l",
                        "out",
                    ],
                ),
            ]
        work = tmp_path / str(case)
        work.mkdir()
        raw = emu.emulate(ck.HEADER, sources, buffers, launches, work)["out"]
        got = emu.from_bf16(raw) if dtype == "bfloat16" else raw.astype(np.float32)
        expected = _reference_decode(
            q, keys, values, tables, lengths, page, scale, window, softcap, scales
        )
        np.testing.assert_allclose(
            got.reshape(expected.shape), expected, rtol=2e-2, atol=2e-2
        )


@needs_compiler
def test_gated_delta_kernel_matches_the_recurrence(tmp_path):
    batch, k_heads, v_heads, k_dim, v_dim = 2, 2, 4, 128, 16
    source = _source(
        "gated_delta",
        ck.GATED_DELTA,
        [
            ("q", "bfloat16"),
            ("k", "bfloat16"),
            ("v", "bfloat16"),
            ("g", "float32"),
            ("beta", "float32"),
            ("state_in", "float32"),
        ],
        [("y", "bfloat16"), ("state_out", "float32")],
        [
            ("T", "bfloat16"),
            ("DK", k_dim),
            ("DV", v_dim),
            ("HK", k_heads),
            ("HV", v_heads),
        ],
    )
    rng = np.random.default_rng(5)
    for steps in (1, 9):
        k = rng.normal(size=(batch, steps, k_heads, k_dim))
        k /= np.linalg.norm(k, axis=-1, keepdims=True)
        q = rng.normal(size=(batch, steps, k_heads, k_dim)) / k_dim**0.5
        v = rng.normal(size=(batch, steps, v_heads, v_dim))
        q_bits, k_bits, v_bits = (emu.to_bf16(a.astype(np.float32)) for a in (q, k, v))
        q, k, v = (emu.from_bf16(b).reshape(b.shape) for b in (q_bits, k_bits, v_bits))
        g = rng.uniform(0.6, 1.0, (batch, steps, v_heads)).astype(np.float32)
        beta = rng.uniform(0, 1, (batch, steps, v_heads)).astype(np.float32)
        state = (0.1 * rng.normal(size=(batch, v_heads, v_dim, k_dim))).astype(
            np.float32
        )
        buffers = {
            "q": emu.Buffer("bfloat16", q_bits),
            "k": emu.Buffer("bfloat16", k_bits),
            "v": emu.Buffer("bfloat16", v_bits),
            "g": emu.Buffer("float32", g),
            "beta": emu.Buffer("float32", beta),
            "state_in": emu.Buffer("float32", state),
            "y": emu.Buffer("bfloat16", size=v.size),
            "state_out": emu.Buffer("float32", size=state.size),
        }
        launch = emu.Launch(
            "gated_delta",
            (32, v_dim, batch * v_heads),
            (32, 4, 1),
            [
                "q",
                f"shape:{batch},{steps},{k_heads},{k_dim}",
                "k",
                "v",
                "g",
                "beta",
                "state_in",
                "y",
                "state_out",
            ],
        )
        work = tmp_path / str(steps)
        work.mkdir()
        out = emu.emulate(ck.HEADER, [source], buffers, [launch], work)
        # float64 recurrence: S = g S; u = beta (v - S k); S += u k^T; y = S q.
        s = state.astype(np.float64).copy()
        y = np.zeros(v.shape)
        for t in range(steps):
            for h in range(v_heads):
                kh = h // (v_heads // k_heads)
                s[:, h] *= g[:, t, h, None, None]
                memory = np.einsum("bvk,bk->bv", s[:, h], k[:, t, kh])
                write = (v[:, t, h] - memory) * beta[:, t, h, None]
                s[:, h] += write[:, :, None] * k[:, t, kh, None, :]
                y[:, t, h] = np.einsum("bvk,bk->bv", s[:, h], q[:, t, kh])
        np.testing.assert_allclose(
            emu.from_bf16(out["y"]).reshape(y.shape), y, rtol=2e-2, atol=2e-2
        )
        np.testing.assert_allclose(
            out["state_out"].reshape(s.shape), s, rtol=1e-4, atol=1e-4
        )


def test_sources_are_complete_cuda():
    sources = [source for _, _, source in compile_kernels.kernels()]
    assert len(sources) == 17
    for source in sources:
        for leftover in (
            "simd_sum",
            "threadgroup",
            "metal::",
            "KEY_SCALE",
            "VALUE_SCALE",
            "WRITE\n",
        ):
            assert leftover not in source
        assert source.count("{") == source.count("}")
    # MLX passes a Shape argument for each input whose "<name>_shape" the
    # body mentions; the kernels index only these.
    for body, inputs, wanted in (
        (
            ck.decode_source(True, ck.SPLIT_WRITE),
            ck.DECODE_INPUTS + ck.QUANTIZED_INPUTS,
            {"tables"},
        ),
        (ck.DECODE_MERGE, ("partial_o", "partial_m", "partial_l"), {"partial_m"}),
        (ck.GATED_DELTA, ck.GATED_DELTA_INPUTS, {"q"}),
    ):
        mentioned = {name for name in inputs if name + "_shape" in body}
        assert mentioned == wanted
        assert not any(
            name + suffix in body for name in inputs for suffix in ("_strides", "_ndim")
        )
