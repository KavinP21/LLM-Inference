"""Run CUDA kernel source on the CPU, one OS thread per CUDA thread.

Enough of CUDA for Instinct's kernels (``backends/cuda_kernels.py``): block
and thread indices, ``__syncthreads``, ``__shfl_xor_sync``, ``__shared__``
arrays and the bf16/fp16 types. Blocks run one at a time, so ``__shared__``
can be a function-static array. It checks what the kernels compute, their
indexing, page tables, windows, split merging and recurrences, on machines
without an NVIDIA GPU. Speed and GPU resource limits are not modelled; the
NVRTC compile in ``tools/nvidia/compile_kernels.py`` covers those.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PRELUDE = r"""
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __grid_constant__
#define __shared__ static
#define MAX_NDIM 10
using Shape = std::array<int32_t, MAX_NDIM>;
namespace cooperative_groups {}

struct dim3 { unsigned x = 1, y = 1, z = 1; };
thread_local dim3 threadIdx, blockIdx;
dim3 blockDim, gridDim;
thread_local int linear_tid;

struct Barrier {
  std::mutex m; std::condition_variable cv; int count, waiting = 0, generation = 0;
  explicit Barrier(int n) : count(n) {}
  void wait() {
    std::unique_lock<std::mutex> lock(m);
    int gen = generation;
    if (++waiting == count) { waiting = 0; ++generation; cv.notify_all(); }
    else cv.wait(lock, [&] { return gen != generation; });
  }
};
static Barrier* block_barrier;
static std::vector<std::unique_ptr<Barrier>> warp_barriers;
static std::vector<float> lane_values;

inline void __syncthreads() { block_barrier->wait(); }
inline float __shfl_xor_sync(unsigned, float v, int offset) {
  int warp = linear_tid / 32, lane = linear_tid % 32;
  lane_values[linear_tid] = v;
  warp_barriers[warp]->wait();
  float r = lane_values[warp * 32 + (lane ^ offset)];
  warp_barriers[warp]->wait();
  return r;
}

struct __nv_bfloat16 { uint16_t bits; };
struct __half { uint16_t bits; };
inline float __bfloat162float(__nv_bfloat16 x) {
  uint32_t u = uint32_t(x.bits) << 16; float f; memcpy(&f, &u, 4); return f;
}
inline __nv_bfloat16 __float2bfloat16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  if (isnan(f)) return {0x7fc0};
  u += 0x7fffu + ((u >> 16) & 1u);
  return {uint16_t(u >> 16)};
}
inline float __half2float(__half h) {
  uint32_t sign = uint32_t(h.bits & 0x8000) << 16, exp = (h.bits >> 10) & 0x1f, man = h.bits & 0x3ff, u;
  if (exp == 0) {
    if (man == 0) u = sign;
    else { exp = 127 - 15 + 1; while (!(man & 0x400)) { man <<= 1; --exp; } man &= 0x3ff; u = sign | (exp << 23) | (man << 13); }
  } else if (exp == 31) u = sign | 0x7f800000u | (man << 13);
  else u = sign | ((exp + 127 - 15) << 23) | (man << 13);
  float f; memcpy(&f, &u, 4); return f;
}
inline __half __float2half(float f) {
  // Round to nearest even through double, which holds every half exactly.
  float a = fabsf(f); uint16_t sign = signbit(f) ? 0x8000 : 0;
  if (isnan(f)) return {uint16_t(sign | 0x7e00)};
  if (a >= 65520.0f) return {uint16_t(sign | 0x7c00)};
  int e; double m = frexp(double(a), &e);  // a = m * 2^e, m in [0.5, 1)
  if (a < 6.103515625e-05f) {  // subnormal: units of 2^-24
    double units = nearbyint(double(a) * 16777216.0);
    return {uint16_t(sign | uint16_t(units))};
  }
  double units = nearbyint(m * 2048.0);  // 11 significant bits
  if (units == 2048.0) { units = 1024.0; ++e; }
  uint16_t exp = uint16_t(e - 1 + 15);
  return {uint16_t(sign | (exp << 10) | (uint16_t(units) & 0x3ff))};
}
inline float __int_as_float(long long i) {
  uint32_t u = uint32_t(i); float f; memcpy(&f, &u, 4); return f;
}
inline float __expf(float x) { return expf(x); }
template <class A> inline A max(A a, A b) { return a > b ? a : b; }
template <class A> inline A min(A a, A b) { return a < b ? a : b; }

template <class F> void launch(dim3 grid, dim3 block, F kernel) {
  gridDim = grid; blockDim = block;
  int n = int(block.x * block.y * block.z);
  lane_values.assign(n, 0.0f);
  for (unsigned z = 0; z < grid.z; ++z)
  for (unsigned y = 0; y < grid.y; ++y)
  for (unsigned x = 0; x < grid.x; ++x) {
    Barrier whole(n);
    block_barrier = &whole;
    warp_barriers.clear();
    for (int w = 0; w * 32 < n; ++w)
      warp_barriers.emplace_back(new Barrier(std::min(32, n - w * 32)));
    std::vector<std::thread> threads;
    for (int t = 0; t < n; ++t) {
      threads.emplace_back([&, t] {
        linear_tid = t;
        threadIdx = {unsigned(t) % block.x, (unsigned(t) / block.x) % block.y,
                     unsigned(t) / (block.x * block.y)};
        blockIdx = {x, y, z};
        kernel();
      });
    }
    for (auto& thread : threads) thread.join();
  }
}

template <class T> std::vector<T> load(const char* path, size_t count) {
  std::vector<T> data(count);
  FILE* f = fopen(path, "rb");
  if (!f || fread(data.data(), sizeof(T), count, f) != count) { fprintf(stderr, "bad %s\n", path); exit(2); }
  fclose(f);
  return data;
}
template <class T> void save(const char* path, const std::vector<T>& data) {
  FILE* f = fopen(path, "wb");
  fwrite(data.data(), sizeof(T), data.size(), f);
  fclose(f);
}
"""

C_TYPES = {
    "bfloat16": "__nv_bfloat16",
    "float16": "__half",
    "float32": "float",
    "int32": "int32_t",
    "int8": "int8_t",
}
STORAGE = {
    "bfloat16": np.uint16,
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
    "int8": np.int8,
}


def to_bf16(values: np.ndarray) -> np.ndarray:
    """float32 values rounded to bf16 bits (nearest even)."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


@dataclass
class Buffer:
    dtype: str  # One of C_TYPES.
    data: np.ndarray | None = None  # Raw storage; None for an output.
    size: int = 0  # Elements, for an output.

    @property
    def count(self) -> int:
        return int(self.data.size) if self.data is not None else self.size


@dataclass
class Launch:
    kernel: str
    grid: tuple[int, int, int]  # Threads, as mx.fast.cuda_kernel takes them.
    threads: tuple[int, int, int]
    args: list[str]  # Buffer names, or "shape:<dims>" for a Shape argument.


def compiler() -> str | None:
    return shutil.which("clang++") or shutil.which("g++")


def emulate(
    header: str,
    sources: list[str],
    buffers: dict[str, Buffer],
    launches: list[Launch],
    workdir: Path,
) -> dict[str, np.ndarray]:
    """Compile ``sources`` (each from ``build_kernel`` without MLX's
    includes) with ``header``, run ``launches`` in order and return every
    output buffer's raw storage."""
    body = [PRELUDE, header, *sources, "int main() {"]
    for name, buffer in buffers.items():
        c_type = C_TYPES[buffer.dtype]
        if buffer.data is not None:
            path = workdir / f"{name}.bin"
            np.ascontiguousarray(buffer.data).astype(
                STORAGE[buffer.dtype], copy=False
            ).tofile(path)
            body.append(f'  auto {name} = load<{c_type}>("{path}", {buffer.count});')
        else:
            body.append(f"  std::vector<{c_type}> {name}({buffer.count});")
    for launch in launches:
        blocks = [-(-g // t) for g, t in zip(launch.grid, launch.threads, strict=True)]
        threads = [min(t, g) for g, t in zip(launch.grid, launch.threads, strict=True)]
        args = []
        for arg in launch.args:
            if arg.startswith("shape:"):
                dims = arg.removeprefix("shape:").split(",")
                args.append("Shape{" + ", ".join(dims) + "}")
            else:
                args.append(f"{arg}.data()")
        call = f"mlx::core::cu::{launch.kernel}({', '.join(args)});"
        body.append(
            f"  launch(dim3{{{blocks[0]}, {blocks[1]}, {blocks[2]}}}, "
            f"dim3{{{threads[0]}, {threads[1]}, {threads[2]}}}, [&] {{ {call} }});"
        )
    outputs = [name for name, buffer in buffers.items() if buffer.data is None]
    for name in outputs:
        body.append(f'  save("{workdir / name}.out", {name});')
    body.append("  return 0;\n}\n")
    source = workdir / "kernels.cpp"
    source.write_text("\n".join(body))
    binary = workdir / "kernels"
    built = subprocess.run(
        [
            compiler(),
            "-std=c++17",
            "-O1",
            "-pthread",
            "-Wno-unknown-pragmas",
            "-Wno-unused-variable",
            "-o",
            str(binary),
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if built.returncode:
        raise RuntimeError(built.stderr[-4000:])
    subprocess.run([str(binary)], check=True, timeout=600)
    return {
        name: np.fromfile(workdir / f"{name}.out", dtype=STORAGE[buffers[name].dtype])
        for name in outputs
    }
