"""Compile Instinct's CUDA kernels with NVRTC, without a GPU.

``mx.fast.cuda_kernel`` wraps each kernel body in a generated signature and
compiles it with NVRTC the first time it runs. This rebuilds the same source
for every shape Gemma 4 and Qwen3.5 use and compiles it the way MLX 0.32.4
does, for several GPU generations, so a syntax or type error, or a kernel
that needs more registers or shared memory than a GPU has, shows up in CI
instead of on someone's machine. (At run time such a kernel would only fall
back to the slower portable path; see ``backends/cuda_kernels.py``.)

Needs, from pip: nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12 and
mlx-cuda-12 (installed with --no-deps; only its headers are used). MLX's
device headers aren't in its wheel, so two are fetched from MLX's source at
the matching tag.

    python tools/nvidia/compile_kernels.py
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import site
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Loaded by path: the kernel module needs nothing else of the package.
_spec = importlib.util.spec_from_file_location(
    "cuda_kernels", ROOT / "python/instinct/backends/cuda_kernels.py"
)
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)

MLX_TAG = "v0.32.4"
DEVICE_HEADERS = ("utils.cuh", "complex.cuh")
ARCHES = ("sm_75", "sm_86", "sm_89", "sm_90")

# What mx.fast.cuda_kernel puts in front of every kernel.
DEFAULT_HEADER = """
#include "mlx/backend/cuda/device/utils.cuh"

#include <cooperative_groups.h>

"""
CUDA_TYPES = {
    "float32": "float",
    "float16": "__half",
    "bfloat16": "__nv_bfloat16",
    "int32": "int32_t",
    "int8": "int8_t",
    "uint32": "uint32_t",
}


def build_kernel(name, header, source, inputs, outputs, template):
    """MLX 0.32.4's ``build_kernel``: ``inputs`` and ``outputs`` are
    (name, dtype) pairs and ``template`` (name, value) pairs, where a
    value is an int, a bool or a dtype name."""
    text = DEFAULT_HEADER + header
    text += "namespace mlx::core::cu {\n\nnamespace cg = cooperative_groups;\n\n"
    text += f"__global__ void {name}(\n"
    for arg, dtype in inputs:
        text += f"    const {CUDA_TYPES[dtype]}* {arg},\n"
        if arg + "_shape" in source:
            text += f"    const __grid_constant__ Shape {arg}_shape,\n"
        if arg + "_strides" in source:
            text += f"    const __grid_constant__ Strides {arg}_strides,\n"
        if arg + "_ndim" in source:
            text += f"    const __grid_constant__ int {arg}_ndim,\n"
    text += ",\n".join(f"    {CUDA_TYPES[dtype]}* {arg}" for arg, dtype in outputs)
    text += ") {\n"
    for arg, value in template:
        if isinstance(value, bool):
            text += f"  constexpr bool {arg} = {'true' if value else 'false'};\n"
        elif isinstance(value, int):
            text += f"  constexpr int {arg} = {value};\n"
        else:
            text += f"  using {arg} = {CUDA_TYPES[value]};\n"
    text += "\n" + source + "\n}\n\n} // namespace mlx::core::cu\n"
    return text


# (label, dtype, kv storage, head dim, q heads, kv heads, window)
ATTENTION = (
    ("qwen-q8", "bfloat16", "int8", 256, 16, 4, 0),
    ("qwen-16", "bfloat16", "bfloat16", 256, 16, 4, 0),
    ("gemma-sliding", "bfloat16", "bfloat16", 256, 16, 8, 1024),
    ("gemma-full", "bfloat16", "bfloat16", 512, 16, 2, 0),
    ("generic-128", "float16", "float16", 128, 32, 8, 0),
)
PAGE_SIZE = 64


def kernels():
    """Every (label, source) the models compile."""
    for label, dtype, storage, dim, q_heads, kv_heads, window in ATTENTION:
        quantized = storage == "int8"
        group, repeat, _ = ck.decode_layout(q_heads, kv_heads, 1, window)
        template = [
            ("T", dtype),
            ("D", dim),
            ("FULL", dim % 32 == 0),
            ("HQ", q_heads),
            ("HKV", kv_heads),
            ("G", group),
            ("R", repeat),
            ("PAGE", PAGE_SIZE),
            ("WINDOW", window),
        ]
        inputs = [
            ("queries", dtype),
            ("keys", storage),
            ("values", storage),
            ("tables", "int32"),
            ("lengths", "int32"),
            ("params", "float32"),
        ]
        if quantized:
            inputs += [("key_scales", "float16"), ("value_scales", "float16")]
        for kind, write, outputs in (
            ("single", ck.SINGLE_WRITE, [("out", dtype)]),
            (
                "split",
                ck.SPLIT_WRITE,
                [
                    ("partial_o", "float32"),
                    ("partial_m", "float32"),
                    ("partial_l", "float32"),
                ],
            ),
        ):
            source = ck.decode_source(quantized, write)
            yield (
                f"decode-{kind}-{label}",
                32 * group * repeat,
                build_kernel(
                    f"custom_kernel_{label.replace('-', '_')}_{kind}",
                    ck.HEADER,
                    source,
                    inputs,
                    outputs,
                    template,
                ),
            )
        yield (
            f"decode-merge-{label}",
            32,
            build_kernel(
                f"custom_kernel_{label.replace('-', '_')}_merge",
                ck.HEADER,
                ck.DECODE_MERGE,
                [
                    ("partial_o", "float32"),
                    ("partial_m", "float32"),
                    ("partial_l", "float32"),
                ],
                [("out", dtype)],
                [("T", dtype), ("D", dim), ("FULL", dim % 32 == 0)],
            ),
        )
    for dtype in ("bfloat16", "float16"):
        yield (
            f"gated-delta-{dtype}",
            128,
            build_kernel(
                "custom_kernel_gated_delta",
                ck.HEADER,
                ck.GATED_DELTA,
                [
                    ("q", dtype),
                    ("k", dtype),
                    ("v", dtype),
                    ("g", "float32"),
                    ("beta", "float32"),
                    ("state_in", "float32"),
                ],
                [("y", dtype), ("state_out", "float32")],
                [("T", dtype), ("DK", 128), ("DV", 128), ("HK", 16), ("HV", 32)],
            ),
        )


def _package_dir(name: str) -> Path | None:
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        path = Path(base) / name
        if path.exists():
            return path
    spec = importlib.util.find_spec(name.split("/")[0])
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).parent / name
    return None


def find_paths(headers: Path | None) -> tuple[Path, list[Path]]:
    nvrtc = _package_dir("nvidia/cuda_nvrtc/lib")
    runtime = _package_dir("nvidia/cuda_runtime/include")
    mlx = _package_dir("mlx/include")
    missing = [
        name
        for name, path in (
            ("nvidia-cuda-nvrtc-cu12", nvrtc),
            ("nvidia-cuda-runtime-cu12", runtime),
            ("mlx-cuda-12 (headers)", mlx),
        )
        if path is None or not path.exists()
    ]
    if missing:
        raise SystemExit("missing pip packages: " + ", ".join(missing))
    library = next(iter(sorted(nvrtc.glob("libnvrtc.so*"))), None)
    if library is None:
        raise SystemExit(f"no libnvrtc in {nvrtc}")
    device = (headers or Path(tempfile.mkdtemp())) / "mlx/backend/cuda/device"
    device.mkdir(parents=True, exist_ok=True)
    for name in DEVICE_HEADERS:
        target = device / name
        if not target.exists():
            url = (
                f"https://raw.githubusercontent.com/ml-explore/mlx/{MLX_TAG}/"
                f"mlx/backend/cuda/device/{name}"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                target.write_bytes(response.read())
    config = mlx / "mlx/backend/cuda/device/config.h"
    if not (device / "config.h").exists():
        (device / "config.h").write_bytes(config.read_bytes())
    includes = [device.parents[3], mlx / "cccl", runtime]
    return library, includes


def compile_all(library: Path, includes: list[Path], arches, verbose: bool) -> int:
    nvrtc = ctypes.CDLL(str(library))
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    failures = 0
    for label, threads, source in kernels():
        for arch in arches:
            program = ctypes.c_void_p()
            check = nvrtc.nvrtcCreateProgram(
                ctypes.byref(program),
                source.encode(),
                f"{label}.cu".encode(),
                0,
                None,
                None,
            )
            if check:
                raise SystemExit(nvrtc.nvrtcGetErrorString(check).decode())
            options = [
                b"--device-as-default-execution-space",
                f"--gpu-architecture={arch}".encode(),
                b"--ptxas-options=-v",
                *(f"--include-path={path}".encode() for path in includes),
            ]
            array = (ctypes.c_char_p * len(options))(*options)
            result = nvrtc.nvrtcCompileProgram(program, len(options), array)
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(program, log)
            nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
            text = log.value.decode(errors="replace").strip()
            spills = "spill stores" in text and " 0 bytes spill stores" not in text
            if result:
                failures += 1
                print(f"FAIL {label} {arch} ({threads} threads/block)\n{text}\n")
            else:
                usage = [
                    line.strip()
                    for line in text.splitlines()
                    if "Used" in line or "spill" in line
                ]
                note = " ".join(usage) if verbose or spills else ""
                flag = "  (spills registers)" if spills else ""
                print(f"ok   {label} {arch} ({threads} threads/block){flag} {note}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", action="append", help="e.g. sm_86; repeatable")
    parser.add_argument("--headers", type=Path, help="where to keep MLX's headers")
    parser.add_argument("--dump", type=Path, help="also write each source here")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for label, _, source in kernels():
            (args.dump / f"{label}.cu").write_text(source)
    library, includes = find_paths(args.headers)
    failures = compile_all(library, includes, args.arch or ARCHES, args.verbose)
    print(f"{failures} failed" if failures else "all kernels compiled")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
