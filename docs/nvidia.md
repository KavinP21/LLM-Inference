# Running on NVIDIA GPUs

The MLX engine, with Gemma 4, Qwen3.5, `ChatRunner` and the `/v1` API, also runs
on NVIDIA GPUs through MLX's CUDA backend. This is separate from the C++ CUDA
engine in [cuda/](../cuda/), which only reads exported Qwen2 `.engine` files and
hasn't run on a GPU.

## Install

You need Linux with glibc 2.35 or newer, an NVIDIA driver of 550.54 or newer,
and a GPU of compute capability 7.5 or newer. MLX's Linux wheels need that
glibc, so on RHEL 9 (glibc 2.34) run inside a container with a newer one,
such as `python:3.12-slim-bookworm` under Apptainer.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "mlx[cuda12]==0.32.4" "nvidia-cuda-runtime-cu12==12.9.*" \
  "transformers>=5.14,<6" "safetensors>=0.8" "huggingface-hub>=1.0" Pillow
export PYTHONPATH=$PWD/python
python -m instinct.prepare_models          # Gemma 4 and Qwen3.5, about 19 GB
```

The pip packages bring the CUDA libraries, so no CUDA toolkit is required. The
`nvidia-cuda-runtime-cu12` package supplies the headers MLX uses to compile
kernels at run time.

## What changes off Metal

`backends/metal_kernels.py` holds the two hand-written Metal kernels: paged
decode attention and Qwen3.5's gated delta rule. Elsewhere the engine uses
[backends/portable.py](../python/instinct/backends/portable.py) instead:

- **Decode attention** gathers each sequence's keys into a dense block and runs
  ordinary attention over it.
- **The gated delta rule** runs 64 tokens at a time with batched matmuls. Only
  the state handoff between chunks is sequential.

MLX's fused ops can be missing for a particular shape on a given backend, and the
error only shows up when the graph runs, mid-step. So when the engine starts
off Metal, it tries each fused op once on tiny inputs and compares the result
with a plain version. Any op that fails or disagrees is replaced by the plain
version. The ops checked are:

- RoPE with per-sequence offsets and explicit frequencies
- attention at head dims 128, 256 and 512, with causal and boolean masks
- RMS norm, with and without a weight
- `mx.compile`

`engine.stats()["kernels"]` shows what's in use:

```json
{"decode_attention": "portable", "gated_delta": "portable", "rope": "fused",
 "rms_norm": "fused", "attention": "fused", "compile": "fused"}
```

`INSTINCT_KERNELS` overrides the choice:

- `portable` skips the Metal kernels.
- `plain` also replaces every fused op and turns `mx.compile` off.

`plain` is an escape hatch, about 2.5x slower on Gemma.

## How it was checked

There's no NVIDIA GPU here, so the portable path is checked three ways:

- **Against the Metal path on an M3 Max**, with real checkpoints and a 3,000-token
  prompt:

  | Model | Logits cosine | Greedy tokens identical | Decode, Metal → portable |
  | --- | --- | --- | --- |
  | Qwen3.5-4B | 0.9994 | 31 of 49 | 69.6 → 63.8 tok/s |
  | Gemma 4 26B A4B | 1.00000 | 49 of 49 | 53.9 → 58.7 tok/s |

  Every op replaced by its plain version on Gemma gives cosine 0.9996 and the
  same top token.
- **On MLX's CPU backend**, which has no Metal either. Every engine test runs
  through the portable path there, in CI on Linux
  (`INSTINCT_TEST_DEVICE=cpu` does the same on a Mac).
- **Against references in [tests/test_portable_kernels.py](../tests/test_portable_kernels.py).**
  The chunked rule matches a float64 recurrence, including correlated keys at
  near-full write strength, where a naive chunk solve overflows. Each plain op
  matches its fused op.

Speed on NVIDIA is unmeasured. Decode attention reads every cached key twice
(gather, then attend), which matters only at long contexts.
