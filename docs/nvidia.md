# Running on NVIDIA GPUs

Instinct runs Gemma 4 and Qwen3.5 on NVIDIA GPUs through MLX's CUDA backend,
with its own CUDA kernels for the two hottest pieces of work. You get the
same app as on a Mac: chats, projects, agents, the `/v1` API and guest
sharing. The interface opens in your browser.

These kernels haven't run on NVIDIA hardware yet. They've been compiled for
every recent GPU generation and run on a CPU emulator against reference math
(see [How it's checked](#how-its-checked)). Each kernel also checks itself
the first time it runs, and any kernel that fails falls back to plain MLX
ops. So on a machine they don't suit, Instinct gets slower but stays correct.

The separate [C++ CUDA engine](../cuda/) is unrelated: it reads only exported
Qwen2 `.engine` files ([setup-wsl.md](setup-wsl.md)).

## Quick start: Windows with WSL2 (an RTX 3070 Ti, say)

1. **Install the current NVIDIA driver in Windows.** Don't install a driver
   inside Linux; WSL uses the Windows one.
2. **Install Ubuntu.** In PowerShell, run this, then reboot if asked:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

3. **Set Instinct up.** In Ubuntu, run:

   ```bash
   sudo apt update && sudo apt install -y git python3-venv
   git clone https://github.com/KavinP21/LLM-Inference.git instinct && cd instinct
   tools/nvidia/instinct.sh setup
   ```

   `setup` makes a Python environment, installs MLX with CUDA and downloads
   the models your card can hold, then runs `check`. No CUDA toolkit is
   needed: pip brings the libraries, and kernels are compiled at run time.

4. **Start it:**

   ```bash
   tools/nvidia/instinct.sh start
   ```

   Instinct opens in your Windows browser, already signed in. If the browser
   doesn't open, the terminal prints a sign-in link. Open it on the same PC;
   it stops working when Instinct stops. Projects default to `~/Instinct`.
   `--workspace DIR` picks another folder, and `/mnt/c/Users/<you>/...`
   reaches your Windows files. Press Ctrl-C to stop.

Plain Linux works the same way: glibc 2.35 or newer (Ubuntu 22.04+), a 550.54+
driver and a GPU of compute capability 7.5 (Turing) or newer.

Settings, API keys, guest invites and the project list live in
`~/.local/share/instinct`. The server listens only on 127.0.0.1. Under WSL,
that's reachable from Windows on the same PC and nowhere else.

**One difference from the Mac:** agent commands aren't sandboxed on Linux.
macOS confines them with Seatbelt; on Linux they run as you, and the app
marks them unsandboxed. Use *Ask* or *Read only* in a chat when that matters.

## Which models fit

`start` sizes everything to the card's memory each time
(`python -m instinct.nvidia profile`):

| GPU memory | Main agent | Sub-agents | Context per chat | Chats at once |
| --- | --- | --- | --- | --- |
| Under 6 GB | not supported | | | |
| 6–12 GB (RTX 3070 Ti: 8 GB) | Qwen3.5 4B | Qwen3.5 4B | 64K–256K | 2–4 |
| 12–20 GB | Qwen3.5 4B | Qwen3.5 4B | 256K | 8 |
| 20–40 GB (RTX 3090/4090) | Gemma 4 26B | Gemma 4 26B | 64K–256K | 4 |
| 40 GB and up (A100, H100) | Gemma 4 26B | Qwen3.5 4B | 256K | 16 |

An 8 GB RTX 3070 Ti gets a 128K context and 4 chats at once. A card just
under the 21 GB Gemma needs gets Qwen instead.

**Why Gemma needs 20 GB.** It's 15.6 GB at 4 bits, and 12.9 GB of that is
its 128 experts, so it can't run from an 8 GB card's memory.

**How the budget is split.** Each worker gets an explicit memory limit:
weights, K/V cache, recurrent-state slots (about 52 MB each for Qwen3.5) and
activations. 1.5 GB is kept back for the Windows desktop and CUDA's own
context. The config's `memory.gpu_gb` holds the total, so a config that
couldn't fit is refused at startup instead of running out of memory
mid-reply ([models.md](models.md), "Memory ceiling").

## Checking a machine

```bash
tools/nvidia/instinct.sh check
```

This loads no model and takes a few seconds. It prints a JSON report:

- the GPU, its memory and driver;
- whether this is WSL;
- MLX's version and backend;
- the profile the card gets;
- under `cuda_kernels`, each CUDA kernel's result at the exact shapes Gemma 4
  and Qwen3.5 use. Each entry has `passed`, `error` and `seconds`, and is
  compared with plain MLX.

It exits non-zero if any kernel fails.

A running engine reports the same in `engine.stats()["kernels"]`:

- `"cuda"` means the custom kernel is in use.
- `"portable"` means plain MLX is used, either because the kernel failed its
  check for some shape or because `INSTINCT_KERNELS=portable` is set.

## The CUDA kernels

[backends/cuda_kernels.py](../python/instinct/backends/cuda_kernels.py) holds
CUDA counterparts of the Metal kernels in `metal_kernels.py`. MLX compiles
them with NVRTC through `mx.fast.cuda_kernel`, per shape, the first time
each shape runs.

- **Paged decode attention** ("flash decoding"). Each block serves one KV
  head of one sequence over one slice of its keys, with a warp per query head
  that shares the KV head. Keys and values are read straight from the page
  arena, so nothing is gathered first, and 8-bit K/V is dequantized on the
  fly. Long contexts split across blocks, and a second kernel merges the
  slices. It handles Gemma's 1,024-token sliding window and its 512-wide
  full-attention heads.
- **Qwen3.5's gated delta rule.** Each warp owns one row of a head's 128×128
  recurrent state in registers and walks the tokens: decay, read the key's
  memory, correct it toward the value, read out with the query.

Without these kernels, decode attention first copies each sequence's keys
into a dense block, so its memory traffic grows with the context. The gated
delta rule falls back to 64-token chunks of batched matmuls.

**The first-use check.** The first time a kernel runs at a shape, it runs on
random inputs at that exact shape. That means the same head dim, head counts,
page size, window and K/V width, both one slice and several. The result is
compared with plain MLX. A kernel that doesn't compile, can't launch or
disagrees is never used for that shape again, and Instinct logs a warning.

`INSTINCT_KERNELS` overrides the choice:

- `portable` skips the custom kernels.
- `plain` also replaces MLX's fused ops (RoPE, RMS norm, attention,
  `mx.compile`) with plain versions. It's about 2.5x slower; use it as a
  last resort.

Off Metal, each fused op is also tried once on small inputs and replaced if
it fails. Everything else (4-bit matmuls, Gemma's expert routing through
`gather_qmm`) is MLX's own CUDA code.

## How it's checked

| Check | What it covers | Where |
| --- | --- | --- |
| NVRTC compile | Every kernel at every model shape, compiled the way MLX 0.32.4 compiles them, for `sm_75`, `sm_86` (RTX 30), `sm_89` (RTX 40) and `sm_90` (H100). None spill registers. The 1,024-thread Gemma full-attention kernel uses 58 registers, under the 64 that allows. | `tools/nvidia/compile_kernels.py`, in CI |
| CPU emulation | The kernels' source run on the CPU with one OS thread per CUDA thread (warp shuffles, `__syncthreads`, shared memory, bf16/fp16), against float64 references. Covers single and split decode, the merge, page tables, windows, 8-bit K/V, softcap, odd head dims and the recurrence. Breaking the window, the merge or the head mapping makes it fail. | `tests/test_cuda_kernels.py` |
| Engine path choice | Stand-in kernels play the CUDA ones. A kernel that passes is used. One that errors or gives wrong answers falls back, and the replies match either way. | `tests/test_cuda_dispatch.py` |
| Portable path | Against the Metal path on an M3 Max (below), and on MLX's CPU backend in CI. | `tests/test_portable_kernels.py` |
| On NVIDIA hardware | Not yet. Run `tools/nvidia/instinct.sh check`. | |

The portable path against the Metal path on an M3 Max, with real
checkpoints and a 3,000-token prompt:

| Model | Logits cosine | Greedy tokens identical | Decode, Metal → portable |
| --- | --- | --- | --- |
| Qwen3.5-4B | 0.9994 | 31 of 49 | 69.6 → 63.8 tok/s |
| Gemma 4 26B A4B | 1.00000 | 49 of 49 | 53.9 → 58.7 tok/s |

## Clusters

[pace.md](pace.md) runs the same engine on an H100 at Georgia Tech's PACE
ICE, inside Apptainer, because RHEL 9's glibc is too old for MLX's wheels.
