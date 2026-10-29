# Forge LLM

An LLM inference engine written from scratch in Python and C++, with an Apple
MLX/Metal backend and a CUDA backend. It runs Qwen2 and Gemma 3 with a paged KV
cache, continuous batching and custom Metal kernels. PyTorch and Transformers are
only used to export weights, tokenize and check outputs; they're never in the
generation loop.

The repo also has a few things built on top of the engine: speculative decoding,
a multi-agent task runtime, a local web UI (Forge Studio), and a Qwen3.5-4B
multimodal path.

## What the engine does

- **Paged KV cache.** 16-token pages, allocated lazily. Each request reserves its
  worst-case page count at admission, so it can't run out halfway through, and
  pages are returned on completion, cancellation or error.
- **Continuous batching.** Every running request decodes in the same step. Long
  prompts are split into prefill chunks that the scheduler interleaves with
  decoding.
- **Metal kernels.** Fused residual + RMSNorm, RoPE fused with the KV page write,
  SwiGLU, and a paged GQA decode attention kernel that walks per-request block
  tables. They can be switched off to fall back to plain MLX.
- **Model files.** A versioned, checksummed single-file format that both the
  Python and C++ readers validate before running anything.
- **Prefix caching** (opt-in). Reference-counted shared pages with copy-on-write
  for the last partial page, namespaces, and LRU eviction.
- **Speculative decoding** (opt-in). N-gram drafts verified in one forward pass,
  with rejected KV rolled back.
- **INT8 weights** (experimental). Per-channel W8A16 with a custom Metal kernel.
  It saves about 35% of file size, but it doesn't match FP16 output closely
  enough to be the default. See [docs/quantization.md](docs/quantization.md).

## Performance

Qwen2.5-0.5B, FP16, Apple M3 Max, MLX 0.32.2. Each request generates 32 tokens.
Throughput is total generated tokens per second, averaged over three runs after
five warmups:

| Prompt tokens | Concurrent requests | Plain MLX | Custom kernels | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 67.4 | 115.4 | 1.71× |
| 128 | 8 | 151.9 | 376.2 | 2.48× |
| 1,024 | 1 | 28.5 | 29.5 | 1.04× |
| 1,024 | 8 | 36.6 | 81.7 | 2.23× |

This compares the engine with and without its own kernels on the same machine,
not against other inference engines. The kernels help most with concurrent
decoding. A single long request barely improves, because MLX arrays are
immutable and the attention kernel has to copy that request's pages into one
array first. [docs/benchmarks.md](docs/benchmarks.md) has latency, memory, a
32K-context run and the correctness checks against Transformers. Raw JSON is in
[benchmarks/results](benchmarks/results/fp16-m3-max-2026-09-23/).

## Status

| Area | Where it stands |
| --- | --- |
| Qwen2.5 on MLX | Matches Transformers greedy output token for token (4 prompts × 32 tokens) |
| Gemma 3 on MLX | 3 of 4 continuations match exactly; the 4th diverges after 25 tokens at an FP16 near-tie |
| 32K context | Runs on both models and releases every page afterwards |
| Prefix cache | Bit-identical outputs with and without reuse; timing not measured yet |
| INT8 | Experimental. No variant matched FP16 on all 25 test prompts (best was 23) |
| CUDA | Compiles in CI. **Hasn't been run on an NVIDIA GPU yet**, so correctness and speed are unknown |

The native engine is greedy-only. There's no sampling, native 4-bit weights,
streaming server or multi-GPU support yet. See
[docs/roadmap.md](docs/roadmap.md).

## Getting started (Apple Silicon)

You need native arm64 Python 3.11+ and the Xcode command-line tools.

```bash
python3 -m venv .venv
source .venv/bin/activate
CMAKE_ARGS="-DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF" \
  pip install -e '.[mlx,export,test]'

forge-export Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b.engine
```

```python
from transformers import AutoTokenizer
from forge_llm import create_engine

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain paged KV caching."}],
    tokenize=True, return_dict=False, add_generation_prompt=True,
)

with create_engine(
    "models/qwen2.5-0.5b.engine",
    backend="mlx",
    max_num_sequences=8,
    kv_cache_bytes=512 << 20,
    prefill_chunk_size=512,
) as engine:
    output = engine.generate(prompt, max_new_tokens=64,
                             eos_token_ids=[tokenizer.eos_token_id])
    print(tokenizer.decode(output, skip_special_tokens=True))
```

For several concurrent requests, use `submit()`, then call `step()` until they
finish. `cancel()` and `stats()` are also available. [docs/setup-mlx.md](docs/setup-mlx.md)
covers Gemma export and the correctness tools, and [docs/setup-wsl.md](docs/setup-wsl.md)
covers building the CUDA backend under WSL2.

## Also in this repo

**Gemma 4 and Qwen3.5 (text, images).** The default setup uses Gemma 4 26B A4B
(QAT 4-bit) as the main model and Qwen3.5-4B for sub-agents. Both run through
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) inside Forge's [VLM runner](docs/vlm-runner.md),
which reuses prompts across turns: a follow-up message starts generating in
57 ms on Qwen3.5 and 84 ms on Gemma, against ~1 s or more for a cold prompt. See
[docs/models.md](docs/models.md).

```bash
pip install -e '.[mlx,vlm,agents]'
python -m forge_llm.prepare_models       # ~18 GB into models/
```

**Forge Studio.** A local web UI (`python tools/forge_studio.py`, or double-click
`Forge Studio.command`) for chatting with the model and letting it work in a
directory. Tool calls, sub-agents and approval prompts for edits and commands
show up inline in the conversation. It has an optional developer panel, image
attachments, skills and stdio MCP servers. `--demo` runs the UI without loading
a model. See [docs/forge-studio.md](docs/forge-studio.md).

**Agent runtime.** Logical agents share a pool of model worker processes (local
or remote over authenticated HTTP), spawn bounded child agents, and journal
everything to SQLite so a run can resume after a crash. Workspace tools are
confined to one directory, with version-checked edits. See
[docs/agent-runtime.md](docs/agent-runtime.md).

**Speculative decoding.** See [docs/speculative-decoding.md](docs/speculative-decoding.md).
On Qwen2.5-0.5B it's up to 4× faster when the output repeats earlier text, and
1.2–1.5× faster on code and arithmetic.

## Tests

```bash
# C++ host components (CMake 3.24+, Ninja)
cmake -S . -B build -G Ninja -DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF
cmake --build build && ctest --test-dir build --output-on-failure

# Python. Metal tests run when mlx is installed and skip otherwise.
FORGE_INSPECT_MODEL=build/forge-inspect-model PYTHONPATH=python python -m pytest -q
```

The Metal tests check tiny random Qwen and Gemma models against independent
references at page boundaries, with mixed-length batches, chunked prefill,
cancellation and the custom kernels on and off. CI runs the full suite on
Linux and macOS (without Metal), the C++ tests under ASan/UBSan, and a CUDA
compile in the `nvidia/cuda` image.

## Layout

| Path | |
| --- | --- |
| [python/forge_llm/](python/forge_llm/) | MLX engine, scheduler, paged KV store, model adapters, Metal kernels, quantization |
| [python/forge_llm/agents/](python/forge_llm/agents/), [harness/](python/forge_llm/harness/) | Agent runtime and workers; Forge Studio |
| [src/](src/), [include/forge/](include/forge/) | C++ model file reader, KV block pool, scheduler |
| [cuda/](cuda/) | CUDA Qwen graph, kernels, cuBLASLt wrapper, memory arenas |
| [benchmarks/](benchmarks/) | Benchmark drivers, prompt sets, saved FP16 results |
| [tests/](tests/) | Python and C++ tests |
| [docs/](docs/) | Design notes and write-ups ([architecture](docs/architecture.md), [design notes](docs/design-notes.md)) |
