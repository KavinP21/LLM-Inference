# LLM Inference

Forge LLM is a C++20 and Python inference runtime with an Apple MLX/Metal backend
and a CUDA backend. It implements transformer execution, KV-cache allocation,
request scheduling, and GPU kernels. PyTorch and Transformers are used for weight
export and reference checks; neither executes the inference loop.


## Runtime

- **Paged KV cache:** 16-token physical pages, lazy allocation, capacity reservations,
  and reclamation on completion, cancellation, or failure.
- **Continuous batching:** decode across runnable requests, with long prompts split
  into scheduler-visible prefill chunks.
- **Metal kernels:** fused RoPE/page writes and paged GQA decode attention; Qwen also
  uses fused residual/RMSNorm and SwiGLU operations. The MLX fallback stays selectable.
- **Model artifacts:** a versioned, checksummed container shared by Python and C++
  readers, with explicit model-family validation.
- **Prefix reuse:** opt-in FP16 caching with reference-counted pages, copy-on-write
  tails, request namespaces, and bounded LRU retention.

Read the [architecture](docs/architecture.md) for execution and memory layout,
and [design notes](docs/design-notes.md) for tradeoffs and numerical issues.

## Measured performance

Qwen2.5-0.5B, FP16, Apple M3 Max, MLX 0.32.2. Each configuration generates 32 tokens
per request, after five warmups, with three measured repetitions. Prompts are fixed
synthetic token arrays. Throughput is aggregate generated tokens/s.

| Prompt tokens | Concurrent requests | MLX fallback | Full Metal path | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 67.4 | 115.4 | 1.71× |
| 128 | 8 | 151.9 | 376.2 | 2.48× |
| 1,024 | 1 | 28.5 | 29.5 | 1.04× |
| 1,024 | 8 | 36.6 | 81.7 | 2.23× |

These September 23 measurements compare two execution modes of this engine on one
machine. The benefit is largest for concurrent decode; page packing limits the
single-request gain at 1,024 tokens. The [benchmark report](docs/resume-benchmark-results.md)
contains latency, memory, source identity, and reproduction commands;
[raw JSON](benchmarks/results/resume-m3-max-2026-09-23/) is checked in.

## Validation and limits

| Area | Evidence |
| --- | --- |
| Python regression | 365 tests passed on M3 Max, including Metal tests, on October 6, 2026. [QA report](benchmarks/results/repository-qa-2026-10-06/summary.json) |
| Qwen reference parity | All 32 greedy tokens matched Transformers on each of four fixed prompts. |
| Gemma reference parity | Four first-token cosine gates pass; three continuations match all 32 tokens. One diverges after 25 tokens at an FP16 near-tie. |
| 32K context | Both models executed a 32,766-token prompt plus two outputs and reclaimed their pages. This checks execution and resource integrity. |
| Prefix cache | All eight registered parity/resource gates pass. Performance and final qualification are pending. |
| INT8 | Experimental; held-out exact-token quality gates fail. FP16 remains the default. |
| CUDA | On-device correctness and performance are unverified. |

See [implementation status](STATUS.md) for the reports behind these claims.
The runtime uses greedy decoding. HTTP serving, speculative decoding, 4-bit
quantization, and distributed execution are outside the implemented scope.

## Run on Apple Silicon

Use native arm64 Python 3.12 with Metal access and the Xcode command-line tools.
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
CMAKE_ARGS="-DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF" \
  python -m pip install -e '.[mlx,export,test]'
forge-export Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b.engine
```

```python
from transformers import AutoTokenizer
from forge_llm import create_engine

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain paged KV caching."}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
)
with create_engine(
    "models/qwen2.5-0.5b.engine",
    backend="mlx",
    max_num_sequences=8,
    max_model_length=32768,
    kv_cache_bytes=512 << 20,
    prefill_chunk_size=512,
) as engine:
    output = engine.generate(
        prompt, max_new_tokens=64, eos_token_ids=[tokenizer.eos_token_id]
    )
    print(tokenizer.decode(output, skip_special_tokens=True))
```

For streaming or concurrent requests, use `submit`, `step`, `cancel`, and `stats`.
[Prefix-cache examples](docs/prefix-cache.md) cover namespaces and cache clearing.
[Apple setup](docs/setup-mlx.md) includes Gemma export and correctness commands;
[WSL2 setup](docs/setup-wsl.md) covers CUDA. Model weights are excluded from Git.

## Tests and benchmarks

```bash
# Portable C++ components; requires CMake 3.24+ and Ninja.
cmake -S . -B build -G Ninja -DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# Full suite on Apple Silicon with test/export/MLX extras installed.
FORGE_INSPECT_MODEL=build/forge-inspect-model PYTHONPATH=python python -m pytest -q

# Repeat with --mlx-kernel-mode baseline for a controlled comparison.
PYTHONPATH=python python benchmarks/run_matrix.py \
  --backend mlx --model models/qwen2.5-0.5b.engine \
  --output-dir benchmark-results/full --mlx-kernel-mode full \
  --prompt-lengths 128 1024 --output-lengths 32 --concurrencies 1 8 \
  --warmups 5 --repetitions 3
```

Tests cover independent model references, artifact corruption, page boundaries,
admission failure, concurrent decode, copy-on-write, and cancellation cleanup.
GitHub Actions runs portable host and evidence-contract tests on Linux and macOS;
Metal and CUDA checks require their respective devices.

## Code map

| Path | Contents |
| --- | --- |
| [python/forge_llm/](python/forge_llm/) | Engine, scheduler, model adapters, physical pages, calibration tools |
| [src/](src/) and [include/forge/](include/forge/) | Portable C++ allocator, scheduler, artifact validation |
| [cuda/](cuda/) | CUDA Qwen graph, kernels, cuBLASLt context, memory arenas |
| [tests/](tests/) | Numerical references, lifecycle tests, evidence audits |
| [benchmarks/](benchmarks/) | Workloads, measurement drivers, saved reports |

A local chat UI is available at [tools/forge_chat.py](tools/forge_chat.py).
The next work is [prefix-cache measurement and NVIDIA validation](docs/roadmap.md).
