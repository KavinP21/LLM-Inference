# Forge LLM

Forge LLM is a focused C++/CUDA inference runtime for `Qwen/Qwen2.5-0.5B-Instruct`. It does not call
PyTorch during inference. Matrix multiplications use cuBLASLt; request scheduling, memory
management, paged KV caching, transformer composition, fused pointwise operations, paged grouped-
query attention, and greedy selection live in this repository.

See [STATUS.md](STATUS.md) for the exact validated/unvalidated boundary before treating this as a
release-ready engine.

The project currently provides an end-to-end correctness-first path with full-prompt GEMMs, causal
paged prefill attention, and dynamically batched one-token decoding. The prefill attention kernel
is deliberately straightforward; profiling it against a tiled/FlashAttention-style implementation
is a future optimization rather than a hidden framework dependency.

## What is implemented

- A deterministic, checksummed, 256-byte-aligned model container and Hugging Face exporter.
- Strict Qwen2 tensor and configuration validation.
- RAII CUDA activation/workspace arenas and cached cuBLASLt execution context.
- Qwen2 FP16 execution with FP32 RMSNorm, softmax, and GEMM accumulation.
- Fused residual/RMSNorm, RoPE/KV scatter, and SwiGLU CUDA kernels.
- A 16-token paged GQA decode-attention kernel.
- Full-prompt Q/K/V and MLP GEMMs with causal paged prefill attention.
- Reservation-aware KV admission, lazy physical block allocation, and fragmentation metrics.
- FCFS iteration-level continuous batching with cancellation and immediate reclamation.
- A pybind11 API, reproducible JSON benchmark runner, portable host tests, and CI.

Not yet implemented: tiled/FlashAttention-style prefill, quantization, probabilistic sampling, HTTP serving,
prefix caching, CUDA graphs, Metal, or multi-GPU execution.

## Development environments

Host-only components build on macOS and Linux. CUDA execution targets Ubuntu 24.04 under WSL2 on
an RTX 3070 Ti (SM 8.6).

See [the WSL2 setup and first-validation runbook](docs/setup-wsl.md) for the NVIDIA host.

```bash
# macOS or a non-CUDA machine
cmake -S . -B build -G Ninja -DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# WSL2 CUDA machine
./tools/verify_wsl_environment.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install '.[export,test]'
```

The CUDA build defaults to SM 8.6. Change `CMAKE_CUDA_ARCHITECTURES` explicitly when targeting a
different GPU.

## Export and generate

```bash
forge-export Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b.engine
forge-inspect-model models/qwen2.5-0.5b.engine

python - <<'PY'
from transformers import AutoTokenizer
from forge_llm import Engine

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokens = tokenizer("Explain paged KV caching:").input_ids
engine = Engine("models/qwen2.5-0.5b.engine", max_num_sequences=16,
                max_model_length=2048, kv_cache_bytes=512 << 20)
output = engine.generate(tokens, max_new_tokens=64,
                         eos_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output))
PY
```

The asynchronous interface consists of `submit`, `step`, `cancel`, and `stats`. Each `step` returns
token events for active requests, allowing new requests to join between decode iterations.

## Benchmark

```bash
forge-bench \
  --model models/qwen2.5-0.5b.engine \
  --prompts benchmarks/prompts.json \
  --output benchmark-results/c4-prompt-mix.json \
  --output-length 32 --warmups 5 --repetitions 3
```

The result contains raw request observations, TTFT/TPOT/end-to-end percentiles, throughput, engine
cache statistics, and a hardware/software manifest. See [the benchmark protocol](docs/benchmarking.md)
and [runtime architecture](docs/architecture.md).

## Correctness gates

- Every custom kernel is compared against a PyTorch reference on the RTX host.
- Full-model logits must reach cosine similarity `>= 0.999`.
- Greedy output must match Transformers for 32 tokens across the fixed prompt corpus.
- Cache tests cover 15/16/17 and 31/32/33 token boundaries, exhaustion, cancellation, and reuse.
- CUDA tests must run under Compute Sanitizer before a release is tagged.

The portable tests already exercise SHA-256, model-container integrity, cache reservations,
fragmentation, FCFS lifecycle, EOS completion, and cancellation. CUDA numerical tests and measured
results must be produced on the NVIDIA host; they are not fabricated on the Mac development host.
