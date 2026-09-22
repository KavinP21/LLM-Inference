# Forge LLM

Forge LLM is a focused inference runtime for `Qwen/Qwen2.5-0.5B-Instruct`. It now has two execution
paths: a validated MLX backend for Apple Silicon and an implemented CUDA backend awaiting RTX
validation. Neither path calls PyTorch during inference. Both consume the same versioned,
checksummed model artifact and expose the same request-oriented Python API.

See [STATUS.md](STATUS.md) for the exact validated/unvalidated boundary before treating this as a
release-ready engine.

The MLX path is the current development target. It executes the complete Qwen2 decoder, maintains
autoregressive K/V state, implements reservation-aware request admission, and has been checked
against the official Transformers model on an M3 Max. It is deliberately a correctness baseline:
its cache arrays are contiguous and active requests execute sequentially inside an iteration.

## What is implemented

- A deterministic, checksummed, 256-byte-aligned model container and Hugging Face exporter.
- Strict Qwen2 tensor and configuration validation.
- A safe Python memory-mapped reader sharing the C++ model-container contract.
- A full FP16 Qwen2 MLX execution path with cached prefill/decode.
- A backend-neutral Python scheduler and KV-capacity accounting layer.
- Automatic or explicit `mlx`/`cuda` backend selection through one Python API.
- RAII CUDA activation/workspace arenas and cached cuBLASLt execution context.
- Qwen2 FP16 execution with FP32 RMSNorm, softmax, and GEMM accumulation.
- Fused residual/RMSNorm, RoPE/KV scatter, and SwiGLU CUDA kernels.
- A 16-token paged GQA decode-attention kernel.
- Full-prompt Q/K/V and MLP GEMMs with causal paged prefill attention.
- Reservation-aware KV admission, lazy physical block allocation, and fragmentation metrics.
- FCFS iteration-level continuous batching with cancellation and immediate reclamation.
- A pybind11 API, reproducible JSON benchmark runner, portable host tests, and CI.

Not yet implemented on MLX: physical paged K/V storage, batched tensor execution, tiled/FlashAttention-
style prefill, custom Metal kernels, 32K-safe attention, quantization, probabilistic sampling,
prefix caching, HTTP serving, or distributed execution.

## Development environments

MLX execution targets Apple Silicon and is currently validated on an M3 Max. Host-only components
build on macOS and Linux. CUDA execution targets Ubuntu 24.04 under WSL2 on an RTX 3070 Ti (SM 8.6).

See [the MLX setup and validation runbook](docs/setup-mlx.md) for Apple Silicon and
[the WSL2 runbook](docs/setup-wsl.md) for the NVIDIA host.

```bash
# Portable C++ host components
cmake -S . -B build -G Ninja -DFORGE_ENABLE_CUDA=OFF -DFORGE_BUILD_PYTHON=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# Apple Silicon MLX backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[mlx,test,export]'
PYTHONPATH=python pytest -q tests/test_mlx_backend.py

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
from forge_llm import create_engine

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokens = tokenizer("Explain paged KV caching:").input_ids
engine = create_engine("models/qwen2.5-0.5b.engine", backend="mlx",
                       max_num_sequences=16, max_model_length=2048,
                       kv_cache_bytes=512 << 20)
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
  --backend mlx \
  --model models/qwen2.5-0.5b.engine \
  --prompts benchmarks/prompts.json \
  --output benchmark-results/c4-prompt-mix.json \
  --output-length 32 --warmups 5 --repetitions 3
```

The result contains raw request observations, TTFT/TPOT/end-to-end percentiles, throughput, engine
cache statistics, and a hardware/software manifest. The current MLX numbers characterize an
unoptimized correctness baseline and are not competitive-performance claims. See
[the benchmark protocol](docs/benchmarking.md) and [runtime architecture](docs/architecture.md).

## Correctness gates

- Every backend operation is compared with an independent numerical reference.
- Full-model logits must reach cosine similarity `>= 0.999`.
- Greedy output must match Transformers for 32 tokens across the fixed prompt corpus.
- Cache tests cover 15/16/17 and 31/32/33 token boundaries, exhaustion, cancellation, and reuse.
- CUDA tests must run under Compute Sanitizer before a release is tagged.

The portable tests exercise SHA-256, model-container integrity, cache reservations, fragmentation,
FCFS lifecycle, EOS completion, and cancellation. MLX tests additionally cover a complete synthetic
Qwen model, cached decode, and the public engine lifecycle. CUDA numerical tests and measurements
must still be produced on the NVIDIA host; they are not inferred from the Apple results.
