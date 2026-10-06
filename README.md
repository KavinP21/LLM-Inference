# Forge LLM

Forge LLM is a focused, multi-family inference runtime. The validated Apple Silicon path supports
`Qwen/Qwen2.5-0.5B-Instruct` and the text-only `google/gemma-3-1b-it`; the CUDA Qwen backend is
implemented and still awaits RTX validation. No execution backend calls PyTorch during inference.
They consume one versioned, checksummed model artifact and expose the same request-oriented Python
API.

See [STATUS.md](STATUS.md) for the exact validated/unvalidated boundary before treating this as a
release-ready engine. The current 32K acceptance evidence is in
[the MLX paged validation report](docs/mlx-32k-results.md), and the custom-kernel evidence is in
[the MLX Metal optimization report](docs/mlx-metal-results.md). Gemma architecture, parity, and 32K
evidence is recorded in [the Gemma 3 validation report](docs/mlx-gemma3-results.md).

The MLX path executes complete Qwen2 and Gemma 3 text decoders, maintains autoregressive state in
physical 16-token K/V pages, and implements reservation-aware admission and continuous batching.
Long prompts are scheduler-visible 512-token chunks, runnable decodes share batched projections and
attention, and both official models have executed through 32,768 tokens. Gemma sliding layers gather
only their live receptive field while every sixth layer retains global attention. The optimized path
uses custom Metal RoPE/page-write and paged GQA decode kernels for both families, plus Qwen-specific
residual/RMSNorm and SwiGLU fusions, while retaining an independently selectable MLX fallback.

## What is implemented

- A deterministic, checksummed, 256-byte-aligned model container and Hugging Face exporter.
- Version-2 multi-family metadata with backward-compatible version-1 Qwen loading.
- Experimental version-3 weight-only INT8 artifacts, per-output-channel FP32 scales, and a custom
  Metal small-batch linear kernel for both families; see [quantization](docs/quantization.md).
- Separate-corpus precision calibration, explicit mixed FP16/INT8 policies, and an accuracy-oriented
  reconstruction mode; [numerical gates improve but greedy gates still fail](docs/mlx-int8-hardening-results.md).
- Activation-calibrated block-diagonal INT8 error compensation and joint precision selection,
  with frozen-before-regression statistics, strict derivation checks, and same-partition RTN controls;
  [the new experiment has mixed quality results and remains uncertified](docs/mlx-second-order-results.md).
- Actual cached-decode activation/probe coverage, bounded full-calibration policy repair, and
  all-position numerical regression; [the cached checkpoint remains experimental](docs/mlx-cached-calibration-results.md).
- Activation-weighted scale selection with whole-row reconstruction fallback, backward-compatible
  packed execution, and a pre-registered stop-before-held-out calibration stage;
  see [the contract](docs/scale-aware-quantization.md) and
  [the completed calibration results (exact-token gates failed)](docs/mlx-scale-aware-results.md).
- Bounded integer-coordinate refinement with fixed fitted scales, fresh calibration, complete-row
  fallback, and independent coefficient/objective readback; see [the contract](docs/refined-quantization.md)
  and [results (Qwen calibration passes; Gemma numerical gate fails)](docs/mlx-refined-results.md).
- A separate, sealed Qwen-only downstream validation driver with all-position held-out checks,
  native/composed byte controls, immutable evidence, and quality-gated resource/performance stages;
  see [the downstream contract](docs/qwen-validation.md). Direct Metal and Gemma cannot inherit
  native Qwen acceptance; [the held-out exact-token gate fails](docs/mlx-qwen-validation-results.md).
  The [delivery roadmap](docs/roadmap.md) counts the remaining major milestones and resumes independent
  FP16 prefix-cache work without weakening quantization gates.
- Opt-in FP16 MLX prefix caching with immutable shared pages, reference-counted ownership,
  partial-tail copy-on-write, namespace-scoped reuse, bounded LRU eviction and safe admission;
  see [the contract and API](docs/prefix-cache.md). [All registered validation gates pass](docs/mlx-prefix-cache-results.md);
  timing/final qualification is deferred at the user's [safe stopping point](docs/prefix-cache-handoff.md).
  Original FP16/app defaults remain unchanged.
- Strict, backend-independent Qwen2 and Gemma 3 tensor/configuration validation.
- A safe Python memory-mapped reader sharing the C++ model-container contract.
- Full FP16 Qwen2 and Gemma 3 MLX execution paths with physical paged K/V storage.
- A model-adapter factory that keeps request orchestration independent of decoder architecture.
- Bounded-workspace fused GQA attention plus an exact online-softmax tiled reference path.
- Shape-specialized custom Metal transformer fusions and direct block-table paged decode attention.
- Chunked prefill to 32K and batched one-token decode across all runnable requests.
- Explicit MLX decode numerical policy: unchanged `batched` default or optional single-row
  projections with batched attention; see [the contract](docs/decode-numerics.md).
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

Not yet implemented on MLX: 4-bit quantization, probabilistic sampling, speculative decoding, HTTP serving, or
distributed execution. The custom attention kernel reads a packed physical-page tensor through
per-request block tables, but MLX arrays are immutable and the Python API cannot expose independent
page allocations as one pointer-addressable pool. The runtime therefore stacks live layer-pages
before each attention launch. This removes logical-history gathering and right-padding, but the
page-pack remains measurable overhead, especially for one long request.

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
forge-export google/gemma-3-1b-it models/gemma-3-1b-it.engine
forge-inspect-model models/qwen2.5-0.5b.engine

python - <<'PY'
from transformers import AutoTokenizer
from forge_llm import create_engine

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokens = tokenizer("Explain paged KV caching:").input_ids
engine = create_engine("models/qwen2.5-0.5b.engine", backend="mlx",
                       max_num_sequences=16, max_model_length=32768,
                       kv_cache_bytes=512 << 20, prefill_chunk_size=512)
output = engine.generate(tokens, max_new_tokens=64,
                         eos_token_ids=[tokenizer.eos_token_id])
print(tokenizer.decode(output))
PY
```

The asynchronous interface consists of `submit`, `step`, `cancel`, and `stats`. Each `step` returns
token events for active requests, allowing new requests to join between decode iterations.

For repeated prompts, opt into MLX prefix reuse with `prefix_cache_bytes` and
`prefix_cache_max_entries`. The retention budget is within the total KV budget; use
`decode_mode="rowwise"` for the qualified exact numerical policy. `submit`/`generate` accept
`cache_namespace`, and `clear_prefix_cache()` releases retained cache pins without cancelling
active requests. Cached pages intentionally survive request completion. Full API examples,
32K retention sizing and limitations are in [the prefix-cache contract](docs/prefix-cache.md).

## Minimal Mac desktop app

The repository includes a small native-feeling Tk interface for the Apple MLX backend. Keep the
app bundle inside this checkout because it launches the checkout's `.venv` and model artifact.
Double-click [Forge LLM.app](../customLLMInferenceEngine/Forge%20LLM.app) in Finder, wait for the
status to become `Ready`, enter a prompt, and press `Generate`.

The UI source is [tools/forge_chat.py](tools/forge_chat.py). It currently selects the Qwen artifact
and tokenizer explicitly, keeps the model loaded between
requests, uses the model's native 32,768-token context with 512-token chunked prefill and paged KV
storage, and reports loading or inference errors in the window instead of requiring a terminal.

## Benchmark

```bash
forge-bench \
  --backend mlx \
  --model models/qwen2.5-0.5b.engine \
  --prompts benchmarks/prompts.json \
  --output benchmark-results/c4-prompt-mix.json \
  --output-length 32 --warmups 5 --repetitions 3 \
  --mlx-kernel-mode full
```

The result contains raw request observations, TTFT/TPOT/end-to-end percentiles, throughput, engine
cache statistics, and a hardware/software manifest. Validate the long-context ladder and collect a
controlled attention A/B profile with:

```bash
forge-validate-long-context --model models/qwen2.5-0.5b.engine \
  --output benchmark-results/mlx-long-context.json --output-tokens 2
forge-profile-attention --model models/qwen2.5-0.5b.engine \
  --output benchmark-results/mlx-attention-ab.json
```

Use `--mlx-kernel-mode baseline`, `fused`, and `full` for controlled comparisons. The current MLX
numbers characterize one M3 Max development host and are not cross-platform claims. See
[the benchmark protocol](docs/benchmarking.md) and [runtime architecture](docs/architecture.md).

## Correctness gates

- Every backend operation is compared with an independent numerical reference.
- Full-model logits must reach cosine similarity `>= 0.999`.
- Greedy output must match Transformers for 32 tokens across the fixed prompt corpus.
- Cache tests cover 15/16/17 and 31/32/33 token boundaries, exhaustion, cancellation, and reuse.
- CUDA tests must run under Compute Sanitizer before a release is tagged.

The portable tests exercise SHA-256, both artifact versions, both family contracts, local Gemma
export, physical block reuse, cache
reservations, fragmentation, FCFS lifecycle, EOS completion, and cancellation. MLX tests
additionally cover complete synthetic Qwen and Gemma models, Transformers parity, chunked and
sliding-window prefill, batched decode, custom/fallback differential execution, physical-page
reclamation, and all 15/16/17/31/32/33 boundaries. CUDA numerical tests and measurements must still
be produced on the NVIDIA host; they are not inferred from the Apple results.
