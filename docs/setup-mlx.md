# Apple Silicon MLX setup and foundation validation

The validated development host is an Apple M3 Max running native arm64 Python 3.12. The MLX wheel
requires Apple Silicon and Metal access; a sandboxed or virtualized shell may import the package but
still be unable to create a Metal device.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[mlx,test,export]'
```

Confirm that Python is arm64 and MLX can see Metal:

```bash
python - <<'PY'
import platform
from importlib.metadata import version
import mlx.core as mx

assert platform.machine() == "arm64"
print("MLX", version("mlx"))
print(mx.device_info())
PY
```

## Portable and Metal-backed tests

```bash
PYTHONPATH=python python -m pytest -q \
  tests/test_format.py \
  tests/test_model_file_reader.py \
  tests/test_export.py \
  tests/test_runtime.py \
  tests/test_benchmark.py

PYTHONPATH=python python -m pytest -q \
  tests/test_mlx_backend.py \
  tests/test_gemma3_backend.py
```

The first group has no Metal requirement. The second runs complete synthetic Qwen and Gemma models,
physical page boundaries, Transformers parity, custom-kernel/fallback differential execution,
mixed-length batched decode, and cancellation cleanup through MLX.

## Official checkpoint

The model is downloaded only for local validation and the `models/` directory is ignored by Git.

```bash
mkdir -p models benchmark-results
forge-export \
  Qwen/Qwen2.5-0.5B-Instruct \
  models/qwen2.5-0.5b.engine

forge-correctness \
  --backend mlx \
  --reference-device mps \
  --engine-model models/qwen2.5-0.5b.engine \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts benchmarks/prompts.json \
  --max-new-tokens 32 \
  --output benchmark-results/mlx-qwen-correctness.json
```

The reference command explicitly disables Qwen's configured repetition penalty so both engines are
compared using unmodified greedy argmax. PyTorch and Transformers are reference-only dependencies;
the MLX execution backend never imports them.

Gemma 3 1B uses the same exporter, artifact, engine, and correctness runner. Hugging Face may require
accepting Google's Gemma license before the first download.

```bash
forge-export \
  google/gemma-3-1b-it \
  models/gemma-3-1b-it.engine

forge-correctness \
  --backend mlx \
  --reference-device mps \
  --engine-model models/gemma-3-1b-it.engine \
  --model google/gemma-3-1b-it \
  --prompts benchmarks/prompts.json \
  --max-new-tokens 32 \
  --output benchmark-results/mlx-gemma3-correctness.json
```

The correctness command is intentionally strict and exits nonzero for any greedy-token divergence.
See the saved Gemma report for the measured FP16 near-tie rather than weakening this gate.

## Long-context gate

```bash
forge-validate-long-context \
  --model models/qwen2.5-0.5b.engine \
  --output benchmark-results/mlx-long-context-ladder.json \
  --context-lengths 2048 4096 8192 16384 32768 \
  --output-tokens 2 \
  --prefill-chunk-size 512
```

This generates deterministic token IDs locally, emits one raw observation per context length, and
fails on an incorrect prefill/decode step count or any reservation/page leak. Two output tokens force
one paged-attention decode iteration after the first token produced by prefill.

Gemma's wider K/V heads require more than the default 512 MiB pool at 32K. The validated command is:

```bash
forge-validate-long-context \
  --model models/gemma-3-1b-it.engine \
  --output benchmark-results/mlx-gemma3-32k.json \
  --context-lengths 32768 \
  --output-tokens 2 \
  --prefill-chunk-size 512 \
  --kv-cache-mib 1024
```

## Attention profile and optional GPU trace

```bash
forge-profile-attention \
  --model models/qwen2.5-0.5b.engine \
  --output benchmark-results/mlx-attention-ab.json \
  --key-lengths 512 2048 8192 --query-tokens 512

MTL_CAPTURE_ENABLED=1 forge-profile-attention \
  --model models/qwen2.5-0.5b.engine \
  --output benchmark-results/mlx-attention-capture.json \
  --key-lengths 512 --capture profiles/mlx-fused-attention.gputrace
```

Open the ignored `.gputrace` bundle in Xcode. Without `MTL_CAPTURE_ENABLED=1`, Metal reports that
the capture layer was not inserted.

## Benchmark smoke test

```bash
forge-bench \
  --backend mlx \
  --model models/qwen2.5-0.5b.engine \
  --prompts benchmarks/prompts.json \
  --output benchmark-results/mlx-smoke.json \
  --output-length 8 \
  --warmups 1 \
  --repetitions 1 \
  --max-sequences 4
```

This only checks the end-to-end API and result schema. Do not publish a one-warmup/one-trial smoke
run as a performance result; use the controlled matrix in `docs/benchmarking.md`.

Use `--mlx-kernel-mode baseline`, `fused`, or `full` for kernel ablations. Capture one end-to-end
measured repetition with:

```bash
MTL_CAPTURE_ENABLED=1 forge-bench \
  --backend mlx --mlx-kernel-mode full \
  --model models/qwen2.5-0.5b.engine \
  --prompts benchmark-results/workloads/p128-c1.json \
  --output benchmark-results/mlx-trace-p128-c1.json \
  --output-length 8 --warmups 1 --repetitions 1 \
  --capture profiles/mlx-p128-c1.gputrace
```

Open the ignored bundle in Xcode. Do not report the capture-instrumented latency as normal runtime
performance.
