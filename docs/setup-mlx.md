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
  tests/test_runtime.py \
  tests/test_benchmark.py

PYTHONPATH=python python -m pytest -q tests/test_mlx_backend.py
```

The first group has no Metal requirement. The second runs a complete synthetic Qwen model and
cached decode through MLX.

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

This only checks the end-to-end API and result schema. Do not publish it as a performance result:
the foundation backend currently executes active sequences individually and uses untiled attention.

