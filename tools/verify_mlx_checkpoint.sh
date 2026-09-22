#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
model_path="${FORGE_MODEL_PATH:-models/qwen2.5-0.5b.engine}"

PYTHONPATH=python "${python_bin}" -m pytest -q \
  tests/test_format.py \
  tests/test_model_file_reader.py \
  tests/test_runtime.py \
  tests/test_benchmark.py

PYTHONPATH=python "${python_bin}" -m pytest -q tests/test_mlx_backend.py

if [[ -f "${model_path}" ]]; then
  mkdir -p benchmark-results
  PYTHONPATH=python "${python_bin}" -m forge_llm.correctness \
    --backend mlx \
    --reference-device mps \
    --engine-model "${model_path}" \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --prompts benchmarks/prompts.json \
    --max-new-tokens 32 \
    --output benchmark-results/mlx-qwen-correctness.json
else
  echo "Skipping official-model parity; export ${model_path} first."
fi
