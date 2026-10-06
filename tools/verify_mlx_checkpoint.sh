#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-.venv/bin/python}"
model_path="${FORGE_MODEL_PATH:-models/qwen2.5-0.5b.engine}"

PYTHONPATH=python "${python_bin}" -m pytest -q \
  tests/test_format.py \
  tests/test_model_file_reader.py \
  tests/test_runtime.py \
  tests/test_quantization.py \
  tests/test_int8_quality_metrics.py \
  tests/test_precision_policy.py \
  tests/test_second_order.py \
  tests/test_second_order_driver.py \
  tests/test_cached_policy.py tests/test_cached_driver.py tests/test_cached_final_audit.py \
  tests/test_scale_aware.py tests/test_scale_driver.py tests/test_scale_objective_readback.py tests/test_scale_final_audit.py \
  tests/test_refined.py tests/test_refined_driver.py tests/test_refined_objective_readback.py tests/test_refined_final_audit.py \
  tests/test_gpu_evidence_guard.py \
  tests/test_int8_batch_driver.py \
  tests/test_decode_policy.py \
  tests/test_batch_numerics_driver.py \
  tests/test_batch_evidence_audit.py \
  tests/test_benchmark.py

PYTHONPATH=python "${python_bin}" -m pytest -q tests/test_mlx_backend.py tests/test_gemma3_backend.py tests/test_int8_metal.py tests/test_batch_numerics_metal.py tests/test_second_order_metal.py tests/test_cached_metal.py tests/test_scale_metal.py tests/test_refined_metal.py

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
