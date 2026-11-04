#!/bin/bash
#SBATCH -J instinct-check
#SBATCH -N1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=224G
#SBATCH -t 0:45:00
#SBATCH -o instinct-check-%j.out
#
# Before any benchmark: does the engine work on this GPU? It stops at the
# first failure so a broken setup doesn't hold the H100.
#   1. MLX op checks (tools/pace/check_mlx.py)
#   2. The engine tests, which compare against NumPy references, plus the
#      real-Qwen runner tests
#   3. Real-model answers, cache reuse and first speeds (tools/pace/smoke.py)
#
#   sbatch tools/pace/check.sbatch      (after setup.sbatch has finished)
#
# 224G per GPU is PACE's recommended memory request for the HGX H100 nodes.

set -euo pipefail
source tools/pace/env.sh
out=$(results_dir check)
echo "Results go to $out"

nvidia-smi | tee "$out/nvidia-smi.txt"
in_container "$VENV/bin/pip" freeze > "$out/pip-freeze.txt"

py tools/pace/check_mlx.py --output "$out/mlx-ops.json"
py -m pytest -q -p no:cacheprovider \
  tests/test_mlx_engine.py tests/test_portable_kernels.py \
  tests/test_chat_runner_prompts.py tests/test_chat_runner.py \
  2>&1 | tee "$out/pytest.txt"
py tools/pace/smoke.py --output "$out/smoke.json"

echo "All checks passed. Next: sbatch tools/pace/bench.sbatch"
