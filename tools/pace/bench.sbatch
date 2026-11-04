#!/bin/bash
#SBATCH -J instinct-bench
#SBATCH -N1 --ntasks=1 --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem-per-gpu=224G
#SBATCH --tmp=60G
#SBATCH -t 4:00:00
#SBATCH -o instinct-bench-%j.out
#
# Benchmarks on one H100, after check.sbatch has passed:
#   1. Engine level: 1-16 users decoding together while an 8K prompt arrives
#      (benchmarks/multi_user.py), for Gemma 4 and Qwen3.5.
#   2. Over HTTP: concurrent multi-turn chats through the /v1 API of one
#      worker per model (benchmarks/serve_load.py).
#   3. Long context: a LONG_CONTEXT-token prompt (default 131072), then a
#      follow-up turn on top of it (benchmarks/long_context.py).
# Results, the worker logs and a GPU utilization trace go to
# benchmark-results/pace-bench-<job id>/. The job ends when the last
# benchmark does.
#
#   sbatch tools/pace/bench.sbatch
#   sbatch --export=ALL,LONG_CONTEXT=262144,USERS=1,4,16 tools/pace/bench.sbatch
#
# The workers listen on 127.0.0.1 only, on a free port, behind a random
# token made for this job and never printed. They stop with the job.

set -euo pipefail
source tools/pace/env.sh
out=$(results_dir bench)
users=${USERS:-1,2,4,8,16}
long=${LONG_CONTEXT:-131072}
echo "Results go to $out"

# Models on the node's local NVMe load faster than from Lustre scratch, and
# the job's local disk is cleared when it ends.
local_models=$TMPDIR/models
mkdir -p "$local_models"
cp -r "$MODELS/gemma-4-26b-a4b-mlx" "$MODELS/qwen3.5-4b-mlx" "$local_models/"
gemma=$local_models/gemma-4-26b-a4b-mlx
qwen=$local_models/qwen3.5-4b-mlx

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
  --format=csv -l 5 > "$out/gpu.csv" &
monitor=$!
worker=
cleanup() {
  [ -n "$worker" ] && kill "$worker" 2>/dev/null || true
  kill "$monitor" 2>/dev/null || true
}
trap cleanup EXIT

# 1. Engine level.
for model in "$gemma" "$qwen"; do
  name=$(basename "$model")
  for n in ${users//,/ }; do
    py benchmarks/multi_user.py "$model" --users "$n" \
      --output "$out/multi-user-$name-$n.json"
  done
done

# 2. Over HTTP.
serve() {
  local name=$1 model=$2
  INSTINCT_WORKER_TOKEN=$(head -c 32 /dev/urandom | base64 | tr -d "/+=\n")
  export INSTINCT_WORKER_TOKEN
  py_background -m instinct.agents.worker --model "$model" --tokenizer "$model" \
    --backend mlx --runner chat --max-model-length 65536 \
    --kv-cache-bytes $((40 << 30)) --max-pending 16 \
    --host 127.0.0.1 --port 0 > "$out/worker-$name.log" 2>&1 &
  worker=$!
  local port=
  for _ in $(seq 600); do
    port=$(sed -n 's/.*"event": "ready".*"port": \([0-9]*\).*/\1/p' "$out/worker-$name.log")
    [ -n "$port" ] && break
    kill -0 "$worker" 2>/dev/null || { echo "worker exited; see worker-$name.log" >&2; return 1; }
    sleep 1
  done
  [ -n "$port" ] || { echo "worker didn't start in 10 minutes" >&2; return 1; }
  py benchmarks/serve_load.py --url "http://127.0.0.1:$port" --users "$users" \
    --turns 3 --document-words 1500 --max-tokens 256 \
    --output "$out/serve-load-$name.json"
  kill "$worker"
  wait "$worker" 2>/dev/null || true
  worker=
  unset INSTINCT_WORKER_TOKEN
}
serve gemma-4 "$gemma"
serve qwen3.5 "$qwen"

# 3. Long context.
py benchmarks/long_context.py "$gemma" --tokens "$long" \
  --output "$out/long-context-gemma-4-$long.json"

echo "Done. Summaries:"
py tools/pace/summary.py "$out" | tee "$out/summary.md"
