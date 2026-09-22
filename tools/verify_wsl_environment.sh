#!/usr/bin/env bash
set -euo pipefail

required=(nvidia-smi nvcc cmake ninja python3 compute-sanitizer nsys ncu)
failed=0
for command_name in "${required[@]}"; do
  if command -v "$command_name" >/dev/null 2>&1; then
    echo "ok: $command_name -> $(command -v "$command_name")"
  else
    echo "missing: $command_name"
    failed=1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
fi

if [[ "$failed" -ne 0 ]]; then
  echo "Install the missing WSL2/CUDA development tools before building the CUDA backend." >&2
  exit 1
fi

