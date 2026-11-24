#!/usr/bin/env bash
# Instinct on Linux or WSL2 with an NVIDIA GPU. See docs/nvidia.md.
#
#   tools/nvidia/instinct.sh setup    once: Python environment, MLX, models
#   tools/nvidia/instinct.sh check    the GPU, MLX and the CUDA kernels
#   tools/nvidia/instinct.sh start    pick models for the card, open the app
#
# start takes --workspace DIR (default ~/Instinct) and --port N (default
# 8765). Settings, API keys, invites and projects live in ~/.local/share/instinct.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
venv="$root/.venv"
python="$venv/bin/python"
home="${XDG_DATA_HOME:-$HOME/.local/share}/instinct"
export PYTHONPATH="$root/python"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

fail() { echo "$*" >&2; exit 1; }

require_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1 && [ ! -x /usr/lib/wsl/lib/nvidia-smi ]; then
    fail "No NVIDIA driver found. On WSL2, install the current NVIDIA driver in Windows (not in Linux) and run 'wsl --update' from Windows."
  fi
}

require_glibc() {
  # MLX's CUDA wheels are built for glibc 2.35 or newer (Ubuntu 22.04+).
  local version
  version=$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)
  if [ -n "$version" ]; then
    local major=${version%%.*} minor=${version##*.}
    if [ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -lt 35 ]; }; then
      fail "This Linux has glibc $version; MLX needs 2.35 or newer (Ubuntu 22.04 or later)."
    fi
  fi
}

require_env() {
  [ -x "$python" ] || fail "Run 'tools/nvidia/instinct.sh setup' first."
}

case "${1:-}" in
  setup)
    require_gpu
    require_glibc
    command -v python3 >/dev/null || fail "Install Python 3.10 or newer: sudo apt install python3 python3-venv"
    if [ ! -x "$python" ]; then
      python3 -m venv "$venv" || fail "Couldn't make a virtual environment: sudo apt install python3-venv"
    fi
    "$python" -m pip install --upgrade pip
    "$python" -m pip install -r tools/nvidia/requirements.txt
    read -r -a models <<<"$("$python" -m instinct.nvidia models)"
    "$python" -m instinct.prepare_models "${models[@]}"
    "$python" -m instinct.nvidia check >/dev/null && echo "Setup done. Start Instinct with: tools/nvidia/instinct.sh start" \
      || echo "Setup done, but a check failed; run 'tools/nvidia/instinct.sh check' to see which."
    ;;
  check)
    require_env
    exec "$python" -m instinct.nvidia check
    ;;
  start)
    require_env
    require_gpu
    shift
    workspace="$HOME/Instinct"
    port=8765
    while [ $# -gt 0 ]; do
      case "$1" in
        --workspace) workspace=${2:?}; shift 2 ;;
        --port) port=${2:?}; shift 2 ;;
        *) fail "unknown option: $1" ;;
      esac
    done
    mkdir -p "$workspace" "$home"
    chmod 700 "$home"
    "$python" -m instinct.nvidia profile --root "$root" --output "$home/nvidia.json"
    exec "$python" -m instinct.harness.cli \
      --config "$home/nvidia.json" \
      --home "$home" \
      --workspace "$workspace" \
      --port "$port" \
      --open
    ;;
  *)
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
