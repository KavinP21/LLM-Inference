#!/usr/bin/env bash
# Starts the app inside the container. tools/docker/instinct.sh runs it with
# the workspace at /workspace, the app's state at /state and the sign-in
# token in $INSTINCT_APP_TOKEN.
set -euo pipefail

port="${INSTINCT_PORT:-8765}"

case "${INSTINCT_MODELS:-host}" in
  host)
    # Written by `python -m instinct.container serve-models` on the Mac.
    config=/state/host-models.json
    if [ ! -f "$config" ]; then
      echo "No $config: start Instinct with tools/docker/instinct.sh, which serves the Mac's models first." >&2
      exit 1
    fi
    mkdir -p "$HOME" /state/cache
    ;;
  nvidia)
    config=/state/nvidia.json
    mkdir -p "$HOME" /state/cache
    python -m instinct.nvidia profile --root /opt/instinct --output "$config"
    ;;
  *)
    echo "INSTINCT_MODELS must be host or nvidia" >&2
    exit 1
    ;;
esac

exec python -m instinct.harness.cli \
  --config "$config" \
  --home /state \
  --workspace /workspace \
  --port "$port" \
  --container
