#!/usr/bin/env bash
# Instinct in a Docker sandbox. See docs/docker.md.
#
#   tools/docker/instinct.sh build            build the image this machine uses
#   tools/docker/instinct.sh start [options]  start the sandbox, open the app
#
# start takes --workspace DIR (default ~/Instinct), the only host folder the
# container can change, --port N (default 8765) and --no-browser (print the
# sign-in link instead of opening it).
#
# On a Mac the models run natively (Docker can't reach the Apple GPU) and the
# app, agents and their commands run in the container. On Linux or WSL2 with
# an NVIDIA GPU, everything runs in the container with the GPU passed through.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

fail() { echo "$*" >&2; exit 1; }

if [ "$(uname -s)" = Darwin ]; then
  mode=host
  target=app
  state="$HOME/Library/Application Support/Instinct/docker"
else
  mode=nvidia
  target=nvidia
  state="${XDG_DATA_HOME:-$HOME/.local/share}/instinct-docker"
fi
image="instinct:$target"

require_docker() {
  command -v docker >/dev/null 2>&1 || fail "Install Docker first: https://docs.docker.com/get-docker/"
  docker info >/dev/null 2>&1 || fail "Docker isn't running. Start Docker Desktop (or the Docker service) and try again."
}

build() {
  require_docker
  docker build -f docker/Dockerfile --target "$target" -t "$image" .
}

open_url() {
  # The sign-in link carries a one-time token after the #, which browsers
  # never send to a server.
  if [ "$mode" = host ]; then
    open "$1"
  elif grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    (cd /mnt/c 2>/dev/null && cmd.exe /c start "" "$1" >/dev/null 2>&1) || return 1
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1" >/dev/null 2>&1
  else
    return 1
  fi
}

start() {
  require_docker
  local workspace="$HOME/Instinct" port=8765 browser=1
  while [ $# -gt 0 ]; do
    case "$1" in
      --workspace) workspace=${2:?}; shift 2 ;;
      --port) port=${2:?}; shift 2 ;;
      --no-browser) browser=0; shift ;;
      *) fail "unknown option: $1" ;;
    esac
  done
  mkdir -p "$workspace" "$state"
  chmod 700 "$state"
  workspace=$(cd "$workspace" && pwd)
  docker image inspect "$image" >/dev/null 2>&1 || build

  # Fresh credentials each run, handed over through the environment so they
  # never appear on a command line.
  INSTINCT_APP_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  export INSTINCT_APP_TOKEN
  local name="instinct-sandbox-$port" models_pid="" run=()
  run=(
    --name "$name" --rm --detach
    --read-only
    --tmpfs /tmp:rw,nosuid,nodev,size=2g
    --cap-drop ALL
    --security-opt no-new-privileges
    --pids-limit 1024
    --publish "127.0.0.1:$port:$port"
    --volume "$workspace:/workspace"
    --volume "$state:/state"
    --env INSTINCT_APP_TOKEN
    --env "INSTINCT_PORT=$port"
  )

  cleanup() {
    docker stop --time 20 "$name" >/dev/null 2>&1 || true
    if [ -n "$models_pid" ]; then
      kill "$models_pid" 2>/dev/null || true
      wait "$models_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT
  trap 'exit 130' INT TERM HUP

  if [ "$mode" = host ]; then
    local python="$root/.venv/bin/python"
    [ -x "$python" ] || fail "No Python environment at $python. Follow the install steps in docs/app.md first."
    if pgrep -x Instinct >/dev/null 2>&1; then
      fail "Quit the Instinct app first: it loads the same models, and both together don't fit in memory."
    fi
    INSTINCT_WORKER_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
    export INSTINCT_WORKER_TOKEN
    PYTHONPATH="$root/python" "$python" -m instinct.container serve-models \
      --config "${INSTINCT_HOST_CONFIG:-examples/instinct.json}" \
      --output "$state/host-models.json" &
    models_pid=$!
    echo "Loading the models natively (this takes a minute)..."
    until [ -f "$state/host-models.json" ]; do
      kill -0 "$models_pid" 2>/dev/null || fail "The models didn't start."
      sleep 1
    done
    run+=(--env INSTINCT_WORKER_TOKEN --add-host host.docker.internal:host-gateway)
  else
    command -v nvidia-smi >/dev/null 2>&1 || [ -x /usr/lib/wsl/lib/nvidia-smi ] \
      || fail "No NVIDIA driver found."
    [ -d "$root/models" ] || fail "No models folder. Download them first: tools/nvidia/instinct.sh setup"
    run+=(
      --gpus all
      --user "$(id -u):$(id -g)"
      --volume "$root/models:/opt/instinct/models:ro"
    )
  fi

  docker run "${run[@]}" "$image" >/dev/null
  echo "Starting Instinct in Docker..."
  local link="http://127.0.0.1:$port/#token=$INSTINCT_APP_TOKEN"
  for _ in $(seq 1 600); do
    if ! docker container inspect "$name" >/dev/null 2>&1; then
      fail "The container stopped while starting."
    fi
    if curl -fs -o /dev/null "http://127.0.0.1:$port/"; then
      if [ "$browser" = 1 ] && open_url "$link"; then
        echo "Opened Instinct in your browser."
      else
        echo "Open this link on this computer to sign in (it stops working when Instinct stops):"
        echo "$link"
      fi
      echo "Workspace: $workspace. Press Ctrl-C to stop."
      # In the background, so Ctrl-C (or a closed terminal) reaches the
      # trap at once instead of after the log stream ends.
      docker logs --follow "$name" 2>&1 | grep -v -- "#token=" &
      wait $! || true
      return 0
    fi
    sleep 1
  done
  fail "Instinct didn't start; see: docker logs $name"
}

case "${1:-}" in
  build) build ;;
  start) shift; start "$@" ;;
  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
