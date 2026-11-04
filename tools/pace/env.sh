# Shared settings for the PACE ICE jobs in this folder. Sourced, not run.
#
# Everything big lives on scratch (300 GB), not in home (30 GB): the container
# image, the Python environment, the models and every download cache.
# Override the location with INSTINCT_PACE_DATA.

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Run this inside a Slurm job (sbatch or salloc). PACE login nodes are" \
    "for editing and submitting jobs, not for computation." >&2
  exit 1
fi

REPO=$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && pwd)
if [ ! -f "$REPO/python/instinct/__init__.py" ]; then
  echo "Submit from the repository's top folder: sbatch tools/pace/<job>.sbatch" >&2
  exit 1
fi

mkdir -p "${INSTINCT_PACE_DATA:-$HOME/scratch/instinct}"
DATA=$(readlink -f "${INSTINCT_PACE_DATA:-$HOME/scratch/instinct}")
IMAGE=$DATA/python-3.12-bookworm.sif
VENV=$DATA/venv
MODELS=$DATA/models

export APPTAINER_CACHEDIR=$DATA/cache/apptainer
export PIP_CACHE_DIR=$DATA/cache/pip
export HF_HOME=$DATA/cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME=$DATA/cache/xdg
export MLX_PTX_CACHE_DIR=$DATA/cache/mlx-ptx
export TOKENIZERS_PARALLELISM=false
# The venv already ignores ~/.local; this covers anything run outside it.
export PYTHONNOUSERSITE=1
export PYTHONPATH=$REPO/python
mkdir -p "$APPTAINER_CACHEDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" \
  "$MLX_PTX_CACHE_DIR"

if ! command -v apptainer >/dev/null 2>&1; then
  module load apptainer >/dev/null 2>&1 || true
fi

# Run a command inside the container: Debian 12, whose glibc (2.36) MLX's
# Linux wheels need. RHEL 9's is 2.34. Home and /storage are mounted by
# default; scratch and the job's local disk are bound explicitly.
container_flags() {
  CONTAINER_FLAGS=(--bind "$DATA" --pwd "$REPO")
  if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ]; then
    CONTAINER_FLAGS+=(--bind "$TMPDIR")
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    CONTAINER_FLAGS+=(--nv)
  fi
}

in_container() {
  container_flags
  apptainer exec "${CONTAINER_FLAGS[@]}" "$IMAGE" "$@"
}

# For "py_background ... &": exec replaces the subshell bash forks for a
# backgrounded function, so $! is the process itself and kill reaches it.
py_background() {
  container_flags
  exec apptainer exec "${CONTAINER_FLAGS[@]}" "$IMAGE" "$VENV/bin/python" "$@"
}

py() {
  in_container "$VENV/bin/python" "$@"
}

results_dir() {
  local dir=$REPO/benchmark-results/pace-$1-$SLURM_JOB_ID
  mkdir -p "$dir"
  echo "$dir"
}
