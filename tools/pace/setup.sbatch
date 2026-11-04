#!/bin/bash
#SBATCH -J instinct-setup
#SBATCH -N1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o instinct-setup-%j.out
#
# Builds the environment and downloads the models (about 19 GB) to scratch.
# CPU only: none of this needs a GPU. PACE also asks that container pulls and
# large downloads stay off the login nodes.
#
#   sbatch tools/pace/setup.sbatch      (from the repository's top folder)

set -euo pipefail
source tools/pace/env.sh

echo "Data folder: $DATA"
if [ ! -f "$IMAGE" ]; then
  apptainer pull "$IMAGE" docker://python:3.12-slim-bookworm
fi
if [ ! -x "$VENV/bin/python" ]; then
  in_container python3 -m venv "$VENV"
fi
in_container "$VENV/bin/pip" install --quiet --upgrade pip
in_container "$VENV/bin/pip" install --quiet -r tools/pace/requirements.txt

# Gemma 4 26B A4B (QAT 4-bit) and Qwen3.5-4B, at pinned revisions.
py -m instinct.prepare_models --root "$DATA"
if [ ! -e "$REPO/models" ]; then
  ln -s "$MODELS" "$REPO/models"
fi

py -c "import importlib.metadata as m; print('mlx', m.version('mlx'), 'ready')"
du -sh "$DATA"/* 2>/dev/null || true
echo "Next: sbatch tools/pace/check.sbatch"
