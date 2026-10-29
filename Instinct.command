#!/bin/zsh
set -e
studio_root="${0:A:h}"
cd "$studio_root"
exec "$studio_root/.venv/bin/python" "$studio_root/tools/instinct.py"
