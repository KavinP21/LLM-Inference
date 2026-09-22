#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build}"
compute-sanitizer --tool memcheck --error-exitcode 1 "$build_dir/forge_cuda_tests"
compute-sanitizer --tool racecheck --error-exitcode 1 "$build_dir/forge_cuda_tests"

