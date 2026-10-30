#!/bin/zsh
# Run macos/ui-test.js inside Instinct.app against a fresh demo workspace.
# Results and window snapshots go to build/ui-test (or $INSTINCT_TEST_OUTPUT).
#   macos/test.sh [--no-build] [--models]
set -euo pipefail

root="${0:A:h:h}"
app="$root/build/Instinct.app/Contents/MacOS/Instinct"
out="${INSTINCT_TEST_OUTPUT:-$root/build/ui-test}"
models=0
for arg in "$@"; do
  case "$arg" in
    --no-build) skip_build=1 ;;
    --models) models=1 ;;
  esac
done
[[ -n "${skip_build:-}" && -x "$app" ]] || "$root/macos/build.sh" >/dev/null

rm -rf "$out"
mkdir -p "$out/workspace"
printf '# Fixture\n\nA small project for the UI test.\n' > "$out/workspace/README.md"
printf 'def add(a, b):\n    return a + b\n' > "$out/workspace/example.py"

INSTINCT_TEST_SCRIPT="$root/macos/ui-test.js" \
INSTINCT_TEST_OUTPUT="$out" \
INSTINCT_TEST_WORKSPACE="$out/workspace" \
INSTINCT_TEST_MODELS="$models" \
  "$app" &
pid=$!
( sleep 240; kill "$pid" 2>/dev/null ) &
watchdog=$!
code=0
wait "$pid" || code=$?
kill "$watchdog" 2>/dev/null || true

"$root/.venv/bin/python" - "$out/result.json" <<'PY'
import json, sys
try:
    results = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit("no results: the app did not finish the test")
for result in results:
    print(("ok   " if result["ok"] else "FAIL ") + result["name"] + ("" if result["ok"] else f": {result['error']}"))
PY
echo "snapshots: $out"
exit "$code"
