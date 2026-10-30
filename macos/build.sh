#!/bin/zsh
# Build Instinct.app into build/. Pass --install to copy it to ~/Applications.
#
# The app runs the Python environment (.venv) and models of this checkout, so
# build it from the checkout you want it to use.
set -euo pipefail

root="${0:A:h:h}"
out="$root/build/Instinct.app"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

rm -rf "$out"
mkdir -p "$out/Contents/MacOS" "$out/Contents/Resources"

xcrun swiftc -O -swift-version 5 \
  -target "$(uname -m)-apple-macos14.0" \
  -framework Cocoa -framework WebKit \
  -o "$out/Contents/MacOS/Instinct" "$root/macos/Instinct.swift"

sed "s|@ROOT@|$root|" "$root/macos/Info.plist" > "$out/Contents/Info.plist"

"$root/.venv/bin/python" "$root/macos/make_icon.py" "$work/AppIcon.iconset"
iconutil -c icns -o "$out/Contents/Resources/AppIcon.icns" "$work/AppIcon.iconset"

# Ad-hoc signature so Gatekeeper runs a locally built app.
codesign --force --sign - "$out"

if [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/Instinct.app"
  cp -R "$out" "$HOME/Applications/"
  echo "Installed $HOME/Applications/Instinct.app"
else
  echo "Built $out"
fi
