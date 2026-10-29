#!/bin/bash
# Builds "Forge LLM.app" in the repository root. Double-clicking it runs
# tools/forge_chat.py with the repository's .venv.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
app="$root/Forge LLM.app"

rm -rf "$app"
mkdir -p "$app/Contents/MacOS"
swiftc -O "$root/tools/ForgeLauncher.swift" -o "$app/Contents/MacOS/Forge LLM"

cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Forge LLM</string>
    <key>CFBundleExecutable</key>
    <string>Forge LLM</string>
    <key>CFBundleIdentifier</key>
    <string>local.forge-llm.chat</string>
    <key>CFBundleName</key>
    <string>Forge LLM</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
</dict>
</plist>
PLIST

# Ad-hoc sign so Gatekeeper treats it like any other locally built app.
codesign --force --sign - "$app"
echo "Built $app"
