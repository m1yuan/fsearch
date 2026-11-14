#!/bin/sh
# Build the sidecar and install it as a per-user launchd agent.
# Run on the Mac, after the model directory exists.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/ane-embed"
swift build -c release
BIN="$ROOT/ane-embed/.build/release/ane-embed"
DEST="${HOME}/.local/bin/ane-embed"
mkdir -p "${HOME}/.local/bin"
cp "$BIN" "$DEST"
LABEL="com.fsearch.ane-embed"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG="${HOME}/Library/Application Support/FSearch/ane-embed.log"
mkdir -p "$(dirname "$LOG")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${DEST}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>${LOG}</string>
  <key>StandardErrorPath</key><string>${LOG}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>FSEARCH_EMBED_PORT</key><string>8765</string>
  </dict>
</dict>
</plist>
EOF
DOMAIN="gui/$(id -u)"
launchctl bootout "${DOMAIN}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"
echo "installed ${DEST}"
echo "LaunchAgent ${LABEL} listens on 127.0.0.1:8765"
echo "log: ${LOG}"
echo "first load specializes the model for this chip and can take a few minutes"
