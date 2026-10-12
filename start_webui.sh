#!/bin/bash
# Start the WebUI in the background (nohup) and print how to reach it.
set -e
WEBUI_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8099}"
mkdir -p "$WEBUI_DIR/logs"
cd "$WEBUI_DIR"
nohup env PORT="$PORT" HOST="${HOST:-0.0.0.0}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
  sh "$WEBUI_DIR/run_webui.sh" > "$WEBUI_DIR/logs/webui.log" 2>&1 &
echo "started pid $! ; log: $WEBUI_DIR/logs/webui.log"
sleep 3
tail -n 20 "$WEBUI_DIR/logs/webui.log" || true
