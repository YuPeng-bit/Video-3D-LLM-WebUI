#!/bin/bash
set -e
WEBUI_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8099}" sh "$WEBUI_DIR/stop_webui.sh"
PORT="${PORT:-8099}" sh "$WEBUI_DIR/start_webui.sh"
