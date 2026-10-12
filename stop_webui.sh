#!/bin/bash
set -e
WEBUI_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$WEBUI_DIR/.env.local" ]; then set -a; . "$WEBUI_DIR/.env.local"; set +a; fi
"${V3D_PYTHON:-python}" "$WEBUI_DIR/tools/stop_service.py" --port "${PORT:-8099}"
