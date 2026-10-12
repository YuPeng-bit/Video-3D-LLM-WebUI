#!/bin/bash
# Launch the Video-3D-LLM WebUI.
#
#   sh run_webui.sh              # foreground
#   PORT=8099 sh run_webui.sh    # custom port
#   CUDA_VISIBLE_DEVICES=1 sh run_webui.sh                    # pin a GPU
#
# The model (16 GB, fp16) is loaded lazily on the first inference request.
set -e

WEBUI_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$WEBUI_DIR/.env.local" ]; then set -a; . "$WEBUI_DIR/.env.local"; set +a; fi
PY="${V3D_PYTHON:-python}"
PORT="${PORT:-8099}"
HOST="${HOST:-0.0.0.0}"

export V3D_REPO="${V3D_REPO:-$WEBUI_DIR/../Video-3D-LLM}"
export V3D_WEBUI="$WEBUI_DIR"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export PYTHONPATH="$WEBUI_DIR:$V3D_REPO${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU="$("$PY" "$WEBUI_DIR/tools/pick_gpu.py" 2>/dev/null | tail -n 1)"
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "[run_webui] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (auto-selected freest GPU)"
else
  echo "[run_webui] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (from environment)"
fi

mkdir -p "$WEBUI_DIR/logs"
cd "$WEBUI_DIR"
echo "[run_webui] serving on http://$HOST:$PORT  (repo=$V3D_REPO)"
exec "$PY" -m uvicorn app.server:app --host "$HOST" --port "$PORT" --workers 1
