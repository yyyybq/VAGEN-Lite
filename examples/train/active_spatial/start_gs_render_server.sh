#!/bin/bash
# =============================================================================
# Compatibility launcher for InteriorGS render service
# =============================================================================
# This script used to launch an external ViewSuite WebSocket service. It now
# starts VAGEN-Lite's built-in HTTP multipart render service with a ViewAgent-
# style worker pool. Use start_gs_render_http_service.sh directly for new runs.
#
# Legacy-compatible example:
#   bash examples/train/active_spatial/start_gs_render_server.sh \
#     --gs-root /path/to/InteriorGS --port 8767 --gpus 0,1 --max-renderers 4
#
# Training side default:
#   RENDER_MODE=remote RENDER_PROTOCOL=http RENDER_HOST=<host> RENDER_PORT=8767
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GS_ROOT="${GS_ROOT:-}"
PORT="${PORT:-8767}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
MAX_WORKERS="${MAX_WORKERS:-${MAX_RENDERERS:-4}}"
MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
CONDA_ENV="${CONDA_ENV:-vagen}"
FORCED_RENDER_SIZE="${FORCED_RENDER_SIZE:-}"
IMAGE_FORMAT="${IMAGE_FORMAT:-PNG}"
IMAGE_QUALITY="${IMAGE_QUALITY:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gs-root) GS_ROOT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --num-shards) shift 2 ;;  # legacy ViewSuite arg, not needed by worker-pool service
        --max-renderers|--max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --max-inflight) MAX_INFLIGHT="$2"; shift 2 ;;
        --forced-render-size) FORCED_RENDER_SIZE="$2"; shift 2 ;;
        --image-format) IMAGE_FORMAT="$2"; shift 2 ;;
        --image-quality) IMAGE_QUALITY="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        --viewsuite) shift 2 ;;  # legacy arg, ignored
        -h|--help) sed -n '2,28p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

CMD=("${SCRIPT_DIR}/start_gs_render_http_service.sh"
    --gs-root "${GS_ROOT}"
    --port "${PORT}"
    --gpus "${GPUS}"
    --max-workers "${MAX_WORKERS}"
    --max-inflight "${MAX_INFLIGHT}"
    --image-format "${IMAGE_FORMAT}"
    --conda-env "${CONDA_ENV}")
if [ -n "${IMAGE_QUALITY}" ]; then
    CMD+=(--image-quality "${IMAGE_QUALITY}")
fi
if [ -n "${FORCED_RENDER_SIZE}" ]; then
    CMD+=(--forced-render-size "${FORCED_RENDER_SIZE}")
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "[start_gs_render_server] Using built-in HTTP renderer. client_url=http://<host>:${PORT}/render"
exec bash "${CMD[@]}"
