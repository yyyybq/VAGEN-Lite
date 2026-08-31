#!/bin/bash
# =============================================================================
# InteriorGS HTTP render service launcher (ViewAgent-style worker pool)
# =============================================================================
# Example:
#   bash examples/train/active_spatial/start_gs_render_http_service.sh \
#     --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS \
#     --port 8767 --gpus 0,1,2,3 --max-workers 8 --max-inflight 16 \
#     --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
#
# Training side:
#   RENDER_MODE=remote RENDER_PROTOCOL=http RENDER_HOST=<host> RENDER_PORT=8767
# =============================================================================

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8767}"
GS_ROOT="${GS_ROOT:-}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
MAX_WORKERS="${MAX_WORKERS:-4}"
MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
ADMIT_TIMEOUT="${ADMIT_TIMEOUT:-5}"
CONDA_ENV="${CONDA_ENV:-vagen}"
FORCED_RENDER_SIZE="${FORCED_RENDER_SIZE:-}"
IMAGE_FORMAT="${IMAGE_FORMAT:-PNG}"
IMAGE_QUALITY="${IMAGE_QUALITY:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gs-root) GS_ROOT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --max-workers) MAX_WORKERS="$2"; shift 2 ;;
        --max-inflight) MAX_INFLIGHT="$2"; shift 2 ;;
        --admit-timeout) ADMIT_TIMEOUT="$2"; shift 2 ;;
        --forced-render-size) FORCED_RENDER_SIZE="$2"; shift 2 ;;
        --image-format) IMAGE_FORMAT="$2"; shift 2 ;;
        --image-quality) IMAGE_QUALITY="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "${GS_ROOT}" ] || [ ! -d "${GS_ROOT}" ]; then
    echo "ERROR: --gs-root is required and must be a directory" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

if [ -x "${CONDA_ENV}/bin/python" ]; then
    ENV_PREFIX="$(cd "${CONDA_ENV}" && pwd)"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    ENV_PREFIX="${CONDA_PREFIX}"
elif [ -x "${HOME}/.conda/envs/${CONDA_ENV}/bin/python" ]; then
    ENV_PREFIX="${HOME}/.conda/envs/${CONDA_ENV}"
    export PATH="${ENV_PREFIX}/bin:${PATH}"
else
    echo "ERROR: cannot locate conda environment: ${CONDA_ENV}" >&2
    exit 1
fi
PYTHON_BIN="${ENV_PREFIX}/bin/python"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
IFS=',' read -r -a GPU_VALUES <<< "${GPUS//;/,}"
if [ "${#GPU_VALUES[@]}" -eq 0 ]; then
    echo "ERROR: --gpus must contain at least one device" >&2
    exit 1
fi
LOGICAL_GPUS="$(seq -s, 0 "$(( ${#GPU_VALUES[@]} - 1 ))")"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_NUM_THREADS="${PYTORCH_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

CONDA_GCC="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
CONDA_GXX="${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
if [ -x "${CONDA_GCC}" ] && [ -x "${CONDA_GXX}" ]; then
    export CC="${CC:-${CONDA_GCC}}"
    export CXX="${CXX:-${CONDA_GXX}}"
    export CUDAHOSTCXX="${CUDAHOSTCXX:-${CONDA_GXX}}"
fi
CUDA_TARGET_INCLUDE="${ENV_PREFIX}/targets/x86_64-linux/include"
CUDA_TARGET_LIB="${ENV_PREFIX}/targets/x86_64-linux/lib"
if [ -d "${CUDA_TARGET_INCLUDE}" ]; then
    export CPATH="${CUDA_TARGET_INCLUDE}:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_TARGET_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi
if [ -d "${CUDA_TARGET_LIB}" ]; then
    export LIBRARY_PATH="${CUDA_TARGET_LIB}:${LIBRARY_PATH:-}"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/${USER:-vagen}/torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

ARGS=(
  -m vagen.envs.active_spatial.render.service_http.service
  --gs-root "${GS_ROOT}"
  --host "${HOST}"
  --port "${PORT}"
  --gpu-ids "${LOGICAL_GPUS}"
  --max-workers "${MAX_WORKERS}"
  --max-inflight "${MAX_INFLIGHT}"
  --admit-timeout "${ADMIT_TIMEOUT}"
  --image-format "${IMAGE_FORMAT}"
)
if [ -n "${IMAGE_QUALITY}" ]; then
  ARGS+=(--image-quality "${IMAGE_QUALITY}")
fi
if [ -n "${FORCED_RENDER_SIZE}" ]; then
  ARGS+=(--forced-render-size "${FORCED_RENDER_SIZE}")
fi

echo "=============================================="
echo "  InteriorGS HTTP render service"
echo "=============================================="
echo "  GS_ROOT:       ${GS_ROOT}"
echo "  HOST:          ${HOST}"
echo "  PORT:          ${PORT}"
echo "  GPUS:          ${GPUS}"
echo "  LOGICAL_GPUS:  ${LOGICAL_GPUS}"
echo "  MAX_WORKERS:   ${MAX_WORKERS}"
echo "  MAX_INFLIGHT:  ${MAX_INFLIGHT}"
echo "  IMAGE_FORMAT:  ${IMAGE_FORMAT}${IMAGE_QUALITY:+ q=${IMAGE_QUALITY}}"
echo "  TORCH_EXT:     ${TORCH_EXTENSIONS_DIR}"
echo "----------------------------------------------"
echo "  client_url:    http://<host>:${PORT}/render"
echo "=============================================="

exec "${PYTHON_BIN}" "${ARGS[@]}"
