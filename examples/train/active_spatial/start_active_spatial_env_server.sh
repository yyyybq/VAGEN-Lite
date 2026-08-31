#!/bin/bash
# =============================================================================
# Active Spatial RemoteEnv service launcher
# =============================================================================
# Runs the full ActiveSpatial environment on this machine behind VAGEN's
# HTTP RemoteEnv protocol. Use this when the service host has local access to
# InteriorGS / gsplat and the training host should talk to it over HTTP.
#
# Example:
#   bash examples/train/active_spatial/start_active_spatial_env_server.sh \
#       --port 8000 --gpus 0 --max-inflight 16
#
# Training side:
#   export REMOTE_ENV_URLS=http://<service-host>:8000
#   bash examples/train/active_spatial/experiments/your_exp.sh
# =============================================================================

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
MAX_SESSIONS="${MAX_SESSIONS:-0}"
SESSION_TIMEOUT="${SESSION_TIMEOUT:-3600}"
ADMIT_TIMEOUT="${ADMIT_TIMEOUT:-5}"
WORKERS="${UVICORN_WORKERS:-1}"
CONDA_ENV="${CONDA_ENV:-vagen}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --max-inflight) MAX_INFLIGHT="$2"; shift 2 ;;
        --max-sessions) MAX_SESSIONS="$2"; shift 2 ;;
        --session-timeout) SESSION_TIMEOUT="$2"; shift 2 ;;
        --admit-timeout) ADMIT_TIMEOUT="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        -h|--help) sed -n '2,35p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# CONDA_ENV may be either an environment name or an absolute environment path.
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
    echo "Pass --conda-env /absolute/path/to/env on non-interactive nodes." >&2
    exit 1
fi
PYTHON_BIN="${ENV_PREFIX}/bin/python"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

# gsplat may JIT-compile its CUDA extension on first render. Conda CUDA
# toolchains often provide prefixed compilers and headers under targets/.
if [ -n "${ENV_PREFIX}" ]; then
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
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/${USER:-vagen}/torch_extensions}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

echo "=============================================="
echo "  Active Spatial RemoteEnv service"
echo "=============================================="
echo "  HOST:            ${HOST}"
echo "  PORT:            ${PORT}"
echo "  CUDA devices:    ${CUDA_VISIBLE_DEVICES}"
echo "  MAX_INFLIGHT:    ${MAX_INFLIGHT}"
echo "  MAX_SESSIONS:    ${MAX_SESSIONS}"
echo "  WORKERS:         ${WORKERS}"
echo "----------------------------------------------"
echo "  Training REMOTE_ENV_URLS=http://<this-host>:${PORT}"
echo "=============================================="

"${PYTHON_BIN}" -m vagen.envs.active_spatial.service \
    --host "${HOST}" \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --max-sessions "${MAX_SESSIONS}" \
    --session-timeout "${SESSION_TIMEOUT}" \
    --admit-timeout "${ADMIT_TIMEOUT}" \
    --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}"
