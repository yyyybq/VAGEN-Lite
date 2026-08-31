#!/bin/bash
# =============================================================================
# Launch helper for D0-passed clean Active Spatial anchors.
#
# Usage:
#   bash examples/train/active_spatial/launch_d0pass_clean_anchors.sh dry-run
#   bash examples/train/active_spatial/launch_d0pass_clean_anchors.sh run qwen
#   bash examples/train/active_spatial/launch_d0pass_clean_anchors.sh run cambrian
#
# Run this only on an idle train node. It never writes historical experiment names.
# =============================================================================
set -euo pipefail

ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
RUN="${ROOT}/examples/train/active_spatial/run_experiment.sh"
EXP_DIR="${ROOT}/examples/train/active_spatial/experiments"
LOG_DIR="${ROOT}/exps/vagen_active_spatial"
mkdir -p "${LOG_DIR}"

CMD="${1:-dry-run}"
TARGET="${2:-}"

resolve_exp() {
  case "$1" in
    qwen|qwen7b|v46) echo "qwen_active_spatial_clean_v1.sh" ;;
    cambrian|c8|b5) echo "cambrian_active_spatial_clean_v1.sh" ;;
    *) return 1 ;;
  esac
}

print_cmds() {
  cat <<'CMDS'
# ---- prerequisites ----
# 1) Jumpbox-local or remote HTTP renderer healthy.
# 2) Idle 8-GPU H800 train node.
# 3) From repo root on that node:

cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite

# ---- Qwen clean anchor ----
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false \
  ACTIVE_SPATIAL_CLEAN_RENDER_HOST=10.119.30.223 \
  ACTIVE_SPATIAL_CLEAN_RENDER_PORT=8768 \
  ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL=http \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/qwen_active_spatial_clean_v1.sh \
    > exps/vagen_active_spatial/qwen_v46_clean_d0pass_v1.log 2>&1 &
echo "QWEN_CLEAN_PID=$!"

# ---- Cambrian clean anchor ----
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false \
  ACTIVE_SPATIAL_CLEAN_RENDER_HOST=10.119.30.223 \
  ACTIVE_SPATIAL_CLEAN_RENDER_PORT=8768 \
  ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL=http \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/cambrian_active_spatial_clean_v1.sh \
    > exps/vagen_active_spatial/cambrian_c8_clean_d0pass_v1.log 2>&1 &
echo "CAMBRIAN_CLEAN_PID=$!"
CMDS
}

case "${CMD}" in
  dry-run|cmds)
    echo "Clean anchors:"
    echo "  Qwen:     ${EXP_DIR}/qwen_active_spatial_clean_v1.sh"
    echo "  Cambrian: ${EXP_DIR}/cambrian_active_spatial_clean_v1.sh"
    echo ""
    print_cmds
    ;;
  run)
    [[ -n "${TARGET}" ]] || { echo "need target: qwen|cambrian"; exit 2; }
    EXP_FILE="$(resolve_exp "${TARGET}")" || { echo "unknown target ${TARGET}"; exit 2; }
    case "${EXP_FILE}" in
      qwen_active_spatial_clean_v1.sh) NAME="${QWEN_CLEAN_EXPERIMENT_NAME:-qwen_v46_clean_d0pass_v1}" ;;
      cambrian_active_spatial_clean_v1.sh) NAME="${CAMBRIAN_CLEAN_EXPERIMENT_NAME:-cambrian_c8_clean_d0pass_v1}" ;;
      *) NAME="${EXP_FILE%.sh}" ;;
    esac
    LOG="${LOG_DIR}/${NAME}.log"
    export USE_GPU_HOLDER="${USE_GPU_HOLDER:-false}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export ACTIVE_SPATIAL_CLEAN_RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
    export ACTIVE_SPATIAL_CLEAN_RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
    export ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"
    echo "[run] ${EXP_FILE} -> ${LOG}"
    nohup bash "${RUN}" "${EXP_DIR}/${EXP_FILE}" > "${LOG}" 2>&1 &
    echo "PID=$! LOG=${LOG}"
    ;;
  *)
    echo "usage: $0 {dry-run|cmds|run <qwen|cambrian>}"
    exit 2
    ;;
esac
