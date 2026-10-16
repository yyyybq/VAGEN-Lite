#!/bin/bash
# =============================================================================
# Launch helper for A1 / A2 / B5 (does not auto-SSH; prints or runs locally)
# Usage:
#   bash examples/train/active_spatial/launch_a1_a2_b5.sh prep
#   bash examples/train/active_spatial/launch_a1_a2_b5.sh dry-run
#   bash examples/train/active_spatial/launch_a1_a2_b5.sh run a1_w005
#   bash examples/train/active_spatial/launch_a1_a2_b5.sh run a2
#   bash examples/train/active_spatial/launch_a1_a2_b5.sh run b5
# =============================================================================
set -euo pipefail
ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
PY="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
RUN="${ROOT}/examples/train/active_spatial/run_experiment.sh"
EXP_DIR="${ROOT}/examples/train/active_spatial/experiments"
LOG_DIR="${ROOT}/exps/vagen_active_spatial"
mkdir -p "${LOG_DIR}"

CMD="${1:-dry-run}"
TARGET="${2:-}"

prep_aux() {
  echo "[prep] building SITE MCQ aux jsonl..."
  "${PY}" "${ROOT}/scripts/prep_spatial_aux_mcq.py" --max-n 2000
}

resolve_exp() {
  case "$1" in
    a1_w005|a1_005|a1-0.05) echo "a1_v46_spatial_aux_w005.sh" ;;
    a1_w010|a1_010|a1-0.10) echo "a1_v46_spatial_aux_w010.sh" ;;
    a2|a2_stop|arrival_stop) echo "a2_v46_arrival_stop.sh" ;;
    b5|b5_c8|wrapper) echo "b5_c8_wrapper_img25_actionvalid.sh" ;;
    *) echo ""; return 1 ;;
  esac
}

case "${CMD}" in
  prep)
    prep_aux
    ;;
  dry-run)
    echo "Planned experiments:"
    echo "  A1 w005: ${EXP_DIR}/a1_v46_spatial_aux_w005.sh"
    echo "  A1 w010: ${EXP_DIR}/a1_v46_spatial_aux_w010.sh"
    echo "  A2:      ${EXP_DIR}/a2_v46_arrival_stop.sh"
    echo "  B5:      ${EXP_DIR}/b5_c8_wrapper_img25_actionvalid.sh"
    echo ""
    echo "Prep aux data first: $0 prep"
    echo "Local launch example:"
    echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false \\"
    echo "    nohup bash ${RUN} ${EXP_DIR}/a1_v46_spatial_aux_w005.sh \\"
    echo "    > ${LOG_DIR}/a1_v46_spatial_aux_w005.log 2>&1 &"
    echo ""
    echo "B5 eval overrides:"
    echo "  export VAGEN_CAMBRIAN_LIMIT_IMAGES=25 VAGEN_CAMBRIAN_MAX_MODEL_LEN=16384"
    ;;
  run)
    [[ -n "${TARGET}" ]] || { echo "need target: a1_w005|a1_w010|a2|b5"; exit 2; }
    EXP_FILE="$(resolve_exp "${TARGET}")" || { echo "unknown target ${TARGET}"; exit 2; }
    if [[ "${EXP_FILE}" == a1_* ]]; then
      prep_aux
    fi
    NAME="${EXP_FILE%.sh}"
    LOG="${LOG_DIR}/${NAME}.log"
    echo "[run] ${EXP_FILE} -> ${LOG}"
    export USE_GPU_HOLDER="${USE_GPU_HOLDER:-false}"
    nohup bash "${RUN}" "${EXP_DIR}/${EXP_FILE}" > "${LOG}" 2>&1 &
    echo "PID=$! LOG=${LOG}"
    ;;
  *)
    echo "usage: $0 {prep|dry-run|run <a1_w005|a1_w010|a2|b5>}"
    exit 2
    ;;
esac
