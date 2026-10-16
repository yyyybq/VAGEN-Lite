#!/bin/bash
# =============================================================================
# Node 10.119.21.155 launcher
# GPU status: 0,1,2,3 free (79GB each) + GPU 6 lightly used (rendering)
# Experiment: v46_7b_nodelta_w3 (KL=0.30, Window=3, 7B)
# =============================================================================
set -e

EXPERIMENT="${1:-v46_7b_nodelta_w3.sh}"
VAGEN_ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
SCRIPT="${VAGEN_ROOT}/examples/train/active_spatial/run_experiment.sh"
EXP_CONFIG="${VAGEN_ROOT}/examples/train/active_spatial/experiments/${EXPERIMENT}"
EXP_NAME="${EXPERIMENT%.sh}"
LOG_FILE="${VAGEN_ROOT}/exps/vagen_active_spatial/${EXP_NAME}.log"

echo "=== Node 10.119.21.155 launcher ==="
echo "  Experiment : ${EXPERIMENT}"
echo "  Free GPUs  : 0,1,2,3 (training, ~79GB each) + GPU 6 (rendering)"
echo "  CUDA_VISIBLE_DEVICES: 0,1,2,3,6"
echo "  Logical mapping: 0→phys0 1→phys1 2→phys2 3→phys3 (render)4→phys6"
echo "  Log: ${LOG_FILE}"

mkdir -p "$(dirname "${LOG_FILE}")"

nohup bash -c "
  export CUDA_VISIBLE_DEVICES=0,1,2,3,6
  cd ${VAGEN_ROOT}
  bash ${SCRIPT} ${EXP_CONFIG}
" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Status:  grep -E 'step|reward|collapse' ${LOG_FILE} | tail -20"
