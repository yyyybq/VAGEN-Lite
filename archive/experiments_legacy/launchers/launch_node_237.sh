#!/bin/bash
# =============================================================================
# Node 10.119.21.237 launcher
# GPU status: 3,4,5,6,7 all free (81GB each)
# Experiment: v48_7b_nodelta_w1 (KL=0.30, Window=1, 7B - ablation on window size)
# =============================================================================
set -e

EXPERIMENT="${1:-v48_7b_nodelta_w1.sh}"
VAGEN_ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
SCRIPT="${VAGEN_ROOT}/examples/train/active_spatial/run_experiment.sh"
EXP_CONFIG="${VAGEN_ROOT}/examples/train/active_spatial/experiments/${EXPERIMENT}"
EXP_NAME="${EXPERIMENT%.sh}"
LOG_FILE="${VAGEN_ROOT}/exps/vagen_active_spatial/${EXP_NAME}.log"

echo "=== Node 10.119.21.237 launcher ==="
echo "  Experiment : ${EXPERIMENT}"
echo "  Free GPUs  : 3,4,5,6,7 (81GB each)"
echo "  CUDA_VISIBLE_DEVICES: 3,4,5,6,7"
echo "  Logical mapping: 0→phys3 1→phys4 2→phys5 3→phys6 (render)4→phys7"
echo "  Log: ${LOG_FILE}"

mkdir -p "$(dirname "${LOG_FILE}")"

nohup bash -c "
  export CUDA_VISIBLE_DEVICES=3,4,5,6,7
  cd ${VAGEN_ROOT}
  bash ${SCRIPT} ${EXP_CONFIG}
" > "${LOG_FILE}" 2>&1 &

PID=$!
echo "PID: ${PID}"
echo "Monitor: tail -f ${LOG_FILE}"
echo "Status:  grep -E 'step|reward|collapse' ${LOG_FILE} | tail -20"
