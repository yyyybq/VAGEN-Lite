#!/bin/bash
# =============================================================================
# Launch helper for clean v46 Qwen2.5-VL 3B / 7B baselines
# Usage:
#   bash examples/train/active_spatial/launch_v46_baselines.sh dry-run
#   bash examples/train/active_spatial/launch_v46_baselines.sh cmds
#   # On target node (or via SSH):
#   bash examples/train/active_spatial/launch_v46_baselines.sh run 3b
#   bash examples/train/active_spatial/launch_v46_baselines.sh run 7b
# =============================================================================
set -euo pipefail
ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
RUN="${ROOT}/examples/train/active_spatial/run_experiment.sh"
EXP_DIR="${ROOT}/examples/train/active_spatial/experiments"
LOG_DIR="${ROOT}/exps/vagen_active_spatial"
mkdir -p "${LOG_DIR}"

CMD="${1:-cmds}"
TARGET="${2:-}"

resolve() {
  case "$1" in
    3b|3B) echo "v46_baseline_qwen25vl_3b.sh" ;;
    7b|7B) echo "v46_baseline_qwen25vl_7b.sh" ;;
    *) return 1 ;;
  esac
}

print_cmds() {
  cat <<'CMDS'
# ---- prerequisites ----
# 1) Jump-host render healthy: curl http://10.119.30.223:8767/health
# 2) Idle 8-GPU train node (do NOT co-locate with A1/A2/B5)
# 3) From repo root on that node:

# ---- 3B baseline ----
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v46_baseline_qwen25vl_3b.sh \
    > exps/vagen_active_spatial/v46_baseline_qwen25vl_3b.log 2>&1 &
echo "3B PID=$!"

# ---- 7B baseline ----
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v46_baseline_qwen25vl_7b.sh \
    > exps/vagen_active_spatial/v46_baseline_qwen25vl_7b.log 2>&1 &
echo "7B PID=$!"

# ---- SSH one-liner examples (from login node) ----
# KEY=/mnt/umm/users/yinbaiqiao/id_rsa
# NODE_3B=<idle-8gpu-ip>
# NODE_7B=<idle-8gpu-ip>
# ssh -i $KEY root@$NODE_3B 'bash -lc "
#   cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
#   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 USE_GPU_HOLDER=false
#   export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:\$PATH
#   nohup bash examples/train/active_spatial/run_experiment.sh \
#     examples/train/active_spatial/experiments/v46_baseline_qwen25vl_3b.sh \
#     > exps/vagen_active_spatial/v46_baseline_qwen25vl_3b.log 2>&1 &
#   echo LAUNCH_PID=\$!
# "'
CMDS
}

case "${CMD}" in
  dry-run|cmds)
    echo "Experiments:"
    echo "  3B: ${EXP_DIR}/v46_baseline_qwen25vl_3b.sh"
    echo "  7B: ${EXP_DIR}/v46_baseline_qwen25vl_7b.sh"
    echo "  Env: ${ROOT}/examples/train/active_spatial/env_config_v46_baseline_6types.yaml"
    echo ""
    print_cmds
    ;;
  run)
    [[ -n "${TARGET}" ]] || { echo "need target: 3b|7b"; exit 2; }
    EXP_FILE="$(resolve "${TARGET}")" || { echo "unknown target ${TARGET}"; exit 2; }
    NAME="${EXP_FILE%.sh}"
    LOG="${LOG_DIR}/${NAME}.log"
    echo "[run] ${EXP_FILE} -> ${LOG}"
    export USE_GPU_HOLDER="${USE_GPU_HOLDER:-false}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    nohup bash "${RUN}" "${EXP_DIR}/${EXP_FILE}" > "${LOG}" 2>&1 &
    echo "PID=$! LOG=${LOG}"
    ;;
  *)
    echo "usage: $0 {dry-run|cmds|run <3b|7b>}"
    exit 2
    ;;
esac
