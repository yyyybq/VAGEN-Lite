#!/usr/bin/env bash
# Evaluate Qwen v48 and v50 directly on one 8-GPU node without SCO scheduling.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
ENTRY="${ROOT}/examples/train/active_spatial/sco_active_spatial_eval_entry.sh"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
OUT_ROOT="${ROOT}/evaluation/sweeps/active_spatial"
STATE_DIR="${OUT_ROOT}/direct_v48_v50_20260829"

V48_EXP="qwen_v48_clean_d0pass_20260823_full"
V48_STEPS="50,100,700"
V48_SWEEP="qwen_v48_clean_all_ckpts_id_ood_qa_20260828"
V50_EXP="qwen_v50_clean_d0pass_20260824_full"
V50_STEPS="50,100,700"
V50_SWEEP="qwen_v50_clean_all_ckpts_id_ood_qa_20260828"

mkdir -p "${STATE_DIR}"
cd "${ROOT}"

write_marker() {
  local event="$1"
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${event}" \
    >> "${STATE_DIR}/timeline.tsv"
}

run_stage() {
  local label="$1" experiment="$2" steps="$3" sweep="$4" mode="$5" gpus="$6"
  write_marker "${label}_start gpus=${gpus}"
  EVAL_TARGET=qwen \
  EVAL_EXPERIMENT="${experiment}" \
  EVAL_STEPS="${steps}" \
  EVAL_SWEEP_NAME="${sweep}" \
  EVAL_MODE="${mode}" \
  EVAL_PARALLEL_GPUS="${gpus}" \
  EVAL_GPU_MEMORY_UTILIZATION="${DIRECT_EVAL_GPU_MEMORY_UTILIZATION:-0.5}" \
  EVAL_INCLUDE_EASI=1 \
  EVAL_QA_BENCHMARKS=easi_8 \
  bash "${ENTRY}" > "${STATE_DIR}/${label}.log" 2>&1
  write_marker "${label}_complete"
}

write_report() {
  local sweep="$1"
  "${PYTHON}" scripts/active_spatial_eval_report.py \
    --nav-summary "${OUT_ROOT}/${sweep}/summary.csv" \
    --easi-summary "${OUT_ROOT}/${sweep}/easi_results/easi_summary.json" \
    --out "${OUT_ROOT}/${sweep}/analysis_report.md"
}

{
  echo "hostname=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "v48_experiment=${V48_EXP}"
  echo "v48_steps=${V48_STEPS}"
  echo "v50_experiment=${V50_EXP}"
  echo "v50_steps=${V50_STEPS}"
  echo "gpu_memory_utilization=${DIRECT_EVAL_GPU_MEMORY_UTILIZATION:-0.5}"
  echo "git_commit=$(git -c safe.directory="${ROOT}" rev-parse HEAD 2>/dev/null || true)"
} > "${STATE_DIR}/frozen_state.txt"

# Navigation needs all GPUs for the large ID/OOD matrix.
run_stage v48_nav "${V48_EXP}" "${V48_STEPS}" "${V48_SWEEP}" nav 0,1,2,3,4,5,6,7

# v48 has three QA checkpoints. Use the remaining five GPUs for v50 navigation.
run_stage v48_qa "${V48_EXP}" "${V48_STEPS}" "${V48_SWEEP}" qa 0,1,2 &
v48_qa_pid=$!
run_stage v50_nav "${V50_EXP}" "${V50_STEPS}" "${V50_SWEEP}" nav 3,4,5,6,7 &
v50_nav_pid=$!

status=0
wait "${v48_qa_pid}" || status=1
wait "${v50_nav_pid}" || status=1
if [[ "${status}" != 0 ]]; then
  write_marker "parallel_stage_failed"
  exit "${status}"
fi

run_stage v50_qa "${V50_EXP}" "${V50_STEPS}" "${V50_SWEEP}" qa 0,1,2
write_report "${V48_SWEEP}"
write_report "${V50_SWEEP}"
write_marker "all_complete"
