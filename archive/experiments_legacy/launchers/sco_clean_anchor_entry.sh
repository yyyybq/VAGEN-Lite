#!/usr/bin/env bash
# Shared SCO entry: preflight the remote renderer, run one 8-GPU acceptance
# step, then launch the frozen canonical clean recipe.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
ANCHOR="${1:-${ANCHOR:-}}"
[[ -n "${ANCHOR}" ]] || { echo "usage: ANCHOR=<qwen|cambrian> $0" >&2; exit 2; }
cd "${ROOT}"

export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}"
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
FULL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
export USE_GPU_HOLDER="${USE_GPU_HOLDER:-false}"
export ACTIVE_SPATIAL_CLEAN_RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
export ACTIVE_SPATIAL_CLEAN_RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
export ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"
export ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE="${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE:-}"
export ACTIVE_SPATIAL_RENDER_ENDPOINT_WAIT_SECONDS="${ACTIVE_SPATIAL_RENDER_ENDPOINT_WAIT_SECONDS:-1800}"
# The first real scene render takes longer than the service's five-second
# admission window when two eight-worker jobs briefly contend for it.
export INTERIORGS_HTTP_RETRIES="${INTERIORGS_HTTP_RETRIES:-6}"
export INTERIORGS_HTTP_BACKOFF="${INTERIORGS_HTTP_BACKOFF:-2}"
export VAGEN_SKIP_RENDERER_PREFLIGHT="${VAGEN_SKIP_RENDERER_PREFLIGHT:-0}"
export VAGEN_SKIP_ACCEPTANCE_SMOKE="${VAGEN_SKIP_ACCEPTANCE_SMOKE:-0}"
export VAGEN_SKIP_REASON="${VAGEN_SKIP_REASON:-none}"

if [[ -n "${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE}" ]]; then
  deadline=$((SECONDS + ACTIVE_SPATIAL_RENDER_ENDPOINT_WAIT_SECONDS))
  while [[ ! -s "${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE}" ]]; do
    if (( SECONDS >= deadline )); then
      echo "[fatal] renderer endpoint file not ready: ${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE}" >&2
      exit 4
    fi
    echo "[renderer] waiting for endpoint file: ${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE}"
    sleep 10
  done
  RENDER_ENDPOINT="$(tr -d '[:space:]' < "${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE}")"
  [[ "${RENDER_ENDPOINT}" =~ ^https?://[^:/]+:[0-9]+$ ]] || {
    echo "[fatal] invalid renderer endpoint: ${RENDER_ENDPOINT}" >&2
    exit 4
  }
  ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL="${RENDER_ENDPOINT%%://*}"
  render_authority="${RENDER_ENDPOINT#*://}"
  ACTIVE_SPATIAL_CLEAN_RENDER_HOST="${render_authority%:*}"
  ACTIVE_SPATIAL_CLEAN_RENDER_PORT="${render_authority##*:}"
  export ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL
  export ACTIVE_SPATIAL_CLEAN_RENDER_HOST
  export ACTIVE_SPATIAL_CLEAN_RENDER_PORT
  echo "[renderer] discovered ${RENDER_ENDPOINT} from ${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE}"
fi

case "${ANCHOR}" in
  qwen)
    RUN_NAME="${QWEN_CLEAN_EXPERIMENT_NAME:-qwen_v46_clean_d0pass_20260821_full_r6}"
    SMOKE_NAME="${D0_17_QWEN_SMOKE_NAME:-${RUN_NAME}_acceptance_smoke}"
    SMOKE_CONFIG="examples/train/active_spatial/experiments/d0_17_qwen_clean_acceptance_smoke.sh"
    FULL_CONFIG="${QWEN_CLEAN_FULL_CONFIG:-examples/train/active_spatial/experiments/qwen_active_spatial_clean_v1.sh}"
    export QWEN_CLEAN_EXPERIMENT_NAME="${RUN_NAME}"
    export D0_17_QWEN_SMOKE_NAME="${SMOKE_NAME}"
    export D0_17_NUM_TRAIN_GPUS="${D0_17_NUM_TRAIN_GPUS:-8}"
    export D0_17_TP_SIZE="${D0_17_TP_SIZE:-4}"
    ;;
  cambrian)
    RUN_NAME="${CAMBRIAN_CLEAN_EXPERIMENT_NAME:-cambrian_c8_clean_d0pass_20260821_full_r4}"
    SMOKE_NAME="${D0_17_CAMBRIAN_SMOKE_NAME:-${RUN_NAME}_acceptance_smoke}"
    SMOKE_CONFIG="examples/train/active_spatial/experiments/d0_17_cambrian_clean_acceptance_smoke.sh"
    FULL_CONFIG="${CAMBRIAN_CLEAN_FULL_CONFIG:-examples/train/active_spatial/experiments/cambrian_active_spatial_clean_v1.sh}"
    export CAMBRIAN_CLEAN_EXPERIMENT_NAME="${RUN_NAME}"
    export D0_17_CAMBRIAN_SMOKE_NAME="${SMOKE_NAME}"
    export D0_17_NUM_TRAIN_GPUS="${D0_17_NUM_TRAIN_GPUS:-8}"
    export D0_17_TP_SIZE="${D0_17_TP_SIZE:-2}"
    ;;
  *) echo "[fatal] unknown anchor: ${ANCHOR}" >&2; exit 2 ;;
esac

RUN_DIR="${ROOT}/exps/vagen_active_spatial/${RUN_NAME}"
SCO_DIR="${RUN_DIR}/sco"
mkdir -p "${SCO_DIR}"
ENTRY_LOG="${SCO_DIR}/entry.log"
exec > >(tee -a "${ENTRY_LOG}") 2>&1
on_exit() {
  status=$?
  printf 'status=%s\ndate_utc=%s\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${SCO_DIR}/exit_status.txt"
  exit "${status}"
}
trap on_exit EXIT
{
  echo "anchor=${ANCHOR}"
  echo "run_name=${RUN_NAME}"
  echo "smoke_name=${SMOKE_NAME}"
  echo "renderer=${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL}://${ACTIVE_SPATIAL_CLEAN_RENDER_HOST}:${ACTIVE_SPATIAL_CLEAN_RENDER_PORT}"
  echo "renderer_endpoint_file=${ACTIVE_SPATIAL_RENDER_ENDPOINT_FILE:-none}"
  echo "renderer_retries=${INTERIORGS_HTTP_RETRIES}"
  echo "renderer_backoff=${INTERIORGS_HTTP_BACKOFF}"
  echo "renderer_max_backoff=${INTERIORGS_HTTP_MAX_BACKOFF:-60.0}"
  echo "skip_renderer_preflight=${VAGEN_SKIP_RENDERER_PREFLIGHT}"
  echo "skip_acceptance_smoke=${VAGEN_SKIP_ACCEPTANCE_SMOKE}"
  echo "skip_reason=${VAGEN_SKIP_REASON}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "hostname=$(hostname)"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -c safe.directory="${ROOT}" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE_ON_WORKER)"
} > "${SCO_DIR}/frozen_state.txt"

RENDER_URL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL}://${ACTIVE_SPATIAL_CLEAN_RENDER_HOST}:${ACTIVE_SPATIAL_CLEAN_RENDER_PORT}"
curl --noproxy "*" -fsS "${RENDER_URL}/health" | tee "${SCO_DIR}/renderer_health.json"
if [[ "${VAGEN_SKIP_RENDERER_PREFLIGHT}" == 1 ]]; then
  printf 'SKIPPED: %s\n' "${VAGEN_SKIP_REASON}" | tee "${SCO_DIR}/renderer_preflight_skipped.txt"
else
  "${PYTHON}" tools/u1_phase3_data_gate_http.py \
    --renderer-url "${RENDER_URL}/render" \
    --env-yaml "${ROOT}/examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml" \
    --out "${SCO_DIR}/renderer_preflight" --timeout 60 --retries "${INTERIORGS_HTTP_RETRIES}"
fi

if [[ "${VAGEN_SKIP_ACCEPTANCE_SMOKE}" == 1 ]]; then
  printf 'SKIPPED: %s\n' "${VAGEN_SKIP_REASON}" | tee "${SCO_DIR}/acceptance_smoke_skipped.txt"
else
  export D0_17_RENDER_MODE=remote
  export D0_17_RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST}"
  export D0_17_RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT}"
  export D0_17_RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL}"
  export D0_17_TOTAL_STEPS="${D0_17_TOTAL_STEPS:-1}"
  export D0_17_CRITIC_WARMUP="${D0_17_CRITIC_WARMUP:-0}"
  bash examples/train/active_spatial/run_experiment.sh "${SMOKE_CONFIG}" 2>&1 | tee "${SCO_DIR}/acceptance_smoke.log"

  AUDIT_JSON="${ROOT}/exps/vagen_active_spatial/${SMOKE_NAME}/d0_17_${ANCHOR}_smoke/d0_16_decoupled_correction_audit_step1.json"
  "${PYTHON}" -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text())
bad = []
for split, stats in d["B_ppo_proximal_current_HF_vs_old_HF"].items():
    if stats["frac_abs_ratio_minus_1_gt_0p05"] > 1e-8: bad.append(split + ": proximal ratio drift")
for split, stats in d["rollout_is_weight_tensor_stats"].items():
    if stats.get("ess_over_n") is not None and stats["ess_over_n"] < .80: bad.append(split + ": low ESS/N")
for split, stats in d["identity_A_equals_B_times_C"].items():
    if stats.get("max_abs_log_error", 0.) > 1e-6: bad.append(split + ": A!=B*C")
if bad: raise SystemExit("acceptance gate failed: " + "; ".join(bad))
print("ACCEPTANCE_JSON_OK", p)
' "${AUDIT_JSON}" | tee "${SCO_DIR}/acceptance_gate_check.txt"
  grep -q "trainer/actor_update_performed:1.0" "${SCO_DIR}/acceptance_smoke.log" || { echo "[fatal] acceptance smoke did not update actor" >&2; exit 3; }
fi

if [[ "${D0_17_SMOKE_ONLY:-0}" == 1 ]]; then exit 0; fi
ray stop --force || true
sleep 5
bash examples/train/active_spatial/run_experiment.sh "${FULL_CONFIG}" 2>&1 | tee "${SCO_DIR}/train_full.log"
