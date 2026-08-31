#!/usr/bin/env bash
# Submit D0-correct canonical Active Spatial anchors through the existing
# /mnt/umm/users/yinbaiqiao/submit.sh SCO wrapper.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
AEC2="${AEC2:-h800}"
WORKER_NODES="${WORKER_NODES:-1}"
NPROC="${NPROC:-8}"
DATE_TAG="${DATE_TAG:-20260820}"
RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"
CLEAN_ENTRY_EXTRA_ENV="${CLEAN_ENTRY_EXTRA_ENV:-}"
TARGET="${1:-both}"

cd "${ROOT}"
export PATH="${HOME}/.sco/bin:${PATH}"

CONTAINER_IMAGE_URL="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
STORAGE_MOUNT="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
case "${AEC2}" in
  h800|zoetrope) WORKER_SPEC_PREFIX="N4lS.Iq.I80." ;;
  *) echo "[fatal] unsupported AEC2=${AEC2}; expected h800 or zoetrope" >&2; exit 2 ;;
esac
SUBMIT_DIR="${ROOT}/exps/vagen_active_spatial/sco_submissions"
mkdir -p "${SUBMIT_DIR}"
SUBMIT_LOG="${SUBMIT_DIR}/clean_anchors_${DATE_TAG}_$(date -u +%H%M%S).log"

submit_one() {
  local anchor="$1"
  local job_name run_name entry_script run_env
  case "${anchor}" in
    qwen)
      job_name="${QWEN_SCO_JOB_NAME:-qwen_v46_clean_d0pass_${DATE_TAG}_full_r6}"
      run_name="${QWEN_CLEAN_EXPERIMENT_NAME:-qwen_v46_clean_d0pass_${DATE_TAG}_full_r6}"
      entry_script="${ROOT}/examples/train/active_spatial/sco_qwen_clean_anchor_entry.sh"
      run_env="QWEN_CLEAN_EXPERIMENT_NAME"
      ;;
    cambrian)
      job_name="${CAMBRIAN_SCO_JOB_NAME:-cambrian_c8_clean_d0pass_${DATE_TAG}_full_r4}"
      run_name="${CAMBRIAN_CLEAN_EXPERIMENT_NAME:-cambrian_c8_clean_d0pass_${DATE_TAG}_full_r4}"
      entry_script="${ROOT}/examples/train/active_spatial/sco_cambrian_clean_anchor_entry.sh"
      run_env="CAMBRIAN_CLEAN_EXPERIMENT_NAME"
      ;;
    qwen_v47)
      job_name="qwen_v47_clean_d0pass_${DATE_TAG}_full"
      run_name="qwen_v47_clean_d0pass_${DATE_TAG}_full"
      entry_script="${ROOT}/examples/train/active_spatial/sco_qwen_v47_clean_entry.sh"
      run_env="QWEN_CLEAN_EXPERIMENT_NAME"
      ;;
    qwen_v48)
      job_name="qwen_v48_clean_d0pass_${DATE_TAG}_full"
      run_name="qwen_v48_clean_d0pass_${DATE_TAG}_full"
      entry_script="${ROOT}/examples/train/active_spatial/sco_qwen_v48_clean_entry.sh"
      run_env="QWEN_CLEAN_EXPERIMENT_NAME"
      ;;
    qwen_v50)
      job_name="qwen_v50_clean_d0pass_${DATE_TAG}_full"
      run_name="qwen_v50_clean_d0pass_${DATE_TAG}_full"
      entry_script="${ROOT}/examples/train/active_spatial/sco_qwen_v50_clean_entry.sh"
      run_env="QWEN_CLEAN_EXPERIMENT_NAME"
      ;;
    cambrian_c4)
      job_name="cambrian_c4_clean_d0pass_${DATE_TAG}_full"
      run_name="cambrian_c4_clean_d0pass_${DATE_TAG}_full"
      entry_script="${ROOT}/examples/train/active_spatial/sco_cambrian_c4_clean_entry.sh"
      run_env="CAMBRIAN_CLEAN_EXPERIMENT_NAME"
      ;;
    *)
      echo "[fatal] unknown anchor ${anchor}" >&2
      exit 2
      ;;
  esac
  command="cd ${ROOT} && ${CLEAN_ENTRY_EXTRA_ENV} ${run_env}=${run_name} bash ${entry_script}"
  echo "[submit] anchor=${anchor} job=${job_name} run=${run_name}" | tee -a "${SUBMIT_LOG}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="${CONTAINER_IMAGE_URL}" \
    --storage-mount="${STORAGE_MOUNT}" \
    --training-framework=pytorch \
    --worker-nodes="${WORKER_NODES}" \
    --worker-spec="${WORKER_SPEC_PREFIX}${NPROC}" \
    --command="${command}" | tee -a "${SUBMIT_LOG}"
}

case "${TARGET}" in
  qwen) submit_one qwen ;;
  cambrian) submit_one cambrian ;;
  qwen_v47|qwen_v48|qwen_v50|cambrian_c4) submit_one "${TARGET}" ;;
  variants)
    submit_one qwen_v47
    submit_one qwen_v48
    submit_one qwen_v50
    submit_one cambrian_c4
    ;;
  both)
    submit_one qwen
    submit_one cambrian
    ;;
  *)
    echo "usage: $0 {qwen|cambrian|both|qwen_v47|qwen_v48|qwen_v50|cambrian_c4|variants}" >&2
    exit 2
    ;;
esac

grep -oE 'pt-[a-z0-9]+' "${SUBMIT_LOG}" | awk '!seen[$0]++' > "${SUBMIT_LOG%.log}_job_ids.txt" || true
echo "SCO job ids: $(tr '\n' ' ' < "${SUBMIT_LOG%.log}_job_ids.txt")"
