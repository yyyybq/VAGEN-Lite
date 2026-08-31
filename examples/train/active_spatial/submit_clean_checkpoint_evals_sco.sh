#!/usr/bin/env bash
# Submit canonical ID/OOD navigation plus EASI-8 QA for clean checkpoints.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
AEC2="${AEC2:-h800}"
DATE_TAG="${DATE_TAG:-20260828}"
TARGET="${1:-all}"
CONTAINER_IMAGE_URL="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
STORAGE_MOUNT="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
WORKER_SPEC="N4lS.Iq.I80.8"
ENTRY="${ROOT}/examples/train/active_spatial/sco_active_spatial_eval_entry.sh"
SUBMIT_DIR="${ROOT}/evaluation/sweeps/active_spatial/sco_submissions"

cd "${ROOT}"
export PATH="${HOME}/.sco/bin:${PATH}"
mkdir -p "${SUBMIT_DIR}"
SUBMIT_LOG="${SUBMIT_DIR}/clean_checkpoint_evals_${DATE_TAG}_$(date -u +%H%M%S).log"

submit_one() {
  local key="$1"
  local eval_target experiment steps sweep_name resume_from job_name
  resume_from=""
  case "${key}" in
    v46)
      eval_target=qwen
      experiment=qwen_v46_clean_d0pass_20260822_sco_renderer_r1_full
      steps=50,100,150,200,250,300,350,400,450,500,550,600,650,700
      sweep_name=qwen_v46_clean_all_ckpts_id_ood_qa_${DATE_TAG}
      resume_from="${ROOT}/evaluation/sweeps/active_spatial/qwen_v46_clean_all_ckpts_nav_20260825_parallel8_r5"
      ;;
    v47)
      eval_target=qwen
      experiment=qwen_v47_clean_d0pass_20260826_recovery_r1_full
      steps=50,100,700
      sweep_name=qwen_v47_clean_all_ckpts_id_ood_qa_${DATE_TAG}
      ;;
    v48)
      eval_target=qwen
      experiment=qwen_v48_clean_d0pass_20260823_full
      steps=50,100,700
      sweep_name=qwen_v48_clean_all_ckpts_id_ood_qa_${DATE_TAG}
      ;;
    v50)
      eval_target=qwen
      experiment=qwen_v50_clean_d0pass_20260824_full
      steps=50,100,700
      sweep_name=qwen_v50_clean_all_ckpts_id_ood_qa_${DATE_TAG}
      ;;
    c8)
      eval_target=cambrian
      experiment=cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full
      steps=50,100,1000
      sweep_name=cambrian_c8_clean_all_ckpts_id_ood_qa_${DATE_TAG}
      ;;
    c4)
      eval_target=cambrian
      experiment=cambrian_c4_clean_d0pass_20260822_h800_r5_full
      steps=960
      sweep_name=cambrian_c4_clean_all_ckpts_id_ood_qa_${DATE_TAG}
      ;;
    *) echo "[fatal] unknown evaluation target: ${key}" >&2; exit 2 ;;
  esac
  job_name="${sweep_name}"
  local command
  command="cd ${ROOT} && EVAL_TARGET=${eval_target} EVAL_EXPERIMENT=${experiment} EVAL_STEPS=${steps} EVAL_SWEEP_NAME=${sweep_name} EVAL_MODE=full EVAL_PARALLEL_GPUS=0,1,2,3,4,5,6,7 EVAL_INCLUDE_EASI=1 EVAL_QA_BENCHMARKS=easi_8"
  if [[ -n "${resume_from}" ]]; then
    command+=" EVAL_RESUME_FROM_SWEEP=${resume_from}"
  fi
  command+=" bash ${ENTRY}"

  echo "[submit] target=${key} job=${job_name} experiment=${experiment} steps=${steps}" | tee -a "${SUBMIT_LOG}"
  sco acp jobs create \
    --workspace-name=aigc \
    --aec2-name="${AEC2}" \
    --job-name="${job_name}" \
    --priority=HIGHEST \
    --container-image-url="${CONTAINER_IMAGE_URL}" \
    --storage-mount="${STORAGE_MOUNT}" \
    --training-framework=pytorch \
    --worker-nodes=1 \
    --worker-spec="${WORKER_SPEC}" \
    --command="${command}" | tee -a "${SUBMIT_LOG}"
}

case "${TARGET}" in
  all)
    for key in v46 v47 v48 v50 c8 c4; do submit_one "${key}"; done
    ;;
  v46|v47|v48|v50|c8|c4) submit_one "${TARGET}" ;;
  *) echo "usage: $0 {all|v46|v47|v48|v50|c8|c4}" >&2; exit 2 ;;
esac

grep -oE 'pt-[a-z0-9]+' "${SUBMIT_LOG}" | awk '!seen[$0]++' > "${SUBMIT_LOG%.log}_job_ids.txt" || true
echo "submit_log=${SUBMIT_LOG}"
echo "job_ids=$(tr '\n' ' ' < "${SUBMIT_LOG%.log}_job_ids.txt")"
