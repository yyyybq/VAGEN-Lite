#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
OUT="${ROOT}/exps/vagen_active_spatial/protocol_only_prepost/sco_submission"
AEC2="${U1_PROTOCOL_SCO_AEC2:-h800}"
JOB_NAME="${U1_PROTOCOL_SCO_JOB_NAME:-u1_protocol_prepost_32_20260829}"
IMAGE="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
mkdir -p "${OUT}"
cd "${ROOT}"
export PATH="${HOME}/.sco/bin:${PATH}"

LOG="${OUT}/submit_$(date -u +%Y%m%dT%H%M%SZ).log"
COMMAND="cd ${ROOT} && bash ${ROOT}/examples/train/active_spatial/sco_u1_protocol_only_prepost_entry.sh"
{
  echo "job_name=${JOB_NAME}"
  echo "workspace=aigc"
  echo "aec2=${AEC2}"
  echo "worker_spec=N4lS.Iq.I80.8"
  echo "worker_nodes=1"
  echo "image=${IMAGE}"
  echo "storage_mount=${MOUNT}"
  echo "command=${COMMAND}"
} > "${OUT}/submission_request.txt"

sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name="${AEC2}" \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url="${IMAGE}" \
  --storage-mount="${MOUNT}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.8 \
  --command="${COMMAND}" | tee "${LOG}"

grep -oE 'pt-[a-z0-9]+' "${LOG}" | awk '!seen[$0]++' > "${OUT}/job_id.txt"
test -s "${OUT}/job_id.txt"
echo "SCO_JOB_ID=$(head -1 "${OUT}/job_id.txt")"
