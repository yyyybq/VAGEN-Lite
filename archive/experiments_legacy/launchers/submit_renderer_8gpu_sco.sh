#!/usr/bin/env bash
# Submit one persistent 8-H800 renderer through the established SCO CLI.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
AEC2="${AEC2:-h800}"
DATE_TAG="${DATE_TAG:-20260822}"
SERVICE_TAG="${SCO_RENDER_SERVICE_TAG:-active_spatial_renderer_8h800_${DATE_TAG}_r1}"
JOB_NAME="${SCO_RENDER_JOB_NAME:-${SERVICE_TAG}}"
STATE_DIR="${SCO_RENDER_STATE_DIR:-${ROOT}/exps/vagen_active_spatial/sco_renderer/${SERVICE_TAG}}"
ENDPOINT_FILE="${SCO_RENDER_ENDPOINT_FILE:-${STATE_DIR}/endpoint.txt}"
CONTAINER_IMAGE_URL="registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
STORAGE_MOUNT="019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"

cd "${ROOT}"
export PATH="${HOME}/.sco/bin:${PATH}"
mkdir -p "${STATE_DIR}"
rm -f "${ENDPOINT_FILE}"

case "${AEC2}" in
  h800|zoetrope) WORKER_SPEC="N4lS.Iq.I80.8" ;;
  *) echo "[fatal] unsupported AEC2=${AEC2}" >&2; exit 2 ;;
esac

command="cd ${ROOT} && SCO_RENDER_SERVICE_TAG=${SERVICE_TAG} SCO_RENDER_STATE_DIR=${STATE_DIR} SCO_RENDER_ENDPOINT_FILE=${ENDPOINT_FILE} bash ${ROOT}/examples/train/active_spatial/sco_renderer_8gpu_entry.sh"
SUBMIT_LOG="${STATE_DIR}/submit.log"
sco acp jobs create \
  --workspace-name=aigc \
  --aec2-name="${AEC2}" \
  --job-name="${JOB_NAME}" \
  --priority=HIGHEST \
  --container-image-url="${CONTAINER_IMAGE_URL}" \
  --storage-mount="${STORAGE_MOUNT}" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec="${WORKER_SPEC}" \
  --command="${command}" | tee "${SUBMIT_LOG}"

grep -oE 'pt-[a-z0-9]+' "${SUBMIT_LOG}" | head -n1 > "${STATE_DIR}/job_id.txt"
echo "endpoint_file=${ENDPOINT_FILE}"
