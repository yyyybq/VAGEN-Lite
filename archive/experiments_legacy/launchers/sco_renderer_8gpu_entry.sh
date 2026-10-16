#!/usr/bin/env bash
# Run a persistent InteriorGS HTTP renderer on one SCO 8-H800 worker.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
GS_ROOT="${GS_ROOT:-/mnt/umm/users/yinbaiqiao/InteriorGS}"
CONDA_ENV="${CONDA_ENV:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite}"
RENDER_PORT="${SCO_RENDER_PORT:-8768}"
RENDER_GPUS="${SCO_RENDER_GPUS:-0,1,2,3,4,5,6,7}"
MAX_WORKERS="${SCO_RENDER_MAX_WORKERS:-32}"
MAX_INFLIGHT="${SCO_RENDER_MAX_INFLIGHT:-64}"
ADMIT_TIMEOUT="${SCO_RENDER_ADMIT_TIMEOUT:-300}"
SERVICE_TAG="${SCO_RENDER_SERVICE_TAG:-active_spatial_renderer_8h800_20260822_r1}"
STATE_DIR="${SCO_RENDER_STATE_DIR:-${ROOT}/exps/vagen_active_spatial/sco_renderer/${SERVICE_TAG}}"
ENDPOINT_FILE="${SCO_RENDER_ENDPOINT_FILE:-${STATE_DIR}/endpoint.txt}"
PYTHON="${CONDA_ENV}/bin/python"

cd "${ROOT}"
mkdir -p "${STATE_DIR}" "$(dirname "${ENDPOINT_FILE}")"
exec > >(tee -a "${STATE_DIR}/service.log") 2>&1

export PATH="${CONDA_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${RENDER_GPUS}"
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
# Reuse the already-built H800-compatible gsplat extension. This avoids a
# concurrent JIT build by all renderer workers on service startup.
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-32}"

advertise_host="${SCO_RENDER_ADVERTISE_HOST:-}"
if [[ -z "${advertise_host}" ]]; then
  advertise_host="$(hostname -I | tr ' ' '\n' | awk '/^10\.119\./ {print; exit}')"
fi
if [[ -z "${advertise_host}" ]]; then
  advertise_host="$(hostname -I | awk '{print $1}')"
fi
[[ -n "${advertise_host}" ]] || { echo "[fatal] could not determine renderer pod IP" >&2; exit 2; }

endpoint="http://${advertise_host}:${RENDER_PORT}"
tmp_endpoint="${ENDPOINT_FILE}.tmp.$$"
rm -f "${ENDPOINT_FILE}"

cleanup() {
  status=$?
  rm -f "${ENDPOINT_FILE}" "${tmp_endpoint}"
  if [[ -n "${service_pid:-}" ]]; then
    kill "${service_pid}" 2>/dev/null || true
    wait "${service_pid}" 2>/dev/null || true
  fi
  printf '{"status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATE_DIR}/exit.json"
  exit "${status}"
}
trap cleanup EXIT INT TERM

{
  echo "service_tag=${SERVICE_TAG}"
  echo "hostname=$(hostname)"
  echo "pod_ip=${advertise_host}"
  echo "endpoint=${endpoint}"
  echo "gpus=${RENDER_GPUS}"
  echo "max_workers=${MAX_WORKERS}"
  echo "max_inflight=${MAX_INFLIGHT}"
  echo "admit_timeout=${ADMIT_TIMEOUT}"
  echo "torch_extensions_dir=${TORCH_EXTENSIONS_DIR}"
  echo "git_commit=$(git -c safe.directory="${ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STATE_DIR}/frozen_state.txt"
nvidia-smi > "${STATE_DIR}/nvidia_smi.txt"

bash examples/train/active_spatial/start_gs_render_http_service.sh \
  --gs-root "${GS_ROOT}" \
  --port "${RENDER_PORT}" \
  --gpus "${RENDER_GPUS}" \
  --max-workers "${MAX_WORKERS}" \
  --max-inflight "${MAX_INFLIGHT}" \
  --admit-timeout "${ADMIT_TIMEOUT}" \
  --conda-env "${CONDA_ENV}" &
service_pid=$!

for _ in $(seq 1 180); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${RENDER_PORT}/health" > "${STATE_DIR}/health.json"; then
    if [[ "${SCO_RENDER_SKIP_WARMUP:-0}" != 1 ]]; then
      echo "[warmup] running a real local reset/step preflight before publishing endpoint"
      NO_PROXY='*' no_proxy='*' "${PYTHON}" tools/u1_phase3_data_gate_http.py \
        --renderer-url "http://127.0.0.1:${RENDER_PORT}/render" \
        --env-yaml examples/train/active_spatial/env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml \
        --out "${STATE_DIR}/local_real_warmup" \
        --final-delivery "${STATE_DIR}/unused_final_delivery.json" \
        --timeout 300 --retries 3
    fi
    printf '%s\n' "${endpoint}" > "${tmp_endpoint}"
    mv -f "${tmp_endpoint}" "${ENDPOINT_FILE}"
    echo "[ready] renderer=${endpoint} endpoint_file=${ENDPOINT_FILE} pid=${service_pid}"
    wait "${service_pid}"
    exit $?
  fi
  if ! kill -0 "${service_pid}" 2>/dev/null; then
    echo "[fatal] renderer exited during startup" >&2
    wait "${service_pid}"
    exit $?
  fi
  sleep 2
done

echo "[fatal] renderer did not become healthy within 360 seconds" >&2
exit 3
