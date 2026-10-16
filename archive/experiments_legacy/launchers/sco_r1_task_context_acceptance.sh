#!/usr/bin/env bash
# Frozen one-update acceptance only. Never invokes the 250-step entry point.
set -euo pipefail
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
MODE=${1:?renderer or training}
RUN=${2:?new isolated run}
FROZEN=${RUN}/frozen
ASSETS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0/assets/ready
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
WORK=${PWD}
export PATH=${ENV}/bin:${PATH} PYTHONPATH=${WORK}/verl:${WORK}/scripts:${WORK}
export PYTHONDONTWRITEBYTECODE=1 NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export CUDA_HOME=${ENV} TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++
export CPATH=${ENV}/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=${RUN}/torch_extensions
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2
export INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1
renderer_pid=
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == training ]]; then touch "${RUN}/STOP_RENDERER"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
{
  hostname; id; date -u +%FT%TZ; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  sha256sum "${RUN}/package/source.tar.gz" "${FROZEN}/SHA256SUMS"
} > "${OUT}/environment.txt"
(cd "${FROZEN}" && sha256sum -c SHA256SUMS)
"${PY}" -m pip freeze > "${OUT}/pip_freeze.txt"
if [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "${ASSETS}" --host 0.0.0.0 --port 8915 --gpus 0 --max-workers 1 --max-inflight 1 --admit-timeout 900 --conda-env "${ENV}" > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!
  for _ in $(seq 1 180); do
    if "${PY}" -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8915/health",timeout=2)' 2>/dev/null; then break; fi
    kill -0 "${renderer_pid}"; sleep 2
  done
  timeout --kill-after=60s 90m "${PY}" scripts/r1_clean_projective_runtime_preflight.py --frozen "${FROZEN}" --renderer-url http://127.0.0.1:8915/render --gs-root "${ASSETS}" --output "${RUN}/runtime_preflight.json"
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8915/render\n' "${worker_ip}" > "${RUN}/renderer_endpoint.tmp"
  mv "${RUN}/renderer_endpoint.tmp" "${RUN}/renderer_endpoint.txt"
  while [[ ! -e ${RUN}/STOP_RENDERER ]]; do
    kill -0 "${renderer_pid}"; sleep 10
  done
elif [[ ${MODE} == training ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  "${PY}" -c 'import json,sys;r=json.load(open(sys.argv[1]));assert r["status"]=="PASS" and r["completed"]==12' "${RUN}/runtime_preflight.json"
  export R1_RENDER_URL
  R1_RENDER_URL=$(<"${RUN}/renderer_endpoint.txt")
  MODEL=$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["model"]["path"])' "${FROZEN}/data_gate.json")
  (cd "${MODEL}" && sha256sum -c "${MODEL}_SHA256SUMS") > "${OUT}/model_hash_check.txt"
  mkdir -p "${RUN}/smoke"
  timeout --kill-after=120s 3h "${PY}" -m vagen.r1_clean_projective_ppo --config "${FROZEN}/smoke.yaml" --endpoint 1 2>&1 | tee "${RUN}/smoke/train.log"
  "${PY}" scripts/r1_clean_projective_smoke_gate.py --run "${RUN}" --frozen "${FROZEN}"
  "${PY}" scripts/r1_task_context_acceptance.py gate --run "${RUN}"
  # Paired evaluation is unreachable unless ALL acceptance gates passed.
  STEP1=${RUN}/smoke/checkpoints/global_step_1/actor/huggingface
  "${PY}" - "${STEP1}" "${RUN}/step1_SHA256SUMS" <<'PY'
import hashlib, pathlib, sys
root=pathlib.Path(sys.argv[1]);lines=[]
assert (root/'model.safetensors.index.json').is_file()
for path in sorted(p for p in root.iterdir() if p.is_file()):
    with path.open('rb') as f: value=hashlib.file_digest(f,'sha256').hexdigest()
    lines.append(f'{value}  {path.name}\n')
pathlib.Path(sys.argv[2]).write_text(''.join(lines))
PY
  export CUDA_VISIBLE_DEVICES=0
  for model_key in base step1; do
    if [[ ${model_key} == base ]]; then model_path=${MODEL}; hashes=${MODEL}_SHA256SUMS
    else model_path=${STEP1}; hashes=${RUN}/step1_SHA256SUMS; fi
    timeout --kill-after=60s 60m "${PY}" scripts/r1_run_canonical_dev_eval32.py --protocol "${FROZEN}/eval_protocol.json" --model-key "${model_key}" --model-path "${model_path}" --model-sha256s "${hashes}" --renderer-url "${R1_RENDER_URL}" --gs-root "${ASSETS}" --output-dir "${RUN}/paired_eval/${model_key}" --max-infrastructure-attempts 1
  done
else
  exit 2
fi
