#!/usr/bin/env bash
# Isolated eight-update gate-aligned reward pilot. Evaluation runs separately.
set -euo pipefail
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
MODE=${1:?renderer or training}
RUN=${2:?isolated pilot run}
FROZEN=${RUN}/frozen
ASSETS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0/assets/ready
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
WORK=${PWD}
export PYTHONDONTWRITEBYTECODE=1 NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export PYTHONPATH=${WORK}/verl:${WORK}/scripts:${WORK}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=${RUN}/torch_extensions
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2
export INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10

configure_cuda_devel() {
  local root header library
  local wheel_root=${ENV}/lib/python3.12/site-packages/nvidia/curand
  if [[ -x ${ENV}/bin/nvcc && -f ${wheel_root}/include/curand.h ]] && compgen -G "${wheel_root}/lib/libcurand.so*" >/dev/null; then
    export CUDA_HOME=${ENV} CURAND_INCLUDE_DIR=${wheel_root}/include CURAND_LIBRARY_DIR=${wheel_root}/lib
  else
    for root in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.1; do
      [[ -x ${root}/bin/nvcc ]] || continue
      if [[ -f ${root}/targets/x86_64-linux/include/curand.h ]]; then header=${root}/targets/x86_64-linux/include
      elif [[ -f ${root}/include/curand.h ]]; then header=${root}/include
      else continue; fi
      if compgen -G "${root}/targets/x86_64-linux/lib/libcurand.so*" >/dev/null; then library=${root}/targets/x86_64-linux/lib
      elif compgen -G "${root}/lib64/libcurand.so*" >/dev/null; then library=${root}/lib64
      else continue; fi
      export CUDA_HOME=${root} CURAND_INCLUDE_DIR=${header} CURAND_LIBRARY_DIR=${library}
      break
    done
  fi
  [[ -n ${CUDA_HOME:-} && -f ${CURAND_INCLUDE_DIR:-}/curand.h ]]
  compgen -G "${CURAND_LIBRARY_DIR}/libcurand.so*" >/dev/null
  export PATH=${CUDA_HOME}/bin:${ENV}/bin:${PATH}
  export CPATH=${CURAND_INCLUDE_DIR}:${ENV}/targets/x86_64-linux/include:${CPATH:-}
  export LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
  export VLLM_USE_FLASHINFER_SAMPLER=1
}

configure_cuda_devel
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1
renderer_pid=
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == training ]]; then touch "${RUN}/STOP_TRAIN_RENDERER"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
{
  hostname; id; date -u +%FT%TZ; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  printf 'CUDA_HOME=%s\nCURAND_INCLUDE_DIR=%s\nCURAND_LIBRARY_DIR=%s\n' "${CUDA_HOME}" "${CURAND_INCLUDE_DIR}" "${CURAND_LIBRARY_DIR}"
  command -v nvcc; nvcc --version
} > "${OUT}/environment.txt"
(cd "${FROZEN}" && sha256sum -c SHA256SUMS)
"${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["status"]=="PASS" and all(d["checks"].values())' "${RUN}/gate_aligned_reward_replay.json"

if [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${ASSETS}" --host 0.0.0.0 --port 8915 --gpus 0 \
    --max-workers 1 --max-inflight 1 --admit-timeout 900 --conda-env "${ENV}" \
    > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!
  for _ in $(seq 1 180); do
    if "${PY}" -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8915/health",timeout=2)' 2>/dev/null; then break; fi
    kill -0 "${renderer_pid}"; sleep 2
  done
  timeout --kill-after=60s 90m "${PY}" scripts/r1_clean_projective_runtime_preflight.py \
    --frozen "${FROZEN}" --renderer-url http://127.0.0.1:8915/render \
    --gs-root "${ASSETS}" --output "${RUN}/runtime_preflight.json"
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8915/render\n' "${worker_ip}" > "${RUN}/renderer_endpoint.tmp"
  mv "${RUN}/renderer_endpoint.tmp" "${RUN}/renderer_endpoint.txt"
  while [[ ! -e ${RUN}/STOP_TRAIN_RENDERER ]]; do kill -0 "${renderer_pid}"; sleep 10; done
elif [[ ${MODE} == training ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  "${PY}" -c 'import json,sys;r=json.load(open(sys.argv[1]));assert r["status"]=="PASS" and r["completed"]==12' "${RUN}/runtime_preflight.json"
  export R1_RENDER_URL
  R1_RENDER_URL=$(<"${RUN}/renderer_endpoint.txt")
  MODEL=$("${PY}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["model"]["path"])' "${FROZEN}/data_gate.json")
  (cd "${MODEL}" && sha256sum -c "${MODEL}_SHA256SUMS") > "${OUT}/model_hash_check.txt"
  mkdir -p "${RUN}/pilot"
  timeout --kill-after=120s 12h "${PY}" -m vagen.r1_clean_projective_ppo \
    --config "${FROZEN}/pilot.yaml" --endpoint 8 2>&1 | tee "${RUN}/pilot/train.log"
  for step in 2 4 8; do
    checkpoint=${RUN}/pilot/checkpoints/global_step_${step}
    [[ -f ${checkpoint}/COMPLETE && -f ${checkpoint}/actor/huggingface/model.safetensors.index.json ]]
    "${PY}" - "${checkpoint}/actor/huggingface" "${RUN}/step${step}_SHA256SUMS" <<'PY'
import hashlib, pathlib, sys
root=pathlib.Path(sys.argv[1]); lines=[]
for path in sorted(p for p in root.iterdir() if p.is_file()):
    lines.append(f"{hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}  {path.name}\n")
pathlib.Path(sys.argv[2]).write_text(''.join(lines))
PY
  done
  "${PY}" - "${RUN}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
payload={"status":"PASS","updates":8,"evaluation_steps":[2,4,8],
         "checkpoints":{str(s):str(root/f"pilot/checkpoints/global_step_{s}/actor/huggingface") for s in (2,4,8)}}
(root/'pilot_training_gate.json').write_text(json.dumps(payload,indent=2)+'\n')
PY
else
  exit 2
fi
