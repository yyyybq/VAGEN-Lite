#!/usr/bin/env bash
# Evaluation-only worker. It never invokes PPO or mutates the step-1 checkpoint.
set -euo pipefail
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
MODE=${1:?preflight, renderer, or evaluation}
RUN=${2:?existing accepted run}
FROZEN=${RUN}/frozen
ASSETS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0/assets/ready
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
WORK=${PWD}
BASE=$(${PY} -c 'import json,sys;print(json.load(open(sys.argv[1]))["model"]["path"])' "${FROZEN}/data_gate.json")
STEP1=${RUN}/smoke/checkpoints/global_step_1/actor/huggingface
export PYTHONDONTWRITEBYTECODE=1 NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${WORK}/verl:${WORK}/scripts:${WORK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2
export INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10

configure_cuda_devel() {
  local root header library
  local wheel_root=${ENV}/lib/python3.12/site-packages/nvidia/curand
  if [[ -x ${ENV}/bin/nvcc && -f ${wheel_root}/include/curand.h ]] && compgen -G "${wheel_root}/lib/libcurand.so*" >/dev/null; then
    export CUDA_HOME=${ENV}
    export CURAND_INCLUDE_DIR=${wheel_root}/include
    export CURAND_LIBRARY_DIR=${wheel_root}/lib
    export PATH=${ENV}/bin:${PATH}
    export CPATH=${CURAND_INCLUDE_DIR}:${ENV}/targets/x86_64-linux/include:${CPATH:-}
    export LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
    export VLLM_USE_FLASHINFER_SAMPLER=1
    return 0
  fi
  for root in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12.1; do
    [[ -x ${root}/bin/nvcc ]] || continue
    if [[ -f ${root}/targets/x86_64-linux/include/curand.h ]]; then
      header=${root}/targets/x86_64-linux/include
    elif [[ -f ${root}/include/curand.h ]]; then
      header=${root}/include
    else
      continue
    fi
    if compgen -G "${root}/targets/x86_64-linux/lib/libcurand.so*" >/dev/null; then
      library=${root}/targets/x86_64-linux/lib
    elif compgen -G "${root}/lib64/libcurand.so*" >/dev/null; then
      library=${root}/lib64
    else
      continue
    fi
    export CUDA_HOME=${root}
    export CURAND_INCLUDE_DIR=${header}
    export CURAND_LIBRARY_DIR=${library}
    export PATH=${root}/bin:${ENV}/bin:${PATH}
    export CPATH=${header}:${ENV}/targets/x86_64-linux/include:${CPATH:-}
    export LIBRARY_PATH=${library}:${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=${library}:${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
    export VLLM_USE_FLASHINFER_SAMPLER=1
    return 0
  done
  echo 'No complete system CUDA toolkit containing nvcc, curand.h and libcurand was found.' >&2
  return 12
}

if [[ ${MODE} == renderer ]]; then
  export PATH=${ENV}/bin:${PATH} CUDA_HOME=${ENV}
  export CPATH=${ENV}/targets/x86_64-linux/include:${CPATH:-}
  export LIBRARY_PATH=${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
else
  configure_cuda_devel
fi

ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/eval_${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1
renderer_pid=
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == evaluation ]]; then touch "${RUN}/STOP_EVAL_RENDERER"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
{
  hostname; id; date -u +%FT%TZ; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  printf 'mode=%s\nCUDA_HOME=%s\nCURAND_INCLUDE_DIR=%s\nCURAND_LIBRARY_DIR=%s\nCPATH=%s\nLIBRARY_PATH=%s\n' "${MODE}" "${CUDA_HOME}" "${CURAND_INCLUDE_DIR:-}" "${CURAND_LIBRARY_DIR:-}" "${CPATH:-}" "${LIBRARY_PATH:-}"
  if [[ ${MODE} != renderer ]]; then
    command -v nvcc; nvcc --version
    find "${CURAND_INCLUDE_DIR}" "${CURAND_LIBRARY_DIR}" -name curand.h -o -name 'libcurand.so*'
  fi
} > "${OUT}/environment.txt"
[[ -f ${STEP1}/model.safetensors.index.json ]]
[[ -f "${RUN}/smoke/checkpoints/global_step_1/COMPLETE" ]]
"${PY}" -c 'import json,sys;assert json.load(open(sys.argv[1]))["status"]=="PASS"' "${RUN}/acceptance_gate.json"

if [[ ${MODE} == preflight ]]; then
  export CUDA_VISIBLE_DEVICES=0
  mkdir -p "${RUN}/eval_preflight_v5"
  for model_key in base step1; do
    if [[ ${model_key} == base ]]; then model_path=${BASE}; hashes=${BASE}_SHA256SUMS
    else model_path=${STEP1}; hashes=${RUN}/step1_SHA256SUMS; fi
    marker=${RUN}/eval_preflight_v5/${model_key}/preflight.json
    setsid timeout --kill-after=60s 45m "${PY}" scripts/r1_vllm_eval_preflight.py \
      --model-key "${model_key}" --model-path "${model_path}" --model-sha256s "${hashes}" \
      --output-dir "${RUN}/eval_preflight_v5/${model_key}" &
    child=$!
    passed=0
    for _ in $(seq 1 540); do
      if "${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["status"]=="PASS" and d["sampled_multimodal_inference"] and d["task_context_check"]["task_present"]' "${marker}" 2>/dev/null; then
        passed=1; break
      fi
      kill -0 "${child}" 2>/dev/null || break
      sleep 5
    done
    if [[ ${passed} != 1 ]]; then wait "${child}"; exit 21; fi
    # vLLM V1 can leave descendants holding stdout after the parent has
    # atomically written PASS. Terminate the isolated process group only.
    kill -TERM -- "-${child}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${child}" 2>/dev/null || true
    wait "${child}" 2>/dev/null || true
    "${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["status"]=="PASS"' "${marker}"
  done
elif [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "${ASSETS}" --host 0.0.0.0 --port 8915 --gpus 0 --max-workers 1 --max-inflight 1 --admit-timeout 900 --conda-env "${ENV}" > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!
  for _ in $(seq 1 180); do
    if "${PY}" -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8915/health",timeout=2)' 2>/dev/null; then break; fi
    kill -0 "${renderer_pid}"; sleep 2
  done
  "${PY}" -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8915/health",timeout=10)'
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8915/render\n' "${worker_ip}" > "${RUN}/eval_renderer_endpoint.tmp"
  mv "${RUN}/eval_renderer_endpoint.tmp" "${RUN}/eval_renderer_endpoint.txt"
  while [[ ! -e ${RUN}/STOP_EVAL_RENDERER ]]; do kill -0 "${renderer_pid}"; sleep 10; done
elif [[ ${MODE} == evaluation ]]; then
  export CUDA_VISIBLE_DEVICES=0
  for model_key in base step1; do
    "${PY}" -c 'import json,sys;assert json.load(open(sys.argv[1]))["status"]=="PASS"' "${RUN}/eval_preflight_v5/${model_key}/preflight.json"
  done
  R1_RENDER_URL=$(<"${RUN}/eval_renderer_endpoint.txt"); export R1_RENDER_URL
  "${PY}" -c 'import urllib.request,os;urllib.request.urlopen(os.environ["R1_RENDER_URL"].replace("/render","/health"),timeout=10)'
  for model_key in base step1; do
    if [[ ${model_key} == base ]]; then model_path=${BASE}; hashes=${BASE}_SHA256SUMS
    else model_path=${STEP1}; hashes=${RUN}/step1_SHA256SUMS; fi
    output=${RUN}/paired_eval_v2/${model_key}
    setsid timeout --kill-after=60s 150m "${PY}" scripts/r1_run_canonical_dev_eval32.py \
      --protocol "${FROZEN}/eval_protocol.json" --model-key "${model_key}" \
      --model-path "${model_path}" --model-sha256s "${hashes}" \
      --renderer-url "${R1_RENDER_URL}" --gs-root "${ASSETS}" \
      --output-dir "${output}" --max-infrastructure-attempts 1 &
    child=$!
    complete=0
    for _ in $(seq 1 1800); do
      if "${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["complete"]==32 and d["infrastructure_errors"]==0' "${output}/summary.json" 2>/dev/null; then
        complete=1; break
      fi
      kill -0 "${child}" 2>/dev/null || break
      sleep 5
    done
    if [[ ${complete} != 1 ]]; then wait "${child}"; exit 22; fi
    kill -TERM -- "-${child}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${child}" 2>/dev/null || true
    wait "${child}" 2>/dev/null || true
    "${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["complete"]==32 and d["infrastructure_errors"]==0' "${output}/summary.json"
  done
  "${PY}" scripts/r1_compare_task_context_eval.py --run "${RUN}" --eval-root paired_eval_v2
else
  exit 2
fi
