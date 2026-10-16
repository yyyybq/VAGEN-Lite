#!/usr/bin/env bash
# Independent vLLM preflight, renderer, and fixed-32 paired evaluation jobs.
set -euo pipefail
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
MODE=${1:?preflight, renderer, or evaluation}
RUN=${2:?gate-aligned pilot run}
EVAL_TAG=${R1_EVAL_TAG:-}
PREFLIGHT_ROOT=${RUN}/eval_preflight${EVAL_TAG}
EVAL_ROOT=paired_eval${EVAL_TAG}
RENDER_ENDPOINT=${RUN}/eval_renderer_endpoint${EVAL_TAG}.txt
STOP_RENDERER=${RUN}/STOP_EVAL_RENDERER${EVAL_TAG}
FROZEN=${RUN}/frozen
ASSETS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0/assets/ready
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
WORK=${PWD}
BASE=$(${PY} -c 'import json,sys;print(json.load(open(sys.argv[1]))["model"]["path"])' "${FROZEN}/data_gate.json")
export PYTHONDONTWRITEBYTECODE=1 NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
export PYTHONPATH=${WORK}/verl:${WORK}/scripts:${WORK}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2 INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10

configure_cuda_devel() {
  local root header library wheel_root=${ENV}/lib/python3.12/site-packages/nvidia/curand
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
      export CUDA_HOME=${root} CURAND_INCLUDE_DIR=${header} CURAND_LIBRARY_DIR=${library}; break
    done
  fi
  [[ -f ${CURAND_INCLUDE_DIR}/curand.h ]]; compgen -G "${CURAND_LIBRARY_DIR}/libcurand.so*" >/dev/null
  export PATH=${CUDA_HOME}/bin:${ENV}/bin:${PATH}
  export CPATH=${CURAND_INCLUDE_DIR}:${ENV}/targets/x86_64-linux/include:${CPATH:-}
  export LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
  export VLLM_USE_FLASHINFER_SAMPLER=1
}
configure_cuda_devel

ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/eval_${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"; exec > >(tee -a "${OUT}/worker.log") 2>&1
renderer_pid=
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == evaluation ]]; then touch "${STOP_RENDERER}"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT; trap 'exit 143' TERM INT
"${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["status"]=="PASS" and d["updates"]==8' "${RUN}/pilot_training_gate.json"

model_path() {
  case "$1" in
    base) printf '%s\n' "${BASE}" ;;
    step2) printf '%s\n' "${RUN}/pilot/checkpoints/global_step_2/actor/huggingface" ;;
    step4) printf '%s\n' "${RUN}/pilot/checkpoints/global_step_4/actor/huggingface" ;;
    step8) printf '%s\n' "${RUN}/pilot/checkpoints/global_step_8/actor/huggingface" ;;
  esac
}
model_hashes() {
  if [[ $1 == base ]]; then printf '%s\n' "${BASE}_SHA256SUMS"
  else printf '%s/step%s_SHA256SUMS\n' "${RUN}" "${1#step}"; fi
}

if [[ ${MODE} == preflight ]]; then
  export CUDA_VISIBLE_DEVICES=0; mkdir -p "${PREFLIGHT_ROOT}"
  for key in base step2 step4 step8; do
    setsid timeout --kill-after=60s 45m "${PY}" scripts/r1_vllm_eval_preflight.py \
      --model-key "${key}" --model-path "$(model_path "${key}")" --model-sha256s "$(model_hashes "${key}")" \
      --output-dir "${PREFLIGHT_ROOT}/${key}" &
    child=$!; marker=${PREFLIGHT_ROOT}/${key}/preflight.json; passed=0
    for _ in $(seq 1 540); do
      if "${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["status"]=="PASS" and d["sampled_multimodal_inference"] and d["task_context_check"]["task_present"]' "${marker}" 2>/dev/null; then passed=1; break; fi
      kill -0 "${child}" 2>/dev/null || break; sleep 5
    done
    if [[ ${passed} != 1 ]]; then wait "${child}"; exit 21; fi
    kill -TERM -- "-${child}" 2>/dev/null || true; sleep 2; kill -KILL -- "-${child}" 2>/dev/null || true; wait "${child}" 2>/dev/null || true
  done
elif [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "${ASSETS}" --host 0.0.0.0 --port 8915 --gpus 0 --max-workers 1 --max-inflight 1 --admit-timeout 900 --conda-env "${ENV}" > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!
  for _ in $(seq 1 180); do
    if "${PY}" -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8915/health",timeout=2)' 2>/dev/null; then break; fi
    kill -0 "${renderer_pid}"; sleep 2
  done
  worker_ip=$(hostname -I | awk '{print $1}'); printf 'http://%s:8915/render\n' "${worker_ip}" > "${RENDER_ENDPOINT}.tmp"; mv "${RENDER_ENDPOINT}.tmp" "${RENDER_ENDPOINT}"
  while [[ ! -e ${STOP_RENDERER} ]]; do kill -0 "${renderer_pid}"; sleep 10; done
elif [[ ${MODE} == evaluation ]]; then
  export CUDA_VISIBLE_DEVICES=0
  for key in base step2 step4 step8; do "${PY}" -c 'import json,sys;assert json.load(open(sys.argv[1]))["status"]=="PASS"' "${PREFLIGHT_ROOT}/${key}/preflight.json"; done
  R1_RENDER_URL=$(<"${RENDER_ENDPOINT}"); export R1_RENDER_URL
  for key in base step2 step4 step8; do
    output=${RUN}/${EVAL_ROOT}/${key}
    setsid timeout --kill-after=60s 150m "${PY}" scripts/r1_run_canonical_dev_eval32.py \
      --protocol "${FROZEN}/eval_protocol.json" --model-key "${key}" --model-path "$(model_path "${key}")" \
      --model-sha256s "$(model_hashes "${key}")" --renderer-url "${R1_RENDER_URL}" --gs-root "${ASSETS}" \
      --output-dir "${output}" --max-infrastructure-attempts 1 &
    child=$!; complete=0
    for _ in $(seq 1 1800); do
      if "${PY}" -c 'import json,sys;d=json.load(open(sys.argv[1]));assert d["complete"]==32 and d["infrastructure_errors"]==0' "${output}/summary.json" 2>/dev/null; then complete=1; break; fi
      kill -0 "${child}" 2>/dev/null || break; sleep 5
    done
    if [[ ${complete} != 1 ]]; then wait "${child}"; exit 22; fi
    kill -TERM -- "-${child}" 2>/dev/null || true; sleep 2; kill -KILL -- "-${child}" 2>/dev/null || true; wait "${child}" 2>/dev/null || true
  done
  "${PY}" scripts/r1_compare_gate_aligned_pilot.py --run "${RUN}" --eval-root "${EVAL_ROOT}"
else
  exit 2
fi
