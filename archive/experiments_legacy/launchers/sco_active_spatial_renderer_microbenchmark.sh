#!/usr/bin/env bash
set -euo pipefail

RUN=${1:?benchmark run root}
GPU_COUNT=${2:?1, 2, 4, or all}
PACKAGE_ROOT=${PWD}
R1_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0
MANIFEST=${R1_ROOT}/frozen_v1/train.jsonl
ASSETS=${R1_ROOT}/assets/ready
ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV_ROOT}/bin/python
VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/torch_extensions/gsplat_cuda

[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
[[ ${GPU_COUNT} == 1 || ${GPU_COUNT} == 2 || ${GPU_COUNT} == 4 || ${GPU_COUNT} == all ]] || exit 2
if [[ ${GPU_COUNT} == all ]]; then
  TOPOLOGIES="1 2 4"
else
  TOPOLOGIES=${GPU_COUNT}
fi
[[ -f ${MANIFEST} && -d ${ASSETS} && -f ${VALIDATED_GSPLAT_CACHE}/gsplat_cuda.so ]]

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NO_PROXY='*' no_proxy='*'
export PYTHONPATH="${PACKAGE_ROOT}/verl:${PACKAGE_ROOT}/scripts:${PACKAGE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=0

WHEEL_ROOT=${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=${ENV_ROOT}
export CURAND_INCLUDE_DIR=${WHEEL_ROOT}/include
export CURAND_LIBRARY_DIR=${WHEEL_ROOT}/lib
export PATH="${CUDA_HOME}/bin:${ENV_ROOT}/bin:${PATH}"
export CPATH="${CURAND_INCLUDE_DIR}:${ENV_ROOT}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/worker/attempts/${ATTEMPT}
mkdir -p "${OUT}" "${RUN}/results"
exec > >(tee -a "${OUT}/worker.log") 2>&1

service_pid=
sampler_pid=
cleanup() {
  status=$?
  trap - EXIT
  if [[ -n ${sampler_pid} ]]; then kill "${sampler_pid}" 2>/dev/null || true; wait "${sampler_pid}" 2>/dev/null || true; fi
  if [[ -n ${service_pid} ]]; then kill "${service_pid}" 2>/dev/null || true; wait "${service_pid}" 2>/dev/null || true; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

{
  hostname
  id
  date -u +%FT%TZ
  "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "${OUT}/environment.txt"

for gpu_count in ${TOPOLOGIES}; do
  case ${gpu_count} in
    1) gpu_list=0 ;;
    2) gpu_list=0,1 ;;
    4) gpu_list=0,1,2,3 ;;
  esac
  topology=${gpu_count}gpu_12workers
  topology_out=${OUT}/${topology}
  mkdir -p "${topology_out}"
  export TORCH_EXTENSIONS_DIR=${topology_out}/torch_extensions
  mkdir -p "${TORCH_EXTENSIONS_DIR}"
  cp -a "${VALIDATED_GSPLAT_CACHE}" "${TORCH_EXTENSIONS_DIR}/gsplat_cuda"

  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${ASSETS}" --host 127.0.0.1 --port 8915 --gpus "${gpu_list}" \
    --max-workers 12 --max-inflight 12 --admit-timeout 900 --conda-env "${ENV_ROOT}" \
    > "${topology_out}/renderer.log" 2>&1 &
  service_pid=$!
  ready=false
  for _ in $(seq 1 180); do
    if "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)' 2>/dev/null; then
      ready=true
      break
    fi
    kill -0 "${service_pid}"
    sleep 2
  done
  [[ ${ready} == true ]]

  printf 'utc,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,utilization_memory_pct,power_w\n' > "${topology_out}/gpu_resources.csv"
  (
    while :; do
      ts=$(date -u +%FT%TZ.%N)
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
        --format=csv,noheader,nounits | sed "s/^/${ts},/" >> "${topology_out}/gpu_resources.csv" || true
      sleep 1
    done
  ) &
  sampler_pid=$!

  "${PY}" -u scripts/active_spatial_renderer_microbenchmark.py \
    --endpoint http://127.0.0.1:8915/render \
    --manifest "${MANIFEST}" --gpu-count "${gpu_count}" \
    --max-wall-seconds 300 --output "${RUN}/results/${topology}.json"

  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
  sampler_pid=
  kill "${service_pid}" 2>/dev/null || true
  wait "${service_pid}" 2>/dev/null || true
  service_pid=
  sleep 3
done
