#!/usr/bin/env bash
# One-card job: the 3DGS render service and our env's evaluation for one model (diagnostics/ownenv_eval.py).
#
#   bash eval_external/diagnostics/ownenv_job.sh NAME MODEL_CODE [normal|shuffle|drop]
#
# The set-up is the "eval" mode of examples/train/active_spatial/sco_interiorgs_v10_train.sh (render service on
# port 8915 of the same card, the validated gsplat build, the same environment variables), with two changes:
# the code directory is this worktree (VAGEN-Lite-eval, not the main checkout, which is never touched) and the
# Python env is eval-external (a clone of vagen-lite; nothing is installed into vagen-lite).  verl is imported
# read-only from the main checkout's submodule (the worktree has no submodule checkout).
set -euo pipefail
NAME=${1:?result name}
MODEL=${2:?model code}
MODE=${3:-normal}
WORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-eval
ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external
PY=${ENV_ROOT}/bin/python
GS_ROOT=/mnt/umm/users/yinbaiqiao/InteriorGS_ply
VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007
LOGS=/mnt/umm/users/yinbaiqiao/eval_external_assets/runs/ownenv/logs/${NAME}
mkdir -p "${LOGS}"
cd "${WORKTREE}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export PYTHONPATH="${WORKTREE}/verl:${WORKTREE}/scripts:${WORKTREE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=${LOGS}/torch_extensions
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2 INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10
WHEEL_ROOT=${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=${ENV_ROOT} CURAND_INCLUDE_DIR=${WHEEL_ROOT}/include CURAND_LIBRARY_DIR=${WHEEL_ROOT}/lib
export PATH="${CUDA_HOME}/bin:${ENV_ROOT}/bin:${PATH}"
export CPATH="${CURAND_INCLUDE_DIR}:${ENV_ROOT}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${TORCH_EXTENSIONS_DIR}"
[[ -d ${TORCH_EXTENSIONS_DIR}/gsplat_cuda ]] || cp -a "${VALIDATED_GSPLAT_CACHE}" "${TORCH_EXTENSIONS_DIR}/gsplat_cuda"
bash examples/train/active_spatial/start_gs_render_http_service.sh \
  --gs-root "${GS_ROOT}" --host 0.0.0.0 --port 8915 --gpus 0 \
  --max-workers 12 --max-inflight 12 --admit-timeout 900 --conda-env "${ENV_ROOT}" \
  > "${LOGS}/renderer_service.log" 2>&1 &
renderer_pid=$!
trap 'kill "${renderer_pid}" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)' 2>/dev/null && break
  kill -0 "${renderer_pid}"
  sleep 2
done
"${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)'
export R1_RENDER_URL=http://127.0.0.1:8915/render
"${PY}" -m eval_external.diagnostics.ownenv_eval --model "${MODEL}" --name "${NAME}" --history-mode "${MODE}"
