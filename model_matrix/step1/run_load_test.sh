#!/usr/bin/env bash
# Step 1 of the model line: the one-card load test as one sco job (1 H800).  Four stages, one process each, so
# every stage starts with an empty card: Qwen3-VL-8B-Instruct through Transformers and through vLLM (the RL
# rollout engine), Cambrian-S through the repository's adapter (Transformers) and its vLLM plugin.
#
#   bash run_load_test.sh OUT_DIR            (STAGES="qwen3vl_vllm cambrian_vllm" bash ... reruns only those)
#
# Nothing is written outside OUT_DIR (caches go to OUT_DIR/tmp, removed on exit); the job runs as root, so OUT_DIR
# is handed back to its owner on exit.
set -uo pipefail
OUT=${1:?output directory}
WT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY_MAIN=${ENV_ROOT}/bin/python
PY_Q3=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/.venv-qwen3vl-runtime/bin/python
QWEN3VL=/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
# the original Cambrian-S weights are not on the cluster; the closest are the August RL checkpoints (same
# architecture, cambrian_qwen), here the earliest one (50 updates at lr 5e-7)
CAMBRIAN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/models/historical/cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full_step50
mkdir -p "${OUT}/logs" "${OUT}/tmp"
OWNER=$(stat -c %u:%g "${OUT}")
trap 'status=$?; [[ -n ${mon_pid:-} ]] && kill "${mon_pid}" 2>/dev/null; rm -rf "${OUT}/tmp"; printf "{\"exit_status\":%s,\"ended_utc\":\"%s\"}\n" "${status}" "$(date -u +%FT%TZ)" > "${OUT}/logs/exit.json"; chown -R "${OWNER}" "${OUT}" 2>/dev/null || true; exit ${status}' EXIT
exec > >(tee -a "${OUT}/logs/job-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "${WT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export PYTHONPATH="${WT}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
WHEEL_ROOT=${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=${ENV_ROOT}
# vLLM's Triton kernels are compiled at start-up and need a C compiler (as sco_interiorgs_v10_train.sh sets)
export CC=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++
export PATH="${CUDA_HOME}/bin:${ENV_ROOT}/bin:${PATH}"
export CPATH="${WHEEL_ROOT}/include:${ENV_ROOT}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${WHEEL_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${WHEEL_ROOT}/lib:${ENV_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${OUT}/tmp/hf_home" HF_HUB_CACHE=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub
export SIGLIP_CACHE=/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub
export VLLM_CACHE_ROOT="${OUT}/tmp/vllm" XDG_CACHE_HOME="${OUT}/tmp/xdg" TRITON_CACHE_DIR="${OUT}/tmp/triton"
export TORCH_EXTENSIONS_DIR="${OUT}/tmp/torch_extensions"
export CAMBRIAN_SRC=/mnt/umm/users/yinbaiqiao/cambrian-s
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || true
git -C "${WT}" rev-parse HEAD || true
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 2000 > "${OUT}/logs/gpu_mem.csv" &
mon_pid=$!

STAGES=${STAGES:-qwen3vl_hf qwen3vl_vllm cambrian_hf cambrian_vllm}  # a rerun can name only some
stage() {  # name python kind backend model
  [[ " ${STAGES} " == *" $1 "* ]] || return 0
  echo "=== $(date -u +%FT%TZ) stage $1"
  timeout 1500 "$2" model_matrix/step1/load_test.py --kind "$3" --backend "$4" --model "$5" --out "${OUT}/$1.json"
  echo "=== $(date -u +%FT%TZ) stage $1 exit $?"
  sleep 5
}
stage qwen3vl_hf "${PY_Q3}" qwen3vl hf "${QWEN3VL}"
stage qwen3vl_vllm "${PY_Q3}" qwen3vl vllm "${QWEN3VL}"
stage cambrian_hf "${PY_MAIN}" cambrian hf "${CAMBRIAN}"
VLLM_ENABLE_V1_MULTIPROCESSING=0 stage cambrian_vllm "${PY_MAIN}" cambrian vllm "${CAMBRIAN}"
