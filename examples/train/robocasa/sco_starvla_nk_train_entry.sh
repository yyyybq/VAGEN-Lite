#!/usr/bin/env bash
# SCO H800 8-GPU official StarVLA-OFT NavigateKitchen train.
# Isolated starVLA env + Qwen3-VL-4B + DeepSpeed ZeRO-2 + real flash-attn.
# LLaMA-Factory 12-D text SFT is not this path.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
STARVLA_ROOT="${STARVLA_ROOT:-$ROOT/third_party/starVLA}"
STARVLA_ENV="${STARVLA_ENV:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA}"
PY="${PY:-$STARVLA_ENV/bin/python}"
WEIGHTS="${WEIGHTS:-$STARVLA_ROOT/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$STARVLA_ROOT/playground/Checkpoints}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-starvla_qwenoft_NavigateKitchen_${STAMP}}"
LOG_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
LOG="${LOG_DIR}/train.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[train] host=$(hostname) date=$(date -u +%FT%TZ)"
echo "[train] cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# shellcheck source=setup_starvla_env.sh
source "${ROOT}/examples/train/robocasa/setup_starvla_env.sh"

[[ -x "${PY}" ]] || { echo "[fatal] python missing: ${PY}"; exit 2; }
[[ -f "${WEIGHTS}/model-00001-of-00002.safetensors" ]] || { echo "[fatal] weights missing: ${WEIGHTS}"; exit 2; }
[[ -f "${WEIGHTS}/model-00002-of-00002.safetensors" ]] || { echo "[fatal] weights missing shard2: ${WEIGHTS}"; exit 2; }
[[ -d "${STARVLA_ROOT}/starVLA" ]] || { echo "[fatal] StarVLA missing: ${STARVLA_ROOT}"; exit 2; }

export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED=true
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-0}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Official RoboCasa365 NavigateKitchen recipe (8x H800).
NUM_GPUS="${NUM_GPUS:-$(${PY} -c 'import torch;print(torch.cuda.device_count())')}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH="${BATCH:-8}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"

echo "[train] NUM_GPUS=${NUM_GPUS} MAX_STEPS=${MAX_STEPS} BATCH=${BATCH}"
echo "[train] RUN_ID=${RUN_ID} RUN_ROOT_DIR=${RUN_ROOT_DIR}"

cd "${ROOT}"
NUM_GPUS="${NUM_GPUS}" \
MAX_STEPS="${MAX_STEPS}" \
BATCH="${BATCH}" \
SAVE_EVERY="${SAVE_EVERY}" \
EVAL_EVERY="${EVAL_EVERY}" \
LOG_EVERY="${LOG_EVERY}" \
RUN_ID="${RUN_ID}" \
RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
PY="${PY}" \
STARVLA_ROOT="${STARVLA_ROOT}" \
ROOT="${ROOT}" \
  bash "${ROOT}/examples/train/robocasa/run_starvla_navigatekitchen.sh"
