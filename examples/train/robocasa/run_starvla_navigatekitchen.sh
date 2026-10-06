#!/usr/bin/env bash
# Wire local NavigateKitchen LeRobot into StarVLA QwenOFT (action_dim=12).
#
# Does not mix kitchen Python 3.13 with the RoboCasa policy venv.
# LLaMA-Factory 12-D text SFT remains a smoke path; this is the VLA entry.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
STARVLA_ROOT="${STARVLA_ROOT:-$ROOT/third_party/starVLA}"
PY="${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
SRC_ROOT="${SRC_ROOT:-$ROOT/playground/Datasets/robocasa365}"
DATA_ROOT="${DATA_ROOT:-$ROOT/playground/Datasets/robocasa365}"
RELPATH="${RELPATH:-v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot}"
DATA_MIX="${DATA_MIX:-robocasa365_navigate_kitchen_pretrain_human}"
YAML="${YAML:-$STARVLA_ROOT/examples/simBenchmarks/Robocasa_365/train_files/starvla_qwenoft_navigatekitchen.yaml}"
SKIP_OVERLAY="${SKIP_OVERLAY:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
MAX_STEPS="${MAX_STEPS:-2}"
BATCH="${BATCH:-1}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-$STARVLA_ROOT/playground/Checkpoints}"
RUN_ID="${RUN_ID:-starvla_qwenoft_NavigateKitchen}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-100}"

cd "${ROOT}"

echo "[starvla] check packed 12-D LeRobot vs StarVLA concat order"
"${PY}" "${ROOT}/scripts/check_starvla_navigatekitchen.py" --data-root "${SRC_ROOT}" --relpath "${RELPATH}"

if [[ "${SKIP_OVERLAY}" != "1" ]]; then
  echo "[starvla] writable overlay -> ${DATA_ROOT}/${RELPATH}"
  "${PY}" "${ROOT}/examples/train/robocasa/prepare_starvla_lerobot_overlay.py" --src-root "${SRC_ROOT}" --dst-root "${DATA_ROOT}" --relpath "${RELPATH}"
else
  DATA_ROOT="${SRC_ROOT}"
fi

echo "[starvla] data_root_dir=${DATA_ROOT}"
echo "[starvla] data_mix=${DATA_MIX}"
echo "[starvla] action_dim=12 state_dim=16"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  echo "[starvla] CHECK_ONLY=1; skip train"
  exit 0
fi

if [[ ! -d "${STARVLA_ROOT}/starVLA" ]]; then
  echo "[fatal] StarVLA missing: ${STARVLA_ROOT}" >&2
  exit 2
fi

export PYTHONPATH="${STARVLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED=true
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-0}"
export PATH="$(dirname "${PY}"):${PATH}"
export CC="$(dirname "${PY}")/gcc"
export CXX="$(dirname "${PY}")/g++"
export CUDA_HOME="${CUDA_HOME:-$(dirname "${PY}")/..}"
# conda env prefix: PY=.../starVLA/bin/python -> CUDA_HOME=.../starVLA
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  CUDA_HOME="$(cd "$(dirname "${PY}")/.." && pwd)"
  export CUDA_HOME
fi
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1

cd "${STARVLA_ROOT}"
mkdir -p "${RUN_ROOT_DIR}"
echo "[starvla] run_root_dir=${RUN_ROOT_DIR} run_id=${RUN_ID}"
NUM_GPUS="${NUM_GPUS:-1}"
DS_CFG="${DS_CFG:-$STARVLA_ROOT/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
echo "[starvla] accelerate+deepspeed num_gpus=${NUM_GPUS} cfg=${DS_CFG}"
# official StarVLA path: accelerate launch + ZeRO-2. raw python hits mpi4py.
ACCEL="$(dirname "${PY}")/accelerate"
exec "${ACCEL}" launch \
  --config_file "${DS_CFG}" \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${YAML}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --framework.action_model.action_dim 12 \
  --framework.action_model.state_dim 16 \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
