#!/usr/bin/env bash
# 1-GPU StarVLA-OFT smoke on NavigateKitchen (action_dim=12).
# Uses vagen-lite (not kitchen 3.13, not the RoboCasa policy venv).
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
STARVLA_ROOT="${STARVLA_ROOT:-$ROOT/third_party/starVLA}"
PY="${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
SRC_ROOT="${SRC_ROOT:-$ROOT/playground/Datasets/robocasa365}"
DATA_ROOT="${DATA_ROOT:-$ROOT/playground/Datasets/robocasa365}"
RELPATH="${RELPATH:-v1.0/pretrain/atomic/NavigateKitchen/20250821/lerobot}"
YAML="${YAML:-$ROOT/examples/train/robocasa/starvla_qwenoft_navigatekitchen_smoke.yaml}"
OUT_ROOT="${OUT_ROOT:-$ROOT/playground/Checkpoints/starvla_nk_smoke}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${STAMP}"
LOG="${RUN_DIR}/smoke.log"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[smoke] host=$(hostname) date=$(date -u +%FT%TZ)"
echo "[smoke] python=${PY}"
echo "[smoke] cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

[[ -x "${PY}" ]] || { echo "[fatal] python missing: ${PY}"; exit 2; }
[[ -f "${YAML}" ]] || { echo "[fatal] yaml missing: ${YAML}"; exit 2; }
[[ -d "${STARVLA_ROOT}/starVLA" ]] || { echo "[fatal] StarVLA missing: ${STARVLA_ROOT}"; exit 2; }

export PATH="$(dirname "${PY}"):${PATH}"
export PYTHONPATH="${STARVLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export STARVLA_DISABLE_DEEPSPEED=1
export NO_ALBUMENTATIONS_UPDATE=1

echo "[1/4] check LeRobot 12-D contract"
"${PY}" "${ROOT}/scripts/check_starvla_navigatekitchen.py" --data-root "${SRC_ROOT}" --relpath "${RELPATH}"

echo "[2/4] writable overlay"
"${PY}" "${ROOT}/examples/train/robocasa/prepare_starvla_lerobot_overlay.py" --src-root "${SRC_ROOT}" --dst-root "${DATA_ROOT}" --relpath "${RELPATH}"

echo "[3/4] dataloader one-batch smoke"
cd "${STARVLA_ROOT}"
"${PY}" "${ROOT}/scripts/smoke_starvla_dataloader.py" --config_yaml "${YAML}" --data_root_dir "${DATA_ROOT}"

echo "[4/4] 2-step QwenOFT train"
cd "${STARVLA_ROOT}"
"${PY}" starVLA/training/train_starvla.py --config_yaml "${YAML}" --datasets.vla_data.data_root_dir "${DATA_ROOT}" --run_root_dir "${RUN_DIR}" --run_id nk_smoke_${STAMP} --trainer.max_train_steps 2 --trainer.save_interval 2 --datasets.vla_data.per_device_batch_size 1

echo "[smoke] done RUN_DIR=${RUN_DIR}"
ls -la "${RUN_DIR}" || true
