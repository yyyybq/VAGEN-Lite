#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY="${QWEN3_VL_PYTHON:-${ROOT}/.venv-qwen3vl-runtime/bin/python}"
MODEL="${QWEN3_VL_MODEL:-/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
RENDER_ENDPOINT_FILE="${QWEN3_VL_RENDER_ENDPOINT_FILE:-${ROOT}/exps/vagen_active_spatial/sco_renderer/active_spatial_renderer_8h800_20260824_r3/endpoint.txt}"
EXPERIMENT="${ROOT}/examples/train/active_spatial/experiments/qwen3vl_active_spatial_smoke.sh"
LOG_DIR="${ROOT}/exps/vagen_active_spatial/qwen3vl_zoetrope_smoke"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/worker_$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1

echo "[qwen3vl-smoke] host=$(hostname) date=$(date -u +%FT%TZ)"
echo "[qwen3vl-smoke] python=${PY} model=${MODEL}"
nvidia-smi -L
[[ -x "${PY}" ]] || { echo "[fatal] Qwen3-VL Python env missing: ${PY}"; exit 2; }
[[ -f "${MODEL}/config.json" && -f "${MODEL}/model.safetensors.index.json" ]] || {
  echo "[fatal] Qwen3-VL checkpoint is not readable at ${MODEL}"; exit 2;
}
[[ -f "${RENDER_ENDPOINT_FILE}" ]] || { echo "[fatal] renderer endpoint file missing: ${RENDER_ENDPOINT_FILE}"; exit 2; }
RENDER_URL="$(tr -d '\r\n' < "${RENDER_ENDPOINT_FILE}")"
RENDER_AUTHORITY="${RENDER_URL#*://}"
RENDER_HOST="${RENDER_AUTHORITY%%:*}"
RENDER_PORT="${RENDER_AUTHORITY##*:}"
[[ -n "${RENDER_HOST}" ]] || { echo "[fatal] empty renderer endpoint"; exit 2; }
[[ "${RENDER_PORT}" =~ ^[0-9]+$ ]] || { echo "[fatal] invalid renderer endpoint: ${RENDER_URL}"; exit 2; }

"${PY}" "${ROOT}/scripts/check_qwen3_vl_env.py"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-/mnt/umm/shared_model/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export VAGEN_PYTHON="${PY}"
export QWEN3_VL_MODEL="${MODEL}"
export QWEN3_VL_RUN_NAME="${QWEN3_VL_RUN_NAME:-qwen3vl_active_spatial_smoke_r3}"
export QWEN3_VL_RENDER_HOST="${RENDER_HOST}"
export QWEN3_VL_RENDER_PORT="${RENDER_PORT}"
export NUM_TRAIN_GPUS=8

echo "[qwen3vl-smoke] renderer=${RENDER_HOST}; beginning one-update PPO smoke"
cd "${ROOT}"
bash "${ROOT}/examples/train/active_spatial/run_experiment.sh" "${EXPERIMENT}"
echo QWEN3VL_ACTIVE_SPATIAL_SMOKE_OK
