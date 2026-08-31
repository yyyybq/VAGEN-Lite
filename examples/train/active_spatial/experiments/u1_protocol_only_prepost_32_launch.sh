#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
OUT="${ROOT}/exps/vagen_active_spatial/protocol_only_prepost"
MODEL="${SENSENOVA_U1_MODEL_PATH:-/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT}"
cd "${ROOT}"

export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export U1_FM_BACKPROP=0
export U1_FM_USE_UND_KV=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export U1_NUM_TRAIN_GPUS=8
export U1_TRAIN_BATCH_SIZE=8
export U1_PPO_MINI_BATCH_SIZE=8
export U1_MINI_BATCH_SIZE=8
export U1_GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.28}"
export U1_RENDER_HOST="${U1_RENDER_HOST:-10.119.27.217}"
export U1_RENDER_PORT="${U1_RENDER_PORT:-8768}"

mkdir -p "${OUT}/checkpoints" "${OUT}/sco_submission"
if [ -e "${OUT}/checkpoints/global_step_8" ] || [ -e "${OUT}/checkpoints/latest_checkpointed_iteration.txt" ]; then
  echo "ERROR: protocol_only_prepost training artifacts already exist; refusing non-independent reuse" >&2
  exit 2
fi

# The immutable base is independently loadable through this explicit path.
if [ ! -e "${OUT}/checkpoints/base" ]; then
  ln -s "${MODEL}" "${OUT}/checkpoints/base"
fi

code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 \
  "http://${U1_RENDER_HOST}:${U1_RENDER_PORT}/health" || true)
if [ "${code}" != "200" ]; then
  echo "ERROR: remote renderer health check failed (${code})" >&2
  exit 1
fi

echo "[phase4b] renderer=http://${U1_RENDER_HOST}:${U1_RENDER_PORT}"
echo "[phase4b] fixed_steps=32 save_freq=8 resume=disable"
echo "[phase4b] U1_FM_BACKPROP=${U1_FM_BACKPROP}; no new SFT/CE/auxiliary loss"

bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/u1_protocol_only_prepost_32.sh
