#!/usr/bin/env bash
# Launch U1 Plan-B short formal run: 8 train GPUs + remote render.
set -euo pipefail
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export U1_FM_BACKPROP="${U1_FM_BACKPROP:-1}"
export U1_FM_USE_UND_KV="${U1_FM_USE_UND_KV:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export U1_NUM_TRAIN_GPUS="${U1_NUM_TRAIN_GPUS:-8}"
export U1_TRAIN_BATCH_SIZE="${U1_TRAIN_BATCH_SIZE:-8}"
export U1_PPO_MINI_BATCH_SIZE="${U1_PPO_MINI_BATCH_SIZE:-8}"
export U1_MINI_BATCH_SIZE="${U1_MINI_BATCH_SIZE:-8}"
export U1_GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.28}"
export U1_RENDER_HOST="${U1_RENDER_HOST:-10.119.18.163}"
export U1_RENDER_PORT="${U1_RENDER_PORT:-8767}"
export U1_TOTAL_STEPS="${U1_TOTAL_STEPS:-30}"

# Preflight remote render
code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 "http://${U1_RENDER_HOST}:${U1_RENDER_PORT}/health" || true)
if [ "$code" != "200" ]; then
  echo "ERROR: remote render health check failed: http://${U1_RENDER_HOST}:${U1_RENDER_PORT}/health -> ${code}"
  exit 1
fi
echo "[u1_short] remote render OK at http://${U1_RENDER_HOST}:${U1_RENDER_PORT} (health=${code})"
echo "[u1_short] FM backprop=${U1_FM_BACKPROP} und_kv=${U1_FM_USE_UND_KV} steps=${U1_TOTAL_STEPS}"

bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_short.sh
