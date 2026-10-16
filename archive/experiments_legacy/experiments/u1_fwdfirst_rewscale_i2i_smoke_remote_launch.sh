#!/usr/bin/env bash
# U1 Plan-B 1-step smoke with remote HTTP render (no local render GPU).
set -euo pipefail
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}"
export U1_NUM_TRAIN_GPUS="${U1_NUM_TRAIN_GPUS:-6}"
export U1_TRAIN_BATCH_SIZE="${U1_TRAIN_BATCH_SIZE:-6}"
export U1_PPO_MINI_BATCH_SIZE="${U1_PPO_MINI_BATCH_SIZE:-6}"
export U1_MINI_BATCH_SIZE="${U1_MINI_BATCH_SIZE:-6}"
export U1_GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.28}"
export U1_RENDER_MODE=remote
export U1_RENDER_PROTOCOL=http
export U1_RENDER_HOST="${U1_RENDER_HOST:-10.119.18.163}"
export U1_RENDER_PORT="${U1_RENDER_PORT:-8767}"
# Metrics-only FM avoids FSDP-offload empty-storage crash in twopass FM
export U1_FM_BACKPROP="${U1_FM_BACKPROP:-0}"
export U1_FM_USE_UND_KV="${U1_FM_USE_UND_KV:-0}"

code=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 "http://${U1_RENDER_HOST}:${U1_RENDER_PORT}/health" || true)
if [ "$code" != "200" ]; then
  echo "ERROR: remote render health failed: http://${U1_RENDER_HOST}:${U1_RENDER_PORT}/health -> ${code}"
  exit 1
fi
echo "[u1_smoke_remote] render OK http://${U1_RENDER_HOST}:${U1_RENDER_PORT} gpus=${CUDA_VISIBLE_DEVICES} n=${U1_NUM_TRAIN_GPUS}"

bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke.sh
