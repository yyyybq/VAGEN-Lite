#!/usr/bin/env bash
# Launch U1 Plan-B PPO 1-step smoke on a full 8xH800 node.
# Layout: 7 train GPUs + 1 local render GPU (7).
set -euo pipefail
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH:-}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export U1_NUM_TRAIN_GPUS="${U1_NUM_TRAIN_GPUS:-7}"
export U1_RENDERING_GPU="${U1_RENDERING_GPU:-7}"
export U1_TRAIN_BATCH_SIZE="${U1_TRAIN_BATCH_SIZE:-7}"
export U1_PPO_MINI_BATCH_SIZE="${U1_PPO_MINI_BATCH_SIZE:-7}"
export U1_MINI_BATCH_SIZE="${U1_MINI_BATCH_SIZE:-7}"
export U1_GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.35}"
bash examples/train/active_spatial/run_experiment.sh \
  examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke.sh
