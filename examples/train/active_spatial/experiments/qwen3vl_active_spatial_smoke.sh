#!/usr/bin/env bash
# Qwen3-VL Active Spatial migration smoke (one PPO update).
# Requires a Transformers/vLLM/SGLang build with Qwen3-VL support and a local
# or mirrored Qwen/Qwen3-VL-8B-Instruct checkpoint.

EXPERIMENT_NAME="${QWEN3_VL_RUN_NAME:-qwen3vl_active_spatial_smoke}"
ENV_CONFIG="env_config_h800_7b_6types_v50_format001.yaml"
MODEL_PATH="${QWEN3_VL_MODEL:-/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b}"
NUM_TRAIN_GPUS="8"
RENDER_MODE="remote"
RENDER_HOST="${QWEN3_VL_RENDER_HOST:-10.119.27.237}"
RENDER_PORT="${QWEN3_VL_RENDER_PORT:-8768}"
RENDER_PROTOCOL="http"
TP_SIZE="4"
GPU_MEM_UTIL="0.30"
USE_GPU_HOLDER="false"
TOTAL_STEPS="1"
SAVE_FREQ="100"
TEST_FREQ="1"
VAL_BEFORE_TRAIN="False"
MAX_RESPONSE_LENGTH="160"
MAX_TRAJECTORY_LENGTH="8192"
TRAIN_BATCH_SIZE="8"
VAL_BATCH_SIZE="1"
N_TRAJECTORY="1"
PPO_MINI_BATCH_SIZE="8"
MINI_BATCH_SIZE="1"
ENTROPY_COEFF="0.001"
ACTOR_LR="5e-7"
CRITIC_LR="2e-5"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.30"
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
LAM="0.95"

# Qwen3-VL currently requires recent attention kernels; TORCH_SDPA is the
# portable fallback on glibc-2.31 nodes and avoids flash-attn ABI failures.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export TRANSFORMERS_ATTN_IMPLEMENTATION="${TRANSFORMERS_ATTN_IMPLEMENTATION:-eager}"
