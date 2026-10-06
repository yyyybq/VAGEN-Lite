#!/usr/bin/env bash
# Capture exactly one no-concat PPO batch.  The hook returns before all updates.
EXPERIMENT_NAME="active_spatial_dense_score_diagnostic_real_batch_20260916_r5"
ENV_CONFIG="env_config_diagnostic_r1_real_batch.yaml"
# Four FSDP ranks are the smallest historical 7B-compatible topology and make
# the frozen 8 sources × rollout.n=4 = 32 rows exactly divisible.  GPU 7 stays
# exclusively with the official renderer; GPUs 4..6 are deliberately unused.
NUM_TRAIN_GPUS=4
RENDER_MODE=remote
RENDER_HOST=127.0.0.1
RENDER_PORT=8898
RENDER_PROTOCOL=http
MODEL_PATH="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898"
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=1
MAX_TURNS=12
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=384
MAX_TRAJECTORY_LENGTH=8192
TEMPERATURE=0.8
TOP_P=0.95
VAL_BEFORE_TRAIN=False
TOTAL_STEPS=1
SAVE_FREQ=999999
TEST_FREQ=999999
RESUME_MODE=disable
ADV_ESTIMATOR=no_concat_gae
HIGH_LEVEL_GAMMA=0.95
LAM=0.95
ACTOR_LR=5e-7
CRITIC_LR=2e-5
CRITIC_WARMUP=0
CLIPRANGE_VALUE=0.5
USE_KL_LOSS=True
KL_LOSS_COEF=0.30
ENTROPY_COEFF=0.001
GRAD_CLIP=0.3
PPO_MINI_BATCH_SIZE=8
TP_SIZE=1
GPU_MEM_UTIL=0.30
# no-concat trajectories have variable turn counts, so the existing PPO path
# must retain DP padding/balancing before per-rank old-logprob computation.
export EXTRA_OVERRIDES="trainer.balance_batch=True filter.enable=False algorithm.use_kl_in_reward=False actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.logprob_temperature=1.0 ++actor_rollout_ref.actor.use_rollout_log_probs=True"
