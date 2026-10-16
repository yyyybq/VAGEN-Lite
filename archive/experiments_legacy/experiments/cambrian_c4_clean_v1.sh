#!/usr/bin/env bash
# D0-correct replay of historical C4: forward-first prompt with the original
# high-variance reward. This is the causal control for the C8/B5 reward recipe.
source "$(dirname "${BASH_SOURCE[0]}")/c4_fwdfirst.sh"

EXPERIMENT_NAME="${CAMBRIAN_C4_CLEAN_EXPERIMENT_NAME:-cambrian_c4_clean_d0pass_v1}"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst_mnt.yaml"
RESUME_MODE="disable"
NUM_TRAIN_GPUS=8
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8
MODEL_PATH="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP"
RENDER_MODE="${ACTIVE_SPATIAL_CLEAN_RENDER_MODE:-remote}"
RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"
export OOD_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
  hydra.run.dir=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/hydra_run \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.logprob_temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  ++actor_rollout_ref.actor.use_rollout_log_probs=True \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.use_policy_gradient=False \
  trainer.max_actor_ckpt_to_keep=20 \
  trainer.max_critic_ckpt_to_keep=20"
