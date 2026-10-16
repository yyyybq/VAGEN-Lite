#!/usr/bin/env bash
# D0-correct replay of historical v50: W3, short response, low entropy,
# and the format-0.01 environment.
source "$(dirname "${BASH_SOURCE[0]}")/v50_7b_w3_format001_stable.sh"

EXPERIMENT_NAME="${QWEN_V50_CLEAN_EXPERIMENT_NAME:-qwen_v50_clean_d0pass_v1}"
RESUME_MODE="disable"
RENDER_MODE="${ACTIVE_SPATIAL_CLEAN_RENDER_MODE:-remote}"
RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
  hydra.run.dir=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/hydra_run \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.logprob_temperature=1.0 \
  ++actor_rollout_ref.actor.use_rollout_log_probs=True \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.use_policy_gradient=False \
  trainer.max_actor_ckpt_to_keep=20 \
  trainer.max_critic_ckpt_to_keep=20"
