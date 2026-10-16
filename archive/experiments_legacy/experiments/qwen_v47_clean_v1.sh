#!/usr/bin/env bash
# D0-correct replay of historical v47: v46 W3 with KL loss 0.40.
source "$(dirname "${BASH_SOURCE[0]}")/v47_7b_nodelta_w3_kl40.sh"

EXPERIMENT_NAME="${QWEN_V47_CLEAN_EXPERIMENT_NAME:-qwen_v47_clean_d0pass_v1}"
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
  algorithm.rollout_correction.use_policy_gradient=False"
