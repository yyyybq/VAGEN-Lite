# =============================================================================
# Canonical clean Qwen Active Spatial config v1.
#
# Base recipe: v46_baseline_qwen25vl_7b.
# Intended differences from v46:
#   - preserve behavior-policy rollout logprobs
#   - use HF/FSDP recompute as PPO proximal anchor
#   - apply detached token-level rollout IS correction
# No Cambrian-specific override is included here.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/v46_baseline_qwen25vl_7b.sh"

EXPERIMENT_NAME="${QWEN_CLEAN_EXPERIMENT_NAME:-qwen_v46_clean_d0pass_v1}"
RESUME_MODE="disable"
RENDER_MODE="${ACTIVE_SPATIAL_CLEAN_RENDER_MODE:-${QWEN_CLEAN_RENDER_MODE:-$RENDER_MODE}}"
RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-${QWEN_CLEAN_RENDER_HOST:-$RENDER_HOST}}"
RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-${QWEN_CLEAN_RENDER_PORT:-$RENDER_PORT}}"
RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-${QWEN_CLEAN_RENDER_PROTOCOL:-$RENDER_PROTOCOL}}"

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
  ${QWEN_CLEAN_EXTRA_OVERRIDES:-}"
