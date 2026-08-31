# =============================================================================
# Canonical clean Cambrian-S Active Spatial config v1.
#
# Base recipe: B5/C8 wrapper action-valid recipe.
# Intended differences from B5:
#   - keep Cambrian plumbing fixes already landed in code
#   - replace bypass old-logprob mode with framework-native decoupled correction
#   - preserve behavior-policy rollout logprobs at raw model T=1
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/b5_c8_wrapper_img25_actionvalid.sh"

EXPERIMENT_NAME="${CAMBRIAN_CLEAN_EXPERIMENT_NAME:-cambrian_c8_clean_d0pass_v1}"
RESUME_MODE="disable"
RENDER_MODE="${ACTIVE_SPATIAL_CLEAN_RENDER_MODE:-${CAMBRIAN_CLEAN_RENDER_MODE:-$RENDER_MODE}}"
RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-${CAMBRIAN_CLEAN_RENDER_HOST:-$RENDER_HOST}}"
RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-${CAMBRIAN_CLEAN_RENDER_PORT:-$RENDER_PORT}}"
RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-${CAMBRIAN_CLEAN_RENDER_PROTOCOL:-$RENDER_PROTOCOL}}"

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
  ${CAMBRIAN_CLEAN_EXTRA_OVERRIDES:-}"
