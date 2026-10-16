# =============================================================================
# D0.12: behavior-policy logprob plumbing smoke.
# Diagnostic only: one PPO update, no validation benchmark, no checkpoint save.
# Inherits the B5/C8 repair:
#   - rollout.calculate_log_probs=True
#   - rollout.logprob_temperature=1.0
#   - algorithm.rollout_correction.bypass_mode=True
#   - actor.use_rollout_log_probs=True
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/b5_c8_wrapper_img25_actionvalid.sh"

EXPERIMENT_NAME="d0_12_b5_logprob_plumbing_smoke"
export VAGEN_VISION_SANITY_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/vision_sanity"
export VAGEN_VISION_SANITY_MAX=8
export VAGEN_D0_3_SMOKE_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/d0_12_smoke"
export VAGEN_D0_13_TIMELINE_DIR="${VAGEN_D0_13_TIMELINE_DIR:-${VAGEN_D0_3_SMOKE_DIR}/d0_13_timeline}"

RENDER_MODE="${D0_12_RENDER_MODE:-${RENDER_MODE}}"
RENDER_HOST="${D0_12_RENDER_HOST:-${RENDER_HOST}}"
RENDER_PORT="${D0_12_RENDER_PORT:-${RENDER_PORT}}"
RENDER_PROTOCOL="${D0_12_RENDER_PROTOCOL:-${RENDER_PROTOCOL}}"
RENDERING_GPU="${D0_12_RENDERING_GPU:-${RENDERING_GPU}}"
RESUME_MODE="disable"
TOTAL_STEPS=1
VAL_BEFORE_TRAIN="False"
TEST_FREQ=-1
SAVE_FREQ=-1
USE_GPU_HOLDER=false
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8
CRITIC_WARMUP=0
VAL_N=1
MAX_TURNS=1
MAX_RESPONSE_LENGTH=128
MAX_TRAJECTORY_LENGTH=4096

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
    trainer.logger=['console'] \
    data.shuffle=False \
    +trainer.d0_12_smoke_dir=${VAGEN_D0_3_SMOKE_DIR}"
