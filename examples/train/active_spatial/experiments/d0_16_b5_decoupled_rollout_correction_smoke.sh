# =============================================================================
# D0.16: decoupled rollout-correction closure.
# Diagnostic only:
#   - rollout.calculate_log_probs=True
#   - rollout.logprob_temperature=1.0
#   - algorithm.rollout_correction.bypass_mode=False
#   - rollout_is=token, threshold=2.0
#   - actor.use_rollout_log_probs=True to preserve trainer-provided old_log_probs
#   - one bounded step, no validation, no checkpoint save
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/b5_c8_wrapper_img25_actionvalid.sh"

EXPERIMENT_NAME="d0_16_b5_decoupled_rollout_correction_smoke"
export VAGEN_VISION_SANITY_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/vision_sanity"
export VAGEN_VISION_SANITY_MAX=4
export VAGEN_D0_3_SMOKE_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/d0_16_smoke"
export VAGEN_D0_13_TIMELINE_DIR="${VAGEN_D0_13_TIMELINE_DIR:-${VAGEN_D0_3_SMOKE_DIR}/d0_13_timeline}"

RENDER_MODE="${D0_16_RENDER_MODE:-${D0_12_RENDER_MODE:-${RENDER_MODE}}}"
RENDER_HOST="${D0_16_RENDER_HOST:-${D0_12_RENDER_HOST:-${RENDER_HOST}}}"
RENDER_PORT="${D0_16_RENDER_PORT:-${D0_12_RENDER_PORT:-${RENDER_PORT}}}"
RENDER_PROTOCOL="${D0_16_RENDER_PROTOCOL:-${D0_12_RENDER_PROTOCOL:-${RENDER_PROTOCOL}}}"
RENDERING_GPU="${D0_16_RENDERING_GPU:-${D0_12_RENDERING_GPU:-${RENDERING_GPU}}}"
RESUME_MODE="disable"
TOTAL_STEPS=1
VAL_BEFORE_TRAIN="False"
TEST_FREQ=-1
SAVE_FREQ=-1
USE_GPU_HOLDER=false
NUM_TRAIN_GPUS="${D0_16_NUM_TRAIN_GPUS:-4}"
TRAIN_BATCH_SIZE="${D0_16_TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE="${D0_16_PPO_MINI_BATCH_SIZE:-4}"
MINI_BATCH_SIZE="${D0_16_MINI_BATCH_SIZE:-4}"
CRITIC_WARMUP=0
VAL_N=1
MAX_TURNS=1
MAX_RESPONSE_LENGTH=128
MAX_TRAJECTORY_LENGTH=4096
GPU_MEM_UTIL="${D0_16_GPU_MEM_UTIL:-0.20}"
TP_SIZE="${D0_16_TP_SIZE:-2}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
    trainer.logger=['console'] \
    data.shuffle=False \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.use_policy_gradient=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprob_temperature=1.0 \
    ++actor_rollout_ref.actor.use_rollout_log_probs=True \
    +trainer.d0_16_smoke_dir=${VAGEN_D0_3_SMOKE_DIR} \
    ${D0_16_EXTRA_OVERRIDES:-}"
