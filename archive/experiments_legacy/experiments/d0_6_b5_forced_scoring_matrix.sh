# =============================================================================
# D0.6: forced scoring / weight equality diagnosis. Bounded only: reuse the B5
# production Cambrian-S-LFP path, perform one rollout/logprob smoke, and skip the
# actor optimizer update. Do not use for D1 or full RL training.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/b5_c8_wrapper_img25_actionvalid.sh"

EXPERIMENT_NAME="d0_6_b5_forced_scoring_matrix"
export VAGEN_VISION_SANITY_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/vision_sanity"
export VAGEN_D0_4_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/d0_6"
export VAGEN_D0_6_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/d0_6"
export VAGEN_D0_6_DOCS_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/docs/diagnosis"
export VAGEN_D0_6_ABORT_ON_WEIGHT_MISMATCH=1
export VAGEN_D0_6_MODEL_CHECKPOINT="${MODEL_PATH}"
export VAGEN_D0_6_STOP_AFTER_LOGPROB=1

RENDER_MODE="${D0_6_RENDER_MODE:-${D0_5_RENDER_MODE:-${RENDER_MODE}}}"
RENDER_HOST="${D0_6_RENDER_HOST:-${D0_5_RENDER_HOST:-${RENDER_HOST}}}"
RENDER_PORT="${D0_6_RENDER_PORT:-${D0_5_RENDER_PORT:-${RENDER_PORT}}}"
RENDER_PROTOCOL="${D0_6_RENDER_PROTOCOL:-${D0_5_RENDER_PROTOCOL:-${RENDER_PROTOCOL}}}"
RENDERING_GPU="${D0_6_RENDERING_GPU:-${D0_5_RENDERING_GPU:-${RENDERING_GPU}}}"

TOTAL_STEPS=1
CRITIC_WARMUP=999
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8
MAX_TURNS=1
MAX_RESPONSE_LENGTH=128
MAX_TRAJECTORY_LENGTH=4096
RESUME_MODE="disable"
VAL_BEFORE_TRAIN="False"
TEST_FREQ=-1
SAVE_FREQ=-1
USE_GPU_HOLDER=false

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprob_temperature=1.0 \
    trainer.logger=['console'] \
    data.shuffle=False \
    +trainer.d0_6_dir=${VAGEN_D0_6_DIR}"
