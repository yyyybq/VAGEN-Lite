# =============================================================================
# D0.17: Cambrian clean-v1 acceptance smoke.
# One real rollout/update with decoupled correction enabled.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/cambrian_active_spatial_clean_v1.sh"

EXPERIMENT_NAME="${D0_17_CAMBRIAN_SMOKE_NAME:-d0_17_cambrian_clean_acceptance_smoke}"
RENDER_MODE="${D0_17_RENDER_MODE:-remote}"
RENDER_HOST="${D0_17_RENDER_HOST:-127.0.0.1}"
RENDER_PORT="${D0_17_RENDER_PORT:-8767}"
RENDER_PROTOCOL="${D0_17_RENDER_PROTOCOL:-http}"
RESUME_MODE="disable"

TOTAL_STEPS="${D0_17_TOTAL_STEPS:-1}"
VAL_BEFORE_TRAIN="False"
TEST_FREQ="-1"
SAVE_FREQ="-1"
USE_GPU_HOLDER=false
CRITIC_WARMUP="${D0_17_CRITIC_WARMUP:-0}"

NUM_TRAIN_GPUS="${D0_17_NUM_TRAIN_GPUS:-8}"
TP_SIZE="${D0_17_TP_SIZE:-2}"
GPU_MEM_UTIL="${D0_17_GPU_MEM_UTIL:-0.20}"
TRAIN_BATCH_SIZE="${D0_17_TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="1"
PPO_MINI_BATCH_SIZE="${D0_17_PPO_MINI_BATCH_SIZE:-8}"
MINI_BATCH_SIZE="${D0_17_MINI_BATCH_SIZE:-8}"
N_TRAJECTORY="${D0_17_N_TRAJECTORY:-1}"
MAX_TURNS="${D0_17_MAX_TURNS:-1}"
MAX_RESPONSE_LENGTH="${D0_17_MAX_RESPONSE_LENGTH:-128}"
MAX_TRAJECTORY_LENGTH="${D0_17_MAX_TRAJECTORY_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${D0_17_MAX_PROMPT_LENGTH:-2048}"
VAL_N="1"

export VAGEN_D0_16_DIR="${VAGEN_D0_16_DIR:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/d0_17_cambrian_smoke}"
unset VAGEN_D0_16_STOP_AFTER_AUDIT
export VAGEN_D0_13_TIMELINE_DIR="${VAGEN_D0_13_TIMELINE_DIR:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/timeline}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
  trainer.logger=['console'] \
  data.shuffle=False \
  actor_rollout_ref.rollout.agent.num_workers=${NUM_TRAIN_GPUS} \
  hydra.run.dir=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/hydra_run \
  ${D0_17_EXTRA_OVERRIDES:-}"
