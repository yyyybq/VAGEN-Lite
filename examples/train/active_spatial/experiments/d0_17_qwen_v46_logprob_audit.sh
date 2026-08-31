# =============================================================================
# D0.17: Qwen v46 real-batch rollout/training-policy audit.
#
# Bounded diagnostic:
#   - starts from historical v46_7b_nodelta_w3
#   - records vLLM rollout logprobs without changing sampling/reward/LR/KL
#   - uses framework-native decoupled correction for the same-batch A/B/C audit
#   - default STOP_AFTER_AUDIT=1, so no optimizer step unless explicitly disabled
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/v46_7b_nodelta_w3.sh"

EXPERIMENT_NAME="${D0_17_QWEN_AUDIT_NAME:-d0_17_qwen_v46_logprob_audit}"
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

NUM_TRAIN_GPUS="${D0_17_NUM_TRAIN_GPUS:-2}"
TP_SIZE="${D0_17_TP_SIZE:-2}"
GPU_MEM_UTIL="${D0_17_GPU_MEM_UTIL:-0.22}"
TRAIN_BATCH_SIZE="${D0_17_TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="1"
PPO_MINI_BATCH_SIZE="${D0_17_PPO_MINI_BATCH_SIZE:-4}"
MINI_BATCH_SIZE="${D0_17_MINI_BATCH_SIZE:-4}"
N_TRAJECTORY="${D0_17_N_TRAJECTORY:-1}"
MAX_TURNS="${D0_17_MAX_TURNS:-1}"
MAX_RESPONSE_LENGTH="${D0_17_MAX_RESPONSE_LENGTH:-128}"
MAX_TRAJECTORY_LENGTH="${D0_17_MAX_TRAJECTORY_LENGTH:-4096}"
MAX_PROMPT_LENGTH="${D0_17_MAX_PROMPT_LENGTH:-2048}"
VAL_N="1"

export VAGEN_D0_16_DIR="${VAGEN_D0_16_DIR:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/d0_17_qwen_audit}"
export VAGEN_D0_16_STOP_AFTER_AUDIT="${VAGEN_D0_16_STOP_AFTER_AUDIT:-1}"
export VAGEN_D0_13_TIMELINE_DIR="${VAGEN_D0_13_TIMELINE_DIR:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/timeline}"

EXTRA_OVERRIDES="${EXTRA_OVERRIDES} \
  trainer.logger=['console'] \
  data.shuffle=False \
  hydra.run.dir=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/hydra_run \
  +ray_kwargs.ray_init.include_dashboard=False \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.logprob_temperature=1.0 \
  ++actor_rollout_ref.actor.use_rollout_log_probs=True \
  algorithm.rollout_correction.bypass_mode=False \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.rollout_correction.use_policy_gradient=False \
  ${D0_17_EXTRA_OVERRIDES:-}"
