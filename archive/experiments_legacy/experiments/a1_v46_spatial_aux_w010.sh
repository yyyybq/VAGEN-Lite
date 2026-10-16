# =============================================================================
# A1: v46 + spatial MCQ aux (ratio=0.10, coef=0.10)
# Unique variable vs a1_w005: SPATIAL_AUX_RATIO=0.10
# =============================================================================
EXPERIMENT_NAME="a1_v46_spatial_aux_w010"
ENV_CONFIG="env_config_h800_7b_6types.yaml"
export AUX_ENV_CONFIG="env_config_a1_spatial_mcq.yaml"
export SPATIAL_AUX_RATIO="0.10"
export SPATIAL_AUX_COEF="1.0"

NUM_TRAIN_GPUS=8
RENDER_MODE="remote"
RENDER_HOST="10.119.30.223"
RENDER_PORT="8767"
RENDER_PROTOCOL="http"
RENDERING_GPU=0
RESUME_MODE="auto"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.30"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE="4"
GPU_MEM_UTIL="0.30"
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"
CRITIC_LR="2e-5"
CRITIC_WARMUP="60"
CLIPRANGE_VALUE="0.5"
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"
MAX_TURNS="12"
WINDOW_SIZE="3"
MAX_TRAJECTORY_LENGTH="8192"
MAX_RESPONSE_LENGTH="384"
MAX_PROMPT_LENGTH="4096"
N_TRAJECTORY="4"
TRAIN_BATCH_SIZE="12"
VAL_BATCH_SIZE="8"
PPO_MINI_BATCH_SIZE="8"
MINI_BATCH_SIZE="8"
SAVE_FREQ="50"
TEST_FREQ="50"
TOTAL_STEPS="700"
VAL_BEFORE_TRAIN="True"
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.002"
LAM="0.95"
export EXTRA_OVERRIDES="  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.05 \
  hydra.run.dir=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/hydra_run \
  +ray_kwargs.ray_init.include_dashboard=False"
export OOD_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
export OOD_VAL_N_ENVS="25"
export TRAIN_EXCLUDE_TASK_TYPES="delta_control"
export ID_VAL_EXCLUDE_TASK_TYPES="delta_control"
