# =============================================================================
# v44_nodelta_w3_fastcosine
# =============================================================================
# 基线: v42_nodelta_w3
# 单因素改动: 更快 cosine 衰减，TOTAL_STEPS 700 -> 500, min_lr_ratio 0.05 -> 0.01
# 目的: 在保留 v42 早中期学习速度的同时，让 step250 后 LR 更快降到低位。

EXPERIMENT_NAME="v44_nodelta_w3_fastcosine"
ENV_CONFIG="env_config_v39_6types_distinfo.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="auto"

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.30"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE="4"
GPU_MEM_UTIL="0.4"

VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP="60"
CLIPRANGE_VALUE="0.5"

# === Gradient ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"

# === Episode ===
MAX_TURNS="12"
WINDOW_SIZE="3"
MAX_TRAJECTORY_LENGTH="18000"
MAX_RESPONSE_LENGTH="384"
MAX_PROMPT_LENGTH="3072"

# === Batch ===
N_TRAJECTORY="4"
TRAIN_BATCH_SIZE="12"
VAL_BATCH_SIZE="8"
PPO_MINI_BATCH_SIZE="8"
MINI_BATCH_SIZE="8"

# === Schedule ===
SAVE_FREQ="50"
TEST_FREQ="50"
TOTAL_STEPS="500"
VAL_BEFORE_TRAIN="True"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.002"
LAM="0.95"

# === LR Scheduler (more aggressive cosine) ===
export EXTRA_OVERRIDES="  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.01"

# === OOD val ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
export OOD_VAL_N_ENVS="25"

# === Training data filter: exclude delta_control ===
export TRAIN_EXCLUDE_TASK_TYPES="delta_control"

# === ID val composition: exclude delta_control (n_id=18) ===
export ID_VAL_EXCLUDE_TASK_TYPES="delta_control"