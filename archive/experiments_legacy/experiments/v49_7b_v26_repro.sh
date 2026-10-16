# =============================================================================
# v49_7b_v26_repro
# =============================================================================
# Purpose: reproduce the historical Qwen2.5-VL-3B v26_klhi_lr5e7 best run on
# the new H800 setup, changing only the model/infrastructure scale:
#   - Qwen2.5-VL-7B-Instruct instead of 3B
#   - 8 visible H800 training GPUs on a remote node
#   - jump host GS render server at 10.119.18.163:8777
#
# Historical reference: v26_klhi_lr5e7 reached ID_m4=0.607 @ step150.
# This script keeps v26's constant LR=5e-7, KL=0.20, W=1, old 7types data,
# and old reward scale to test whether the new server/model can reproduce the
# same learning curve rather than the later no-delta/v42 recipe.
# =============================================================================

EXPERIMENT_NAME="v49_7b_v26_repro"
ENV_CONFIG="env_config_h800_7b_v26_100scenes.yaml"
NUM_TRAIN_GPUS=8
RENDER_MODE="remote"
RENDER_HOST="10.119.18.163"
RENDER_PORT="8777"
RENDERING_GPU=0

RESUME_MODE="auto"

# === Model ===
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# === Actor: match v26 ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE="4"
GPU_MEM_UTIL="0.30"

# === Validation sampling ===
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic: match v26 ===
CRITIC_LR="2e-5"
CRITIC_WARMUP="60"
CLIPRANGE_VALUE="0.5"

# === Optim: match v26 ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"

# === Trajectory: match v26, with 7B-safe token budget ===
MAX_TURNS="12"
WINDOW_SIZE="1"
MAX_TRAJECTORY_LENGTH="8192"
MAX_RESPONSE_LENGTH="384"
MAX_PROMPT_LENGTH="4096"

# === Data / GroupAdv ===
N_TRAJECTORY="4"
TRAIN_BATCH_SIZE="12"
VAL_BATCH_SIZE="8"
PPO_MINI_BATCH_SIZE="8"
MINI_BATCH_SIZE="8"

# === Trainer ===
SAVE_FREQ="50"
TEST_FREQ="50"
TOTAL_STEPS="700"
VAL_BEFORE_TRAIN="True"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === OOD validation: old v26 OOD split, shared path ===
export OOD_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS="19"
