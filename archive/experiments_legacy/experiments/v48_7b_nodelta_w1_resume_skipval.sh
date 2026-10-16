# =============================================================================
# v48_7b_nodelta_w1
# =============================================================================
# 消融实验：在 7B 模型上对比 window_size=1 vs window_size=3 (v46)
#
# 科学问题：Window Size 效应是否在 7B 模型上同样成立？
#   - v42_nodelta_w3 > v42_nodelta_w1 (3B, 证明 w3 更好)
#   - v46 = 7B + w3 (KL=0.30)
#   - v48 = 7B + w1 (KL=0.30)  ← 本实验
#
# 设计：
#   - 模型: Qwen2.5-VL-7B-Instruct（与 v46 相同）
#   - 超参与 v46 完全一致，仅 WINDOW_SIZE 从 3 改为 1
#   - GPU: Node 10.119.21.237, GPUs 3,4,5,6 (训练) + GPU 7 (渲染)
#
# 预期：
#   - w1 通常比 w3 更不稳定（信用分配视野小）
#   - 如果 7B 版 w1 也弱于 w3，印证 window_size 效应的模型无关性

EXPERIMENT_NAME="v48_7b_nodelta_w1"
ENV_CONFIG="env_config_h800_7b_6types.yaml"
NUM_TRAIN_GPUS=8
RENDER_MODE="remote"
RENDER_HOST="10.119.18.163"
RENDER_PORT="8777"
RENDERING_GPU=0

RESUME_MODE="auto"

# === Model ===
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.30"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE="4"
# Colocated FSDP+vLLM on H800: keep headroom for wake_up weight sync.
GPU_MEM_UTIL="0.30"

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
WINDOW_SIZE="1"    # Key difference: w1 vs v46's w3
MAX_TRAJECTORY_LENGTH="8192"
MAX_RESPONSE_LENGTH="384"
MAX_PROMPT_LENGTH="4096"

# === Batch ===
N_TRAJECTORY="4"
TRAIN_BATCH_SIZE="12"
VAL_BATCH_SIZE="8"
PPO_MINI_BATCH_SIZE="8"
MINI_BATCH_SIZE="8"

# === Schedule ===
SAVE_FREQ="50"
TEST_FREQ="50"
TOTAL_STEPS="700"
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.002"
LAM="0.95"

# === LR Scheduler (fast cosine, same as v46) ===
export EXTRA_OVERRIDES="  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.05"

# === OOD val ===
export OOD_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
export OOD_VAL_N_ENVS="25"

# === Exclude delta_control ===
export TRAIN_EXCLUDE_TASK_TYPES="delta_control"
export ID_VAL_EXCLUDE_TASK_TYPES="delta_control"
