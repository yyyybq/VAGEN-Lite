# =============================================================================
# v50_7b_w3_format001_stable
# =============================================================================
# V50: stable no-delta W3 run with tiny format reward and shorter responses
#
# 科学问题：7B 模型在相同 RL 超参下是否比 3B 有更好的稳定性和性能上限？
#
# 设计：
#   - 模型: Qwen2.5-VL-7B-Instruct（vs 3B in v42）
#   - 超参与 v42_nodelta_w3 完全对应（直接对照实验）
#   - 7B on H800: 4 training GPUs (TP=4) + 1 rendering GPU (5 total)
#   - GPU_MEM_UTIL 0.4 → 0.55（H800 80GB 内存充足）
#   - 排除 delta_control（沿用 v42 的结论）
#
# 预期：
#   - 7B 语义理解更强 → 更高 ID_m4
#   - RL 稳定性可能更强（大模型 KL 约束更有效）
#   - 如果仍然发生 collapse，将在 v47/v48 测试 higher KL

EXPERIMENT_NAME="v50_7b_w3_format001_stable"
ENV_CONFIG="env_config_h800_7b_6types_v50_format001.yaml"
NUM_TRAIN_GPUS=8
RENDER_MODE="remote"
RENDER_HOST="10.119.18.163"
RENDER_PORT="8767"
RENDER_PROTOCOL="http"
RENDERING_GPU=0

RESUME_MODE="auto"

# === Model ===
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# === Actor: reduce token-level exploration to avoid format drift ===
ENTROPY_COEFF="0.001"
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

# === Episode: restore W3 context and shorten response budget ===
MAX_TURNS="12"
WINDOW_SIZE="3"
MAX_TRAJECTORY_LENGTH="8192"
MAX_RESPONSE_LENGTH="160"
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
VAL_BEFORE_TRAIN="True"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.002"
LAM="0.95"

# === LR Scheduler (fast cosine, same as v42) ===
export EXTRA_OVERRIDES="  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.05"

# === OOD val ===
export OOD_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
export OOD_VAL_N_ENVS="25"

# === Exclude delta_control ===
export TRAIN_EXCLUDE_TASK_TYPES="delta_control"
export ID_VAL_EXCLUDE_TASK_TYPES="delta_control"
