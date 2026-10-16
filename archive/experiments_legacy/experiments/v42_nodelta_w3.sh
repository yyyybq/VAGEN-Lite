# =============================================================================
# v42_nodelta_w3
# =============================================================================
# 科学问题（与 v42_nodelta_w1 配对消融）：
#   在去除 delta_control 训练污染的前提下，WINDOW_SIZE=3 是否比 W=1 带来更高
#   的非 delta 任务成功率？
#
# 设计思路：
#   - 训练/val 均去掉 delta_control
#   - WINDOW_SIZE=3（与 v40 一致）
#   - 通过与 v42_nodelta_w1 对比，单独评估 W=3 的增益
#
# 与 v42_nodelta_w1 的差异：
#   ★ WINDOW_SIZE=3（vs W=1）
#   ★ MAX_PROMPT_LENGTH=3072（vs 2048，适配更大的 window context）
#
# 预期：
#   - 若 v42_nodelta_w3 > v42_nodelta_w1：W=3 对非 delta 任务确实有帮助
#   - 若 ≈ 或 <：W=3 的额外 context 噪声抵消了记忆增益（与 v40 vs v39 一致）
#
# 启动:
#: <<'RUN'
# cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
#   nohup bash examples/train/active_spatial/run_experiment.sh \
#     examples/train/active_spatial/experiments/v42_nodelta_w3.sh \
#     > v42_nodelta_w3.log 2>&1 &
# echo "PID: $!"
# RUN

EXPERIMENT_NAME="v42_nodelta_w3"
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
TOTAL_STEPS="700"
VAL_BEFORE_TRAIN="True"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.002"
LAM="0.95"

# === LR Scheduler (fast cosine) ===
export EXTRA_OVERRIDES="  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.05"

# === OOD val ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
export OOD_VAL_N_ENVS="25"

# === Training data filter: exclude delta_control ===
# Removes ~2003/11157 (~18%) of training samples.
export TRAIN_EXCLUDE_TASK_TYPES="delta_control"

# === ID val composition: exclude delta_control (n_id=18) ===
export ID_VAL_EXCLUDE_TASK_TYPES="delta_control"
