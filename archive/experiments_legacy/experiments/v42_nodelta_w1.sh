# =============================================================================
# v42_nodelta_w1
# =============================================================================
# 科学问题（承接 v40/v41 分析）：
#   delta_control 是否通过训练数据"污染"影响了其他任务的 policy 质量？
#
# 设计思路：
#   - 从训练和 ID val 中完全去掉 delta_control
#   - WINDOW_SIZE=1（与 v39 一致，基准配置）
#   - 其余超参与 v39/v40 完全一致（masked_gae, KL=0.30, LR=5e-7 fast cosine）
#
# 预期：
#   - 如果非 delta ID_m4 超过 v39 峰值 0.5714，说明 delta 在训练中确实是噪声
#   - 如果与 v39 持平，说明 delta 对其他任务无负迁移
#   - OOD 曲线作为泛化能力参照
#
# 与 v39 的差异：
#   ★ TRAIN_EXCLUDE_TASK_TYPES=delta_control  (训练数据过滤)
#   ★ ID_VAL_EXCLUDE_TASK_TYPES=delta_control (ID val 排除 delta，n=18)
#   ★ WINDOW_SIZE: 1 (与 v39 相同，不引入 W=3 变量)
#   - TOTAL_STEPS: 700（与 v40 一致，给充分收敛窗口）
#
# 与 v42_nodelta_w3 的差异：
#   ★ WINDOW_SIZE=1 vs W=3
#
# 启动:
# cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
#   nohup bash examples/train/active_spatial/run_experiment.sh \
#     examples/train/active_spatial/experiments/v42_nodelta_w1.sh \
#     > v42_nodelta_w1.log 2>&1 &
# echo "PID: $!"

EXPERIMENT_NAME="v42_nodelta_w1"
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
WINDOW_SIZE="1"
MAX_TRAJECTORY_LENGTH="18000"
MAX_RESPONSE_LENGTH="384"
MAX_PROMPT_LENGTH="2048"

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
