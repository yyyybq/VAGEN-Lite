# =============================================================================
# v19_dual — score 系统消融: progress_mode potential -> dual
# =============================================================================
# yaml 改动 (相对 v19_thr70_andgate, 3 处同一 hypothesis):
#   potential_field_progress_mode : "potential" -> "dual"
#   position_reward_scale         : (new) 0.3
#   orientation_reward_scale      : (new) 0.7
# 其余 (thr=0.70 AND-gate / auto_term ON / near_success constant / success_reward=50
#       / format=0.05 / step_penalty=-0.005) 全部沿用 v19_thr70_andgate.
#
# 目的: 在 v19 stack 下验证 "per-axis credit 能否激活 ori 通道学习".
# 关键观测指标 (在 wandb 上):
#   - env_metrics/turn_left, turn_right 使用率 > 0
#   - env_metrics/ori_score mean 抬升
#   - env_turns/mean 不塌 (>= 4)
#   - actor/entropy 不爆 (< 4)
#
# sh 部分全沿用 v19_thr70_andgate.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v19_dual.sh \
    > v19_dual.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v19_dual"
ENV_CONFIG="env_config_v19_dual.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling ===
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30
CLIPRANGE_VALUE="0.8"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data ===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
