# =============================================================================
# v18_dual2 — v18_dual 修复版 (cliff 削平 + thr 提高 + AND-gate)
# =============================================================================
# yaml 改动 (相对 v18_dual):
#   position_reward_scale       : 0.3  -> 0.4
#   orientation_reward_scale    : 0.7  -> 0.6
#   success_score_threshold     : 0.30 -> 0.45   (AND-gate ON 时备份)
#   success_reward              : 50.0 -> 30.0   (削 cliff)
#   success_require_both        : false -> true
#   success_position_threshold  : 0.45
#   success_orientation_threshold: 0.45
#
# Actor / Critic / Optim 沿用 v18_dual / v18_potential2 共用模板.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_dual2.sh \
    > v18_dual2.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_dual2"
ENV_CONFIG="env_config_v18_dual2.yaml"
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

# === Validation sampling (F2) ===
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
