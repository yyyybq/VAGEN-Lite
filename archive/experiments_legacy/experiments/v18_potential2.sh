# =============================================================================
# v18_potential2 — v18_potential 续作 (3 大修复 + threshold 0.30 -> 0.40)
# =============================================================================
# 修复:
#   F1 (env.py)   info["success"]/["is_success"] 解锁 val_traj_success metric
#   F2 (run_*.sh) val_kwargs 注入, 打破 val greedy 确定性
#   F3 (yaml)     success_score_threshold 0.30 -> 0.40, 防止 pos 单通道过早赢
#
# 其余 actor / critic / optim / Ng1999 γ-potential shaping 全部沿用
# v18_potential, 便于对照.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_potential2.sh \
    > v18_potential2.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_potential2"
ENV_CONFIG="env_config_v18_potential2.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (沿用 v18_potential) ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling (F2 修复) ==========================================
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic (沿用 v18_potential) ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30
CLIPRANGE_VALUE="0.8"

# === Optim (沿用 v18_potential) ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === 轨迹 (沿用 v18_potential) ===
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
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
