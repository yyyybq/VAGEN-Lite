# =============================================================================
# v18_dual (S1 真双通道 + E1 + E4 + E5)
# =============================================================================
# 在测什么:
#   S1 (真实现, 不是反转权重 hack): env.py 新增 progress_mode="dual",
#       r_t = α_pos · Δpos_score + α_ori · Δori_score
#   两个通道各自产生 advantage 信号; 模型只有真正旋转才能涨 Δori。
#   血洗 v17 现象: rollout step 210 显示 turn_left=turn_right=0,
#   ori_score 永远学不会, val_success=0.
#
#   α_pos=0.3, α_ori=0.7 是补偿 ori_score 学习信号弱 (起点低、增长慢),
#   不是反转重要性。pos_score 可平移获得, ori_score 必须旋转才能获得。
#
#   total_score 仍按 0.7/0.3 加权用于 success_score_threshold 门控
#   (auto-success 仍然需要"位置 + 朝向都达标"), 但 shaping reward 走
#   双通道独立路径, 二者解耦。
#
#   E1: KL=0.05, ent=0.001  E4: step 0.30/20°  E5: prompt 已 fix
#
# 启动命令:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_dual.sh \
    > v18_dual.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_dual"
ENV_CONFIG="env_config_v18_dual.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor: E1 ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

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
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
