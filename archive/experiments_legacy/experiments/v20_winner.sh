# =============================================================================
# v20_winner — Triple winner merger × 后期崩塌补丁
# =============================================================================
# yaml = v19_no_farm_dual.yaml (no_farm + dual + thr70_andgate, 三路合并)
# sh 改动 (相对 v19_thr70_andgate / v19_no_farm_dual, 用于修 v19_thr70_andgate 的
#          val best@4 早期 0.859 → 末期 0.000 崩塌轨迹):
#
#   ★ KL_LOSS_COEF:      0.05 -> 0.10     # 拉紧 KL, 防策略漂得太远
#   ★ CLIPRANGE_VALUE:   0.8  -> 0.5      # critic 更稳, 不让 V 估计抖
#   ★ CRITIC_WARMUP:     30   -> 60       # 给 critic 多 30 步纯监督, 减少早期 advantage 噪声
#   ★ ENTROPY_COEFF:     0.001 -> 0.002   # 稍提探索, 抵消 KL_LOSS_COEF 拉紧带来的过度收敛
#
# 假设:
#   - v19_thr70_andgate 后期崩塌主因是 critic 失稳 / 策略 KL 泄漏 / format-farm.
#     yaml 已通过 no_farm 修掉 format-farm; sh 这里再修后两点.
#   - 期望: val best@4 持续 ≥ 0.85 至少 100+ 步, 不出现 0.7 → 0 的塌方.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v20_winner.sh \
    > v20_winner.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v20_winner"
ENV_CONFIG="env_config_v19_no_farm_dual.yaml"   # 复用 v19_no_farm_dual 的 yaml
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.002"          # ★ 0.001 -> 0.002
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"            # ★ 0.05 -> 0.10
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
CRITIC_WARMUP=60               # ★ 30 -> 60
CLIPRANGE_VALUE="0.5"          # ★ 0.8 -> 0.5

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
