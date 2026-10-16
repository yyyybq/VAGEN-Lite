# =============================================================================
# v18_sigmoid (S2 真 sigmoid + E1 + E4 + E5)
# =============================================================================
# 在测什么:
#   S2 (真实现, 不是 config 近似): env.py 新增 near_success_mode="sigmoid"，
#   per-step bonus 由"score >= thr 时常数 +0.2"改为
#         bonus = near_success_bonus · σ(k · (score − near_success_threshold))
#   彻底消除 v17 reward landscape 的两个悬崖 (success 阶跃 + near_success
#   阶跃)。配合 success_reward 50→20 降低终止时跃迁峰值。
#
#   Q: 还需要 near_success_bonus 字段吗? → 需要。它从"开关式常数 +0.2"
#       变成"sigmoid 的饱和峰值 α"。该字段被泛化，没废弃。
#
#   E1: KL_LOSS_COEF 0.02 -> 0.05, ENTROPY_COEFF 0.005 -> 0.001
#       (抑制 v17 后期 entropy 0.41 -> 9.92 的爆炸)
#   E4: step_translation 0.20 -> 0.30, step_rotation_deg 15 -> 20 (yaml 内)
#   E5: prompt.py 已永久加 "先转向后平移" 提示
#
# 启动命令:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_sigmoid.sh \
    > v18_sigmoid.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_sigmoid"
ENV_CONFIG="env_config_v18_sigmoid.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor: E1 ===
ENTROPY_COEFF="0.001"           # ★ v17 0.005 -> 0.001
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"             # ★ v17 0.02 -> 0.05
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

# === 轨迹 (沿用 v17) ===
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
