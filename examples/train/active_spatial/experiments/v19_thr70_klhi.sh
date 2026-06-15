# =============================================================================
# v19_thr70_klhi — 抢救 v19_thr70_andgate 后期崩塌, sh-only 补丁
# =============================================================================
# 背景:
#   v19_thr70_andgate val best@4 早期 0.859 (step ~40), 之后一路掉到 0.000
#   (0.859 → 0.742 → 0.689 → 0.600 → 0.036 → 0.243 → 0.000).
#   train score 仍维持 +4.94, 但 val 完全反向 = 典型策略漂 / critic 失稳 /
#   format-farm 入侵.
#
# 本实验保持 yaml 不变 (env_config_v19_thr70_andgate.yaml), 仅在 sh 层加 RL 稳定补丁.
# 与 v20_winner 区别: v20_winner 同时叠加 dual + no_farm 的 yaml 合并;
#                     v19_thr70_klhi 是 yaml 不变的 ablation, 用来隔离
#                     "sh 级稳定补丁" 单独是否能修崩塌.
#
# 改动 (相对 v19_thr70_andgate, sh-only):
#   ★ KL_LOSS_COEF:      0.05 -> 0.10
#   ★ CLIPRANGE_VALUE:   0.8  -> 0.5
#   ★ CRITIC_WARMUP:     30   -> 60
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v19_thr70_klhi.sh \
    > v19_thr70_klhi.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v19_thr70_klhi"
ENV_CONFIG="env_config_v19_thr70_andgate.yaml"  # 复用 thr70_andgate yaml, 不变
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.001"
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
