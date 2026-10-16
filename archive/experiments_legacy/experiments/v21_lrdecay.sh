# =============================================================================
# v21_lrdecay — v20_winner + cosine actor LR decay
# =============================================================================
# 动机:
#   v19/v20 全家桶呈现一个高度一致的现象 —— val best@4 在 step ~40 达峰 (0.82–0.88),
#   随后开始单调下滑或剧烈震荡 (v19_thr70_andgate 在 step 120→0.036, v19_no_farm
#   在 step 180→0).  这是典型的 "actor LR 在过 sweet spot 后过大" 表现:
#   策略不再做 fine-tune 而开始随 advantage 噪声漂移.
#
# yaml: 沿用 v20_winner 的 env_config_v19_no_farm_dual.yaml
# sh:   v20_winner 全部稳定补丁 + cosine LR schedule
#
# 改动 (相对 v20_winner):
#   ★ actor.optim.lr_scheduler_type:   constant -> cosine
#   ★ actor.optim.lr_warmup_steps_ratio: 0 -> 0.10  (前 ~50 步线性热身到 ACTOR_LR
#       avoid 初期 critic 还没 warmup 完成时 actor 走太远)
#   ★ actor.optim.min_lr_ratio:        - -> 0.1   (终端 LR = 1e-7)
#   ★ TOTAL_STEPS:                     2000 -> 500  (cosine 在 500 步内走完一个周期;
#       2000 步的 cosine 会让 step 40 的 LR 几乎没变)
#
# 期望:
#   - step 0–50:   LR 0 → 1e-6 (warmup)
#   - step 50–500: LR 1e-6 → 1e-7 (cosine decay)
#   - step ~40 峰值之后, LR 在 step 80 时已 ~7e-7, step 200 时 ~3e-7
#     → 同样的 advantage 信号产生更小的 step, 抑制崩塌轨迹
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v21_lrdecay.sh \
    > v21_lrdecay.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v21_lrdecay"
ENV_CONFIG="env_config_v19_no_farm_dual.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (与 v20_winner 一致) ===
ENTROPY_COEFF="0.002"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling ===
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic (与 v20_winner 一致) ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"        # peak LR; cosine 终端 = ACTOR_LR * min_lr_ratio = 1e-7

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
TOTAL_STEPS=500          # ★ 2000 -> 500  (cosine 周期对齐预期 plateau 步数)
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === LR schedule (通过 run_experiment.sh 的 EXTRA_OVERRIDES 注入) ===
# verl optim 字段: actor_rollout_ref.actor.optim.{lr_scheduler_type,
#   lr_warmup_steps_ratio, min_lr_ratio}
export EXTRA_OVERRIDES="\
actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.10 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1"
