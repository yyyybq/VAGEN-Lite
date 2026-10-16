# =============================================================================
# v26_klhi_lr5e7 — v25_klhi + actor LR 减半（1e-6 → 5e-7）
# =============================================================================
# 动机:
#   基于 v25 系列分析（2026-05-29）:
#
#   核心发现:
#     - 所有实验（v19-v25）的崩塌或 mean@4 退化，均发生在 actor LR 到达或
#       维持峰值 ~1e-6 时段内
#     - v25_klhi 以 KL=0.20 + entropy=0.005 阻止了崩塌，但 LR=1e-6 恒定导致
#       step 150 后 mean@4 持续缓降（step 300 时 ID m@4=0.238）
#     - v25_cosine 试图用 LR decay 解决此问题，但 warmup 设计缺陷导致失败
#     - 最直接的假设：LR=1e-6 对于这个任务在 step 100+ 之后偏大，
#       减半到 5e-7 是最小改动、最易解读的验证
#
#   本实验假设:
#     ACTOR_LR: 1e-6 → 5e-7（恒定，无 schedule）：
#       1. 更保守的梯度更新 → 减缓 step 100+ 的 mean@4 退化
#       2. 同样保留 v25_klhi 的 KL/entropy 稳定配置
#       3. 单变量消融：仅改 LR，其他完全对齐 v25_klhi
#
#   预期 vs v25_klhi:
#     - 峰值可能出现更晚（step 150-200 而非 100）
#     - 峰值后退化速度减缓（mean@4 维持时间更长）
#     - 代价：前期学习可能更慢（step 50 时 m@4 可能低于 v25_klhi）
#
#   消融位置 (相对 v25_klhi):
#     ★ ACTOR_LR: 1e-6 → 5e-7
#     其他全部不变（KL=0.20, entropy=0.005, TOTAL_STEPS=2000）
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v26_klhi_lr5e7.sh \
    > v26_klhi_lr5e7.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v26_klhi_lr5e7"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # 94 scenes, 28662 tasks（与 v24/v25 共用）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="auto"

# === Actor ===
ENTROPY_COEFF="0.005"              # 同 v25_klhi
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"                # 同 v25_klhi
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
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"                    # ★ 1e-6 → 5e-7：减半，其余完全对齐 v25_klhi

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=12
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=80
TEST_FREQ=50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
