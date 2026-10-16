# =============================================================================
# v26_klhi_lrdecay — v25_klhi + cosine LR decay（无 warmup，直接从 1e-6 衰减）
# =============================================================================
# 动机:
#   基于 v25 系列分析（2026-05-29）:
#
#   核心发现:
#     - v25_cosine: cosine 设计失败，根因是 warmup_steps_ratio=0.1 对应 50 步升温
#       被 critic_warmup=60 完全吞掉（步 1-60 actor LR=0），LR 实际在 step ~120
#       才到峰值 ~1e-6，此后 cosine decay 太慢（step 200 时 LR 仍 9.1e-7），
#       entropy 爆炸于 step 160 → val 在 step 200 崩塌至 0.238
#     - v25_klhi: KL=0.20 + entropy=0.005 成功阻止崩塌（step 316 仍无崩），
#       OOD b@4=0.947 为历史最高，但 mean@4 step 150 后持续缓降
#     - mean@4 缓降根因假设：actor LR 恒定 1e-6 → 后期更新仍偏大 → 策略
#       逐步向少数 lucky case 集中，整体分布退化
#
#   本实验假设:
#     在 v25_klhi 稳定底座基础上，将 cosine decay 的 warmup_steps_ratio 设为 0，
#     即 LR 从训练第 1 步开始就以 cosine 曲线衰减（从 1e-6 → 1e-7）：
#       1. 跳过升温期，彻底消除 critic_warmup 对 warmup 的干扰
#       2. LR 在关键的 step 100-200 区间（峰值前后）显著低于 1e-6，
#          理论上减缓 mean@4 退化
#       3. 保留 v25_klhi 的 KL/entropy 配置，验证两者可叠加
#
#   与 v25_cosine 的区别:
#     v25_cosine: warmup_steps_ratio=0.1 → LR 先升后降，warmup 被吞
#     v26_klhi_lrdecay: warmup_steps_ratio=0 → LR 只降不升（从 step 1 开始）
#
#   消融位置 (相对 v25_klhi):
#     ★ EXTRA_OVERRIDES: optim.lr_scheduler_type=cosine
#                        optim.lr_warmup_steps_ratio=0      ← 无 warmup
#                        optim.min_lr_ratio=0.1             ← min LR=1e-7
#     其他全部不变（KL=0.20, entropy=0.005, TOTAL_STEPS=2000）
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v26_klhi_lrdecay.sh \
    > v26_klhi_lrdecay.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v26_klhi_lrdecay"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # 94 scenes, 28662 tasks（与 v24/v25 共用）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

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
ACTOR_LR="1e-6"                    # 基础 LR；cosine 将从此值衰减到 1e-7

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

# === ★ cosine LR decay（无 warmup，直接从 step 1 开始衰减）===
# lr_warmup_steps_ratio=0  → 跳过升温，LR 从第 1 步开始按 cosine 曲线下降
# min_lr_ratio=0.1         → 最小 LR = 1e-6 × 0.1 = 1e-7
# 在 step 100 时 LR ≈ 9.75e-7，step 500 ≈ 8.5e-7，step 1000 ≈ 5.5e-7
export EXTRA_OVERRIDES="\
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
