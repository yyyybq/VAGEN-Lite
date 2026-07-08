# =============================================================================
# v31_grpo_rewscale — 双重修复: GRPO 算法 + 低方差奖励结构
# =============================================================================
# 动机（基于 v29/v30 消融结果，2026-06-08）:
#
#   v29 消融（仅 GRPO）: entropy 在 step 77 时 0.554，极度健康；消除 reference
#     staleness 问题，但数据点不足（仅 step50 一个 val）。
#
#   v30 消融（仅奖励缩放）: 历史首次 entropy 自愈 (step194: 0.537 < step1: 0.534)；
#     critic score 震荡消失（0.12-0.65 vs v28 的 1.27-3.74）；
#     ID_m4 稳定爬升至 0.512，未见 v28 那样的 step320 崩塌。
#
#   v31 假设：
#     GRPO（消除 reference staleness）+ 低方差奖励（消除 critic 震荡）双管齐下，
#     可能在 step300+ 仍保持 entropy < 0.7，最终 ID_m4 超过 v26 的历史峰值 0.607。
#
# 相对 v29 的改动:
#   ★ ENV_CONFIG: env_config_v24_100scenes_lm.yaml → env_config_v24_100scenes_rewscale.yaml
#     (success_reward=5, potential_scale=1.0, near_bonus=0.5)
#
# 相对 v30 的改动:
#   ★ ADV_ESTIMATOR: masked_gae → grpo （PPO → GRPO，无 critic）
#
# GPU: 5卡 H200 (4 train + 1 render)，与 v29/v30 一致
#
# 消融矩阵（v28 基准）:
#   v28: PPO, success=50, scale=0.5   → collapse @step320        ← 已完成
#   v29: GRPO, success=50, scale=0.5  → 仅改算法                 ← step77 停
#   v30: PPO,  success=5,  scale=1.0  → 仅改奖励，entropy 自愈   ← step194 停
#   v31: GRPO, success=5,  scale=1.0  → 双改，期望协同效果        ← 本实验
#
# 启动 (7卡 H200):
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v31_grpo_rewscale.sh \
    > v31_grpo_rewscale.log 2>&1 &
echo "PID: $!"
RUN
EXPERIMENT_NAME="v31_grpo_rewscale"      # ★ 显式命名，避免落到 unnamed_*
ENV_CONFIG="env_config_v24_100scenes_rewscale.yaml"   # ★ success=5, scale=1.0, near_bonus=0.5
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"          # GRPO 路径下不生效，保留兼容性
KL_LOSS_COEF="0.20"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling ===
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic (GRPO 下 critic 自动禁用，但保留以备 fallback) ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"

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
SAVE_FREQ=50
TEST_FREQ=50
TOTAL_STEPS=1000
VAL_BEFORE_TRAIN="True"

# === Algorithm: GRPO (★ 核心改动) ===
ADV_ESTIMATOR="grpo"        # ★ masked_gae → grpo
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === cosine LR decay (与 v29/v30 相同) ===
export EXTRA_OVERRIDES="\
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

# === OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
