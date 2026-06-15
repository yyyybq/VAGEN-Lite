# =============================================================================
# v30_rew_scale — v28 基础配置 + 消融: 仅改奖励结构（降低 rew_var）
# =============================================================================
# 动机（基于 v28 collapse 分析，2026-06-03）:
#
#   v28 的 rew_var=75-212 的根本原因：
#     success_reward=50 vs 失败轨迹奖励 ≈ 0~5，组内方差 ≈ 50²/4 ≈ 625。
#     高方差导致 critic 震荡（crit_sc 在 1.27~3.74 间剧烈摆动），
#     一次噪声批次（step320 grad_norm=9.648）即可触发 entropy 爆炸。
#
#   奖励重设计假设：
#     将 success_reward 从 50 降至 5，将 potential_field_reward_scale 从 0.5
#     提升至 1.0，使两者量级对齐（最大进度奖励 ≈ 5m × 1.0 = 5.0 ≈ success_reward）：
#       - rew_var 从 75-212 → 预计 1-8（平方关系，降幅 ~100x）
#       - critic 不再需要建模量级差异 50x 的奖励，震荡应消失
#       - near_success_bonus=0.5 增强 last-mile 梯度（relative to success_reward=5）
#     若 v30 的 critic 震荡消失、grad_norm 稳定，说明 rew_var 是 v28 collapse 的次因。
#
# 相对 v28 的唯一差异（env_config 层面）：
#   ★ success_reward:             50.0 → 5.0  (降低 10x)
#   ★ potential_field_reward_scale: 0.5 → 1.0  (提升 2x，维持进度信号相对强度)
#   ★ near_success_bonus:          0.2 → 0.5  (last-mile 增强，与 v29 一致)
#   所有训练超参（LR、algorithm、batch 等）与 v28 完全相同
#
# 消融矩阵：
#   v28: PPO, success_reward=50, potential_scale=0.5, near_bonus=0.2  ← 已跑（collapse）
#   v29: GRPO, success_reward=50, potential_scale=0.5, near_bonus=0.5  ← 仅改算法
#   v30: PPO,  success_reward=5,  potential_scale=1.0, near_bonus=0.5  ← 本实验：仅改奖励
#
# 注意事项：
#   - success_reward 降低 10x 不代表学习信号变弱；进度奖励对应提升 2x 补偿。
#   - 若模型从未成功，success_reward 量级对梯度无直接影响；
#     影响的是成功/失败样本间的 advantage 差异（这正是降低的目标）。
#
# 启动：
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v30_rew_scale.sh \
    > v30_rew_scale.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v30_rew_scale"
ENV_CONFIG="env_config_v24_100scenes_rewscale.yaml"   # ★ success=5, scale=1.0, near_bonus=0.5
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
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

# === Critic ===
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

# === Algorithm（与 v28 完全相同）===
ADV_ESTIMATOR="masked_gae"  # PPO + no_concat_gae，不变
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === cosine LR decay（与 v28 完全相同）===
export EXTRA_OVERRIDES="\
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

# === OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
