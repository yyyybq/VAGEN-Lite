# =============================================================================
# v29_grpo — v28 基础配置 + 消融: 仅改算法 PPO → GRPO
# =============================================================================
# 动机（基于 v28 collapse 分析，2026-06-03）:
#
#   v28 的 collapse 根因：固定 π_ref 随训练步数增加而过时，
#   KL 锚点失效 → entropy 从 0.700 (step200) 升至 1.419 (step362)。
#
#   GRPO 消融假设：
#     - GRPO 不依赖固定 π_ref（无 KL 惩罚），advantage 由同组内 N=4 轨迹的
#       相对奖励归一化计算，从根本上消除"reference 过时"问题。
#     - 若 v29 在 step 300+ 仍保持 entropy < 1.0，说明 reference staleness
#       是 v28 collapse 的主因，GRPO 是有效修复。
#     - 若 v29 与 v28 同样 collapse，说明是奖励方差（rew_var=75-212）更主要。
#
# 相对 v28 的唯一差异：
#   ★ ADV_ESTIMATOR: masked_gae → grpo
#   ★ ENV_CONFIG: env_config_v24_100scenes_lm.yaml（near_success_bonus 0.2→0.5）
#   其他所有参数（LR、cosine schedule、batch size、KL config等）与 v28 完全相同
#
#   注意：GRPO 时 critic 自动禁用（need_critic() 检测到非 GAE 返回 False），
#         无需显式设置；USE_KL_LOSS 保留但 GRPO 路径中不生效。
#
# 消融矩阵：
#   v28: PPO + no_concat_gae, success_reward=50, near_success_bonus=0.2  ← 已跑（collapse）
#   v29: GRPO,               success_reward=50, near_success_bonus=0.5  ← 本实验：仅改算法
#   v30: PPO + no_concat_gae, success_reward=5,  near_success_bonus=0.5  ← 仅改奖励
#
# 启动：
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v29_grpo.sh \
    > v29_grpo.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v29_grpo"
ENV_CONFIG="env_config_v24_100scenes_lm.yaml"   # ★ near_success_bonus=0.5（last-mile 增强）
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

# === Critic（GRPO 下自动禁用，此处参数不生效）===
CRITIC_LR="2e-5"
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"             # 与 v28 相同

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

# === ★ Algorithm: GRPO（唯一与 v28 不同之处）===
ADV_ESTIMATOR="grpo"        # ★ masked_gae → grpo
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
