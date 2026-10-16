# v14: N_TRAJECTORY=4 + gamma 0.99，基于 v12 改进
#
# ============================================================
# v12 → v14 改进
# ============================================================
# [改进 1] N_TRAJECTORY: 2 → 4
#   每个 prompt 采样 4 条独立轨迹（vs v12 的 2 条）：
#   - Critic 更稳健：4 条路径的 value 估计方差更低
#     预期 vf_explained_var > v12 的 ~0.2-0.35
#   - Actor 梯度质量提升：4 条路径的优劣对比更明确
#   - 为切换 GRPO（需 N≥2）提供更好的组内归一化基础
#
# [改进 2] HIGH_LEVEL_GAMMA: 0.95 → 0.99
#   同 v13 分析：50 步 episode 下 success_reward 的折现值从 0.08 提升到 0.61
#
# [配套调整] TRAIN_BATCH_SIZE: 12 → 8（总轨迹控制）
#   N=4 + TRAIN_BATCH_SIZE=8 → 总轨迹 32（vs v12 的 24）
#   TRAIN_BATCH_SIZE=8 满足 4-GPU 整除（8/4=2 per GPU ✓）
#   总轨迹 32 仍可控：no_concat 模式下 critic 按单 turn 独立前向，无 packed OOM 风险
#   注：若需与 v12 完全相同的总轨迹数(24)，需 TRAIN_BATCH_SIZE=6，但 6/4 不整除
#
# [其余全部继承 v12]
#   WINDOW_SIZE=1, MAX_TURNS=50, MAX_TRAJECTORY_LENGTH=28000
#   256×256 图像, premature_done_penalty=-0.3
#   所有稳定性超参不变
#
# ============================================================
# v14 参数总结（相对 v12 的变化用 ★ 标注）
# ============================================================
#   N_TRAJECTORY:     ★ 2 → 4
#   TRAIN_BATCH_SIZE: ★ 12 → 8   (8×4=32 总轨迹；8/4=2 per GPU ✓)
#   PPO_MINI_BATCH_SIZE: ★ 12 → 8 (32/8=4 mini-batches; 8/4=2 per GPU ✓)
#   HIGH_LEVEL_GAMMA: ★ 0.95 → 0.99
#   其余与 v12 完全相同
# ============================================================
#
# 启动命令：
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v14_n4_gamma99.sh \
    > v14_n4_gamma99_v2.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v14_n4_gamma99_v2"  # _v2 suffix: fresh run after reward-shaping rebalance (potential 1->5, vis 0.3->0.05, success 1->5) + critic_warmup 40->10. Old wandb run preserved.
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 继承 v12 稳定性参数 ===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4  # 0.5->0.4: vLLM 要求 free >= util*total(140GB)，FSDP init 占 ~76GB 后 free 只剩 ~64GB < 70GB 起不来；降到 56GB OK

# === Critic（继承 v12）===
CRITIC_LR="2e-5"
CRITIC_WARMUP=10  # reward rebalance (potential 1->5, vis 0.3->0.05, success 1->5): warmup shortened to start actor updates sooner
CLIPRANGE_VALUE="0.8"

# === 梯度（继承 v12）===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数（继承 v12）===
MAX_TURNS=50
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=28000
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=2048

# === ★ 核心变化：N=4 + 调整 TRAIN_BATCH_SIZE ===
N_TRAJECTORY=4             # ★ 2 → 4：每个 prompt 采样 4 条轨迹
TRAIN_BATCH_SIZE=8         # ★ 12 → 8：8×4=32 总轨迹；8/4=2 per GPU ✓
VAL_BATCH_SIZE=4           # 继承 v12

# === PPO mini-batch（调整适配 32 总轨迹）===
PPO_MINI_BATCH_SIZE=8      # ★ 12 → 8：32/8=4 mini-batches; 8/4=2 per GPU ✓
MINI_BATCH_SIZE=8          # 32/8=4 rollout mini-batches ✓

# === 训练参数 ===
SAVE_FREQ=30
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.99"    # ★ 0.95 → 0.99：修复 50 步 episode 的 success 信号折扣问题
KL_COEF="0.001"
LAM="0.95"
