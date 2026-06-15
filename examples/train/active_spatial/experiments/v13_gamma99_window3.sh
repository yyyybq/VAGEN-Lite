# v13: gamma 0.95→0.99 + window_size 1→3，基于 v12 改进
#
# ============================================================
# v12 → v13 改进
# ============================================================
# [改进 1] HIGH_LEVEL_GAMMA: 0.95 → 0.99
#   v12 问题：50 步 episode 中，终点的 success_reward 折现后只剩
#     0.95^49 ≈ 0.08，而 step_penalty=-0.01×50=-0.5 是它的 6 倍
#   → 模型实际上在最小化 step_penalty，而不是追求 success
#   修复：gamma=0.99 → 0.99^49 ≈ 0.61，success 信号恢复显著
#     success_reward 的折现值 ≈ 0.61，vs step_penalty_total ≈ -0.5 → success 占主导
#
# [改进 2] WINDOW_SIZE: 1 → 3
#   window_size=1：模型只看到上一帧，无法判断移动方向和历史轨迹
#   window_size=3：模型看到最近 3 帧 obs + action，可感知：
#     - 是否在前进（连续 3 帧 score 变化趋势）
#     - 自己走了什么路径（避免原地打转）
#     - 当前方向是否正确
#
# [配套调整] MAX_PROMPT_LENGTH: 2048 → 4096
#   window_size=3 的每轮 prompt：
#     system ≈ 100 tok
#     3 × (256px image ~342tok + action text ~100tok) = 1326 tok
#     current obs ≈ 342 + 100 = 442 tok
#     total ≈ 1868 tok > 2048（原限制会截断）→ 调整为 4096
#
# [配套调整] MAX_TRAJECTORY_LENGTH: 28000 → 60000
#   window_size=3 时，每轮 (prompt+response) ≈ 1868+200 = 2068 tok
#   50 turns × 2068 ≈ 103,400 tok（全部 50 轮）
#   设 60000：有效约 29 轮，兼顾内存和足够的 episode 长度
#   （window_size=1 时 28000/~1000tok/turn ≈ 28 轮，故此次约等效）
#
# [其余全部继承 v12]
#   N_TRAJECTORY=2, TRAIN_BATCH_SIZE=12, 总轨迹=24
#   256×256 图像, MAX_TURNS=50, premature_done_penalty=-0.3
#   所有稳定性超参不变
#
# ============================================================
# v13 参数总结（相对 v12 的变化用 ★ 标注）
# ============================================================
#   HIGH_LEVEL_GAMMA: ★ 0.95 → 0.99
#   WINDOW_SIZE:      ★ 1 → 3
#   MAX_PROMPT_LENGTH:★ 2048 → 4096
#   MAX_TRAJECTORY_LENGTH: ★ 28000 → 60000
#   其余与 v12 完全相同
# ============================================================
#
# 启动命令：
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v13_gamma99_window3.sh \
    > v13_gamma99_window3_v2.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v13_gamma99_window3_v2"  # _v2 suffix: fresh run after reward-shaping rebalance (potential 1->5, vis 0.3->0.05, success 1->5) + critic_warmup 40->10. Old wandb run preserved.
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
GPU_MEM_UTIL=0.3  # 0.5->0.4: vLLM 要求 free >= util*total(140GB)，FSDP init 占 ~76GB 后 free 只剩 ~64GB < 70GB 起不来；降到 56GB OK

# === Critic（继承 v12）===
CRITIC_LR="2e-5"
CRITIC_WARMUP=10  # reward rebalance (potential 1->5, vis 0.3->0.05, success 1->5): warmup shortened to start actor updates sooner
CLIPRANGE_VALUE="0.8"

# === 梯度（继承 v12）===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数 ===
MAX_TURNS=50
WINDOW_SIZE=3              # ★ 1 → 3：看最近 3 帧历史
MAX_TRAJECTORY_LENGTH=60000  # ★ 28000 → 60000：支持 window=3 更长 prompt
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=4096     # ★ 2048 → 4096：容纳 window=3 的 prompt（~1868 tok）

# === N=2（继承 v12）===
N_TRAJECTORY=2
TRAIN_BATCH_SIZE=12        # 12×2=24 总轨迹；12/4=3 per GPU ✓
VAL_BATCH_SIZE=4

# === PPO mini-batch（继承 v12）===
PPO_MINI_BATCH_SIZE=12     # 24/12=2 mini-batches; 12/4=3 per GPU ✓
MINI_BATCH_SIZE=8          # 24/8=3 rollout mini-batches ✓

# === 训练参数 ===
SAVE_FREQ=30
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.99"    # ★ 0.95 → 0.99：50 步 episode 下 success 信号不再被折扣压制
KL_COEF="0.001"
LAM="0.95"
