# v12_v3: 基于 v12_v2 的稳定性修复
#
# ============================================================
# v12_v2 → v12_v3 修复（解决 entropy explosion + response 截断崩溃）
# ============================================================
# [问题诊断] v12_v2 在 step ~80 后出现三重崩溃信号：
#   1. response_length/clip_ratio 从 5% 飙升至 100%
#      → MAX_RESPONSE_LENGTH=512 太短，模型输出大量重复动作被截断
#      → 截断后 format_ok=False → reward=0 → 梯度推模型输出更多动作 → 恶循环
#   2. actor/entropy 从 0.76 爆炸至 10+（正常范围 ~0.5-2.0）
#      → entropy regularization 强度不足
#   3. episode_length/mean 坍缩至 2.0（仅 1 次环境交互）
#      → 策略彻底退化，只输出无意义重复动作或乱码
#
# [修复 1] MAX_RESPONSE_LENGTH: ★ 512 → 1024
#   给模型足够空间输出完整动作序列，避免截断触发格式错误
#
# [修复 2] ENTROPY_COEFF: ★ 0.008 → 0.02
#   增强熵正则化系数，约束 entropy 在合理范围内，防止爆炸
#
# [修复 3] KL_LOSS_COEF: ★ 0.015 → 0.05
#   增强 KL 散度惩罚，防止 policy 偏离 reference model 过远
#
# [修复 4] CRITIC_WARMUP: ★ 10 → 30
#   v12_v2 将 warmup 从 40 压缩到 10，导致 Critic 未稳定就开始更新 Actor
#   恢复适度 warmup，给 Critic 足够时间收敛再驱动 Actor 更新
#
# [继承 v12_v2 的改动]
#   N_TRAJECTORY=2, TRAIN_BATCH_SIZE=12
#   reward rebalance: potential 1→5, vis 0.3→0.05, success 1→5
#   所有其他超参不变
#
# ============================================================
# v12_v3 参数总结（相对 v12_v2 的变化用 ★ 标注）
# ============================================================
#   MAX_RESPONSE_LENGTH: ★ 512 → 1024
#   ENTROPY_COEFF:       ★ 0.008 → 0.02
#   KL_LOSS_COEF:        ★ 0.015 → 0.05
#   CRITIC_WARMUP:       ★ 10 → 30
#   其余全部与 v12_v2 相同
# ============================================================

# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v12_n2_max_turns50_v3.sh \
    > v12_n2_max_turns50_v3.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v12_n2_max_turns50_v3"
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 稳定性参数（相对 v12_v2 的修复用 ★ 标注）===
ENTROPY_COEFF="0.02"   # ★ 0.008 → 0.02：防止 entropy explosion
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"    # ★ 0.015 → 0.05：更强 KL 约束，防止 policy 发散
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.5

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30       # ★ 10 → 30：更充分的 warmup，Critic 稳定后再驱动 Actor
CLIPRANGE_VALUE="0.8"

# === 梯度（继承 v12）===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数 ===
MAX_TURNS=50
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=20000
MAX_RESPONSE_LENGTH=1024  # ★ 512 → 1024：避免重复动作截断触发格式错误
MAX_PROMPT_LENGTH=2048

# === N=2 + TRAIN_BATCH_SIZE/2（继承 v12）===
N_TRAJECTORY=2             # 每个 prompt 采样 2 条轨迹
TRAIN_BATCH_SIZE=12        # 12×2=24 总轨迹
VAL_BATCH_SIZE=4

# === PPO mini-batch（继承 v12）===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === 训练参数 ===
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数（继承 v12）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
