# v14_v3: 基于 v14_v2 的稳定性修复
#
# ============================================================
# v14_v2 → v14_v3 修复（解决 premature done 坍塌 + response 截断崩溃）
# ============================================================
# [问题诊断] v14_v2 在 step ~13 后出现三重崩溃信号：
#   1. response_length/clip_ratio 从 5% 飙升至 40%（step 18）
#      → MAX_RESPONSE_LENGTH=512 太短，模型输出长 think + 动作序列被截断
#      → 截断后 format_ok=False → reward=0 → 梯度推模型输出更长 → 恶循环
#   2. episode_length/mean 从 17.4 坍缩至 3.9（step 20）
#      → 模型学到 shortcut：第 2-3 步直接 done，用 premature_done_penalty=-0.3
#         换掉无休止积累的 step_penalty（与 v12_v2 完全相同的模式）
#   3. reward_variance 降至 0.005（step 20）
#      → N=4 条轨迹几乎都走同样的早退出路径 → PPO 梯度接近零 → 学习停滞
#
# [修复 1] MAX_RESPONSE_LENGTH: ★ 512 → 1024
#   给模型足够空间输出完整动作序列，避免截断触发格式错误正反馈循环
#   （与 v12_v3 相同修复）
#
# [修复 2] ENTROPY_COEFF: ★ 0.008 → 0.02
#   增强熵正则化系数，约束 entropy 在合理范围内，防止爆炸
#   同时惩罚"只输出 done"这种低熵的退化策略
#   （与 v12_v3 相同修复）
#
# [修复 3] KL_LOSS_COEF: ★ 0.015 → 0.05
#   增强 KL 散度惩罚，防止 policy 偏离 reference model 过远
#   reference model 不会早退出，KL 惩罚可有效遏制 done 坍塌
#   （与 v12_v3 相同修复）
#
# [修复 4] CRITIC_WARMUP: ★ 10 → 30
#   v14_v2 将 warmup 压缩到 10，导致 Critic 未稳定就开始更新 Actor
#   恢复适度 warmup，给 Critic 足够时间收敛再驱动 Actor 更新
#   （与 v12_v3 相同修复）
#
# [继承 v14_v2 的改动]
#   N_TRAJECTORY=4, HIGH_LEVEL_GAMMA=0.99
#   TRAIN_BATCH_SIZE=8（8×4=32 总轨迹）
#   GPU_MEM_UTIL=0.4（避免 vLLM+FSDP 同时驻存时 OOM）
#   所有其他超参不变
#
# ============================================================
# v14_v3 参数总结（相对 v14_v2 的变化用 ★ 标注）
# ============================================================
#   MAX_RESPONSE_LENGTH: ★ 512 → 1024：避免截断触发格式错误恶循环
#   ENTROPY_COEFF:       ★ 0.008 → 0.02：防止 entropy 爆炸 + done 坍塌
#   KL_LOSS_COEF:        ★ 0.015 → 0.05：更强 KL 约束，遏制 policy 发散
#   CRITIC_WARMUP:       ★ 10 → 30：Critic 稳定后再驱动 Actor 更新
#   其余全部与 v14_v2 相同
# ============================================================
#
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v14_n4_gamma99_v3.sh \
    > v14_n4_gamma99_v3.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v14_n4_gamma99_v3"
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 稳定性参数（相对 v14_v2 的修复用 ★ 标注）===
ENTROPY_COEFF="0.02"   # ★ 0.008 → 0.02：防止 entropy 爆炸 + 遏制 done 坍塌
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"    # ★ 0.015 → 0.05：更强 KL 约束，防止 policy 发散
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4  # 继承 v14_v2：vLLM+FSDP 同驻时 free mem 约束

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30       # ★ 10 → 30：更充分的 warmup，Critic 稳定后再驱动 Actor
CLIPRANGE_VALUE="0.8"

# === 梯度（继承 v14_v2）===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数 ===
MAX_TURNS=50
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=28000
MAX_RESPONSE_LENGTH=1024  # ★ 512 → 1024：避免重复动作截断触发格式错误
MAX_PROMPT_LENGTH=2048

# === N=4 + TRAIN_BATCH_SIZE（继承 v14_v2）===
N_TRAJECTORY=4             # 每个 prompt 采样 4 条轨迹
TRAIN_BATCH_SIZE=8         # 8×4=32 总轨迹；8/4=2 per GPU ✓
VAL_BATCH_SIZE=4

# === PPO mini-batch（继承 v14_v2）===
PPO_MINI_BATCH_SIZE=8      # 32/8=4 mini-batches; 8/4=2 per GPU ✓
MINI_BATCH_SIZE=8          # 32/8=4 rollout mini-batches ✓

# === 训练参数 ===
SAVE_FREQ=30
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数（继承 v14_v2）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.99"    # 继承 v14：50 步 episode 下 success 信号不被折扣压制
KL_COEF="0.001"
LAM="0.95"
