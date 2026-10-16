# v12: N_TRAJECTORY=2，基于 v11 改进
#
# ============================================================
# v11 → v12 改进
# ============================================================
# [核心变化] N_TRAJECTORY: 1 → 2
#   同一 prompt 采样 2 条独立轨迹：
#   1. Critic 更稳健：相同起点的不同路径 → value 估计方差更低
#      预期 vf_explained_var 从 v11 的 ~0.05-0.1 提升至 ~0.2-0.35
#   2. Actor 梯度信号更丰富：2 条路径的优劣对比明确
#   3. 为未来 GRPO-style 组内归一化奠基
#
# [配套调整] TRAIN_BATCH_SIZE: 24 → 12（总轨迹数保持 24 不变）
#   12 unique prompts × 2 轨迹 = 24 总轨迹 = v11 的 24×1
#   → Critic per-GPU packed length 不变，无 OOM 风险
#   → Actor/rollout 计算量不变
#
# [其余全部继承 v11]
#   MAX_TURNS=50, 256×256 图像, premature_done_penalty=-0.3
#   critic bug 修复（ray_trainer.py + dp_critic.py 已在代码层修复）
#   所有稳定性超参不变
#
# ============================================================
# v12 参数总结（相对 v11 的变化用 ★ 标注）
# ============================================================
#   N_TRAJECTORY:     ★ 1 → 2
#   TRAIN_BATCH_SIZE: ★ 24 → 12  (12×2=24 总轨迹，与 v11 相同)
#   VAL_BATCH_SIZE:   ★ 8 → 4    (整除 4)
#   其余全部与 v11 相同
# ============================================================

# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v12_n2_max_turns50.sh \
    > v12_n2_max_turns50_v2.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v12_n2_max_turns50_v2"  # _v2 suffix: fresh run after reward-shaping rebalance (potential 1->5, vis 0.3->0.05, success 1->5) + critic_warmup 40->10. Old wandb run preserved.
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 继承 v11 稳定性参数 ===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.5

# === Critic（继承 v11）===
CRITIC_LR="2e-5"
CRITIC_WARMUP=10  # reward rebalance (potential 1->5, vis 0.3->0.05, success 1->5): warmup shortened to start actor updates sooner
CLIPRANGE_VALUE="0.8"

# === 梯度（继承 v11）===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数（继承 v11）===
MAX_TURNS=50
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=20000
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=2048

# === ★ 核心变化：N=2 + TRAIN_BATCH_SIZE/2 ===
N_TRAJECTORY=2             # ★ 1 → 2：每个 prompt 采样 2 条轨迹
TRAIN_BATCH_SIZE=12        # ★ 24 → 12：12×2=24 总轨迹，与 v11 的 24×1 相同
VAL_BATCH_SIZE=4           # ★ 8 → 4：整除 4 ✓

# === PPO mini-batch（总轨迹数不变，保持与 v11 一致）===
PPO_MINI_BATCH_SIZE=12     # 24/12=2 mini-batches; 12/4=3 per GPU ✓
MINI_BATCH_SIZE=8          # 24/8=3 rollout mini-batches ✓

# === 训练参数 ===
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数（继承 v11）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
