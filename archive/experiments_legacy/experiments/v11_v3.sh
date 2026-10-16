# v11_v3: 修复 entropy 爆炸 + GPU OOM（继承 v11 核心修复）
#         原名 v12_entropy_fix，按版本序重命名为 v11_v3
#
# ============================================================
# v11 失败根因分析（step 1–300）
# ============================================================
# [根本问题 1] GPU 4（渲染卡）OOM ★
#   gpu_holder 在渲染 GPU 上预分配 35% ≈ 49 GiB 内存
#   多个 AgentLoopWorker 同时在 GPU 4 加载 GS 场景（每个 ~10-12 GiB）
#   8 workers × 11 GiB ≈ 88 GiB + holder 49 GiB = 137 GiB ≈ 满载 → OOM
#   已修复：run_experiment.sh 中渲染卡 HOLDER_MEM_FRAC=0.35 → 0.0
#
# [根本问题 2] entropy 爆炸（step ~30-50 已崩溃）★★
#   v11 entropy 轨迹（步长 NR=1,20,50,100,150,200,250,300）:
#     0.74 → 0.66 → 4.07 → 7.31 → 9.84 → 9.77 → 7.61 → 9.83
#   Step 50 entropy=4.07，Step 150 entropy=9.84（≈最大值 ln(32000)≈10.37）
#   原因：entropy_coeff=0.008 过高；OOM 导致 reward 信号污染，
#         KL loss 无法拉回，entropy 快速逃逸到均匀分布
#   注：v10 同样在 step 193-196 发生 entropy 爆炸（entropy_coeff=0.008）
#   v11 因为 OOM 更早爆炸（step ~40）
#
# [关键观察] v11 critic bug 已修复（成功）
#   step 1-4 critic/returns/mean ≈ -0.13 ~ -0.24（合理值，非 v10 的 -99）
#   Critic 已正确工作，该修复有效
#
# [v11 checkpoints 状态]
#   saved: step 80, 160, 240（save_freq=80）
#   但 step 50 entropy 已 >4.0 → 所有 checkpoint 均在 entropy 爆炸后
#   → 不可用于 resume，必须从头训练
#
# ============================================================
# v11_v3 核心修复
# ============================================================
# [修复 1] GPU OOM：run_experiment.sh 渲染卡 HOLDER_MEM_FRAC=0.35 → 0.0
#   渲染卡只需维持 SM 利用率（matmul），不预分配显存
#   为 GS 渲染 workers 留出充足 GPU 内存
#
# [修复 2] entropy_coeff: 0.008 → 0.001（核心）★
#   v10、v11 均在 entropy_coeff=0.008 下发生 entropy 爆炸
#   0.001 减少 8x，足以防止爆炸，同时保持探索
#   对比：标准 PPO 通常用 0.0 ~ 0.01，导航任务 action 空间小建议更低
#
# [修复 3] GPU_MEM_UTIL: 0.5 → 0.4 ★
#   原 v12_entropy_fix 以 GPU_MEM_UTIL=0.5 启动，多实验并行时
#   GPU free mem 仅 ~63 GiB < 需求 69.9 GiB → startup 失败（0 步）
#   0.4 只需 ~55 GiB，安全裕量充足
#
# [继承 v11 全部修复]
#   MAX_TURNS=50（核心，v10 的 15→50 修复）
#   256×256 图像（防 ViT OOM）
#   Critic bug 已修复（ray_trainer.py + dp_critic.py）
#   MAX_TRAJECTORY_LENGTH=28000
#   WINDOW_SIZE=1
#
# ============================================================
# v11_v3 参数总结（相对 v11 变化加 ★）
# ============================================================
#   ENTROPY_COEFF:  ★ 0.001（was 0.008，减少 8x，防 entropy 爆炸）
#   GPU_MEM_UTIL:   ★ 0.4（was 0.5，防并行启动 OOM）
#   GPU holder:     ★ HOLDER_MEM_FRAC=0.0（已在 run_experiment.sh 修复，防 OOM）
#   RESUME:         disable（从头训练）
#   --- 以下继承 v11 ---
#   env_config:     env_config_single_action_256.yaml
#   MAX_TURNS:      50
#   MAX_TRAJECTORY_LENGTH: 28000
#   WINDOW_SIZE:    1
#   MAX_RESPONSE_LENGTH: 512
#   kl_loss_coef:   0.015
#   critic_warmup:  10
#   cliprange_value: 0.8
#   critic_lr:      2e-5
#   grad_clip:      0.5
#   actor_lr:       1e-6
#   TRAIN_BATCH_SIZE: 24
# ============================================================
#
# 启动命令:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v11_v3.sh \
    > v11_v3.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v11_v3"
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === ★ 核心修复：entropy_coeff 0.008 → 0.001 ===
ENTROPY_COEFF="0.001"

USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4  # ★ 0.5→0.4：多实验并行时 GPU free mem ~63GB < 70GB；0.4 只需 ~55GB

CRITIC_LR="2e-5"
CRITIC_WARMUP=10
CLIPRANGE_VALUE="0.8"

GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

MAX_TURNS=50
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=28000
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=2048

TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8
N_TRAJECTORY=1

PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
