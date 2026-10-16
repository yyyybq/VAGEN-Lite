# =============================================================================
# v27_from_v25ckpt — 从 v25_klhi step-240 权重出发，以极低 LR 重启 RL 训练
# =============================================================================
# 科学问题（实验 A）:
#   v25_klhi 在 step ≈ 150 达到 ID_m4=0.500 / OOD_b4=0.947 的峰值，之后持续退化
#   （step 300 时 ID_m4=0.238）。本实验回答：
#
#     「峰值后退化是否可逆？以极低 LR 从已退化的 RL 权重重启能否恢复性能？」
#
#   若能恢复（m@4 回升）→ 退化由 LR 过大导致的参数漂移，降低 LR 是正路
#   若不能恢复          → 退化有更深层原因（表示空间退化、critic 偏差、
#                         reward hacking 导致的 mode collapse 等），不可仅靠 LR 解决
#
# 与现有实验的关系:
#   v25_klhi (base):    从头 RL，LR=1e-6（恒定）→ step 150 峰值后退化
#   v26_klhi_lr5e7:     从头 RL，LR=5e-7（恒定）→ 对照：低 LR 是否延缓峰值退化
#   v26_klhi_lrdecay:   从头 RL，LR=1e-6→1e-7（cosine）→ 对照：LR 衰减是否有效
#   v27_from_v25ckpt:   ★ 从 v25_klhi step-240 HF 权重开始，LR=1e-7（恒定）
#                         → 验证：退化可逆性 + RL warm-start 的实际价值
#
# 检查点选择说明:
#   v25_klhi 只有 global_step_240 存有完整模型权重（80/160 只是 DataLoader stub）。
#   step 240 对应的 val（step 250）显示 ID_m4=0.333、OOD_b4=0.579，已过峰值。
#   以此为起点重新训练，能回到 step 150 的峰值水平即为成功。
#
# 技术设计:
#   - MODEL_PATH 直接指向 HuggingFace 格式权重（只加载模型，不加载 optimizer/scheduler）
#   - RESUME_MODE=disable → optimizer 全新初始化，从 step 0 开始计数
#   - CRITIC_WARMUP=60 → critic 需要从头热身（无历史 critic checkpoint 可用）
#   - VAL_BEFORE_TRAIN=True → 第一步就做 val，确认起点性能（应接近 step-250 水平）
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v27_from_v25ckpt.sh \
    > v27_from_v25ckpt.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v27_from_v25ckpt"

# ★ 关键：使用 v25_klhi step-240 的 HuggingFace 格式权重作为初始化
MODEL_PATH="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/exps/vagen_active_spatial/v25_groupadv_100scenes_klhi/checkpoints/global_step_240/actor/huggingface"

ENV_CONFIG="env_config_v24_100scenes.yaml"    # 与 v24/v25/v26 共用（94 scenes）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

# ★ disable：只通过 MODEL_PATH 加载模型权重，optimizer 从零初始化
RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"        # 同 v25_klhi：保持探索
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"          # 同 v25_klhi：KL 约束防崩塌
                              # 注意：reference model 也从 MODEL_PATH 初始化，
                              # 即 KL 相对于 v25_klhi step-240 权重计算，起点 KL=0
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
CRITIC_WARMUP=60              # 无 critic 历史权重，需要热身；全新 critic 在步 1-60 学习
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-7"               # ★ 核心变量：比 v25_klhi 低 10x，比 v26_lr5e7 低 5x
                              # 假设：极小的梯度步长 → 保留 RL 已学到的策略轮廓
                              # 同时允许 critic 重新校准价值估计

# === 轨迹（完全对齐 v25_klhi）===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data（GroupAdv: 同 prompt 4 rollout）===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=12
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=50                  # 比 v25_klhi（80）更密，每次 val 都有对应 ckpt 可分析
TEST_FREQ=50
TOTAL_STEPS=1000              # 1000 步足够观察是否恢复；可后续 resume 延长
VAL_BEFORE_TRAIN="True"       # ★ 先测一次 val，建立起点 baseline
                              # 预期：接近 v25_klhi step-250 水平（ID_m4≈0.33，OOD_b4≈0.58）
                              # 若 val@0 明显更好/更差 → 说明存在随机采样方差

# === Algorithm（PPO + no_concat_gae）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
