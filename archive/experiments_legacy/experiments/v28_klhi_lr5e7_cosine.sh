# =============================================================================
# v28_klhi_lr5e7_cosine — 以 5e-7 为起点的余弦衰减（修复 v26_klhi_lrdecay 的设计缺陷）
# =============================================================================
# 动机:
#   基于 v26/v27 系列分析（2026-05-31）:
#
#   核心发现:
#     - v26_klhi_lr5e7: LR=5e-7 恒定，step150 时 ID_m4=0.607 超过 v25_klhi 历史
#       峰值(0.500)，三个验证点单调递增，训练完全健康（entropy<0.8, KL<0.003）
#     - v26_klhi_lrdecay: cosine 从 1e-6 起，但以 2000步为周期，step181 时
#       LR 仅衰减 0.8%（9.92e-7），等同于恒定 1e-6，完全复现了 v25_klhi 的
#       entropy 爆炸（step160 entropy=1.028, KL=0.01）
#     - v26_lr5e7 的 act_entropy (动作分布熵) 在 step180 降至 0.461（move_forward
#       占 87%），行为集中化是退化的先导信号
#
#   v26_lrdecay 的失败根因：
#     起点 LR=1e-6 本身就在不稳定区域，以 2000步为周期的余弦在前 200 步几乎不衰减
#     → 相当于在最不稳定的阶段一直维持最高 LR
#
#   v28 的修复逻辑：
#     以"已验证稳定"的 5e-7 为起点，以 1000步为周期做余弦衰减到 5e-8：
#       step 100: LR ≈ 4.5e-7  (轻微衰减，学习信号仍强)
#       step 300: LR ≈ 2.5e-7  (明显低于 v26_lr5e7，更保守)
#       step 500: LR ≈ 5e-8    (到达最小值，防止 action concentration)
#     关键改进 vs v26_lrdecay: 在 step100~200 时 LR 已实质性低于起点
#
#   消融位置 (相对 v26_klhi_lr5e7):
#     ★ EXTRA_OVERRIDES: optim.lr_scheduler_type=cosine
#                        optim.lr_warmup_steps_ratio=0      ← 无 warmup
#                        optim.min_lr_ratio=0.1             ← min LR=5e-8
#     ★ TOTAL_STEPS: 2000 → 1000  (保证余弦在训练中期完成衰减，后半段用最低LR)
#     其他全部不变（KL=0.20, entropy=0.005, ACTOR_LR=5e-7作为峰值）
#
#   科学假设:
#     若 v26_lr5e7 在 step 200+ 出现 action concentration 加剧和 val 退化，
#     v28 应通过 LR 渐进衰减延缓该退化。若两者最终均退化，退化步数差异
#     将量化"从稳定起点做余弦衰减"的额外收益。
#
# LR 曲线对比 (同步):
#   step | v25(1e-6) | v26_lr5e7(5e-7) | v28_cosine
#   100  | 1e-6      | 5e-7            | ~4.5e-7
#   200  | 1e-6      | 5e-7            | ~3.5e-7
#   300  | 1e-6      | 5e-7            | ~2.5e-7
#   500  | 1e-6      | 5e-7            | 5e-8 (min)
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v28_klhi_lr5e7_cosine.sh \
    > v28_klhi_lr5e7_cosine_resume.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v28_klhi_lr5e7_cosine"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # 94 scenes, 28662 tasks（与 v24/v25/v26 共用）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="auto"

# === Actor ===
ENTROPY_COEFF="0.005"              # 同 v25_klhi / v26_klhi_lr5e7
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"                # 同 v25_klhi / v26_klhi_lr5e7
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
ACTOR_LR="5e-7"                    # ★ 与 v26_klhi_lr5e7 相同；cosine 将从此值衰减到 5e-8

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
TOTAL_STEPS=1000                   # ★ 2000 → 1000: 余弦在训练中段完成有效衰减
VAL_BEFORE_TRAIN="True"            # 记录起点性能

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === ★ cosine LR decay（以 5e-7 为峰值，无 warmup，直接从 step 1 开始衰减）===
# lr_warmup_steps_ratio=0  → 跳过升温，LR 从第 1 步开始按 cosine 曲线下降
# min_lr_ratio=0.1         → 最小 LR = 5e-7 × 0.1 = 5e-8
# 以 TOTAL_STEPS=1000 为周期:
#   step 100: LR ≈ 4.5e-7
#   step 300: LR ≈ 2.5e-7
#   step 500: LR = 5e-8 (最小值后维持)
export EXTRA_OVERRIDES="\
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
