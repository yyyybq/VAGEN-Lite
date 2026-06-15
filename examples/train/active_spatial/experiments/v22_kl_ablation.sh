# =============================================================================
# v22_kl_ablation — no_farm yaml + KL=0.05 (诊断: KL 是否是峰值瓶颈?)
# =============================================================================
# 动机:
#   v20 引入了 4 个稳定性补丁 (相对 v19_no_farm):
#     1. ENTROPY_COEFF:   0.001 -> 0.002   (提高探索熵)
#     2. KL_LOSS_COEF:    0.05  -> 0.10    (更强的 KL 约束)
#     3. CRITIC_WARMUP:   30    -> 60      (更长 critic 预热)
#     4. CLIPRANGE_VALUE: 0.8   -> 0.5     (更紧的 value clip)
#
#   效果对比:
#     v19_no_farm (原参数, KL=0.05):   tsucc_b4 峰值 = 1.000 @ step40, 但 step180 崩塌
#     v21_no_farm_only_stab (4个补丁): tsucc_b4 峰值 = 0.895 @ step40-60, 稳定
#
#   峰值从 1.000 → 0.895 的下降由 4 个补丁共同造成, 但 KL_LOSS_COEF 最可能是主因:
#     - KL loss 直接约束 policy 不要偏离 reference model
#     - KL=0.10 (2× KL=0.05) 会将 policy 拉回 ref, 限制学到的最优解的幅度
#     - 而 ENTROPY/CLIPRANGE/WARMUP 主要影响训练稳定性, 不直接 cap 策略能力
#
#   本实验: 只改回 KL=0.05, 保留其他三个补丁. 这是一个受控消融:
#     - 若峰值恢复到 ≥0.947 而不崩塌 → KL=0.10 是瓶颈, v23 可试 KL=0.07+cosine
#     - 若崩塌回来 → KL=0.10 是稳定性必需, 其余补丁不足以单独撑住
#     - 若峰值仍 0.895 → 其他三个补丁才是瓶颈, 需要再做单独消融
#
# yaml: env_config_v19_no_farm.yaml  (= v21_no_farm_only_stab)
# sh:   v21_no_farm_only_stab 基础上只修改 KL_LOSS_COEF: 0.10 -> 0.05
#
# 与 v21_no_farm_only_stab 的唯一差异:
#   ★ KL_LOSS_COEF:  0.10 -> 0.05  (恢复到 v19_no_farm 原始值)
#
# 期望:
#   - 如果 KL=0.05 有效: 峰值 tsucc_b4 ≥ 0.947 (可能趋近 1.000)
#   - 如果 3 个剩余补丁足够防崩: step80-100 后不剧烈下跌
#   - 结果提供 v23 的 KL 最优点估计 (0.05 < best_KL < 0.10)
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v22_kl_ablation.sh \
    > v22_kl_ablation.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v22_kl_ablation"
ENV_CONFIG="env_config_v19_no_farm.yaml"   # 与 v21_no_farm_only_stab 相同
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.002"          # v20 stability patch: 保留
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"            # ★ v21_no_farm_stab 是 0.10; 恢复到 v19_no_farm 原始值
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
CRITIC_WARMUP=60               # v20 stability patch: 保留
CLIPRANGE_VALUE="0.5"          # v20 stability patch: 保留

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data ===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=20
TOTAL_STEPS=2000               # 长训练, 观察崩塌是否发生及何时发生
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
