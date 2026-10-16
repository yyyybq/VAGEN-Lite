# =============================================================================
# v23_groupadv_no_farm — GroupAdv (N=4) + no_farm yaml + OOD val
# =============================================================================
# 动机:
#   截至 v21/v22 分析:
#     - v21_groupadv (dual yaml + N=4) 是唯一显示 m4 持续上升的实验
#         (s60→s100: 0.474 → 0.513 → 0.579)
#         但 b4 仅停在 0.842-0.895, 未触及 0.9474.
#     - v21_no_farm_only_stab (no_farm + N=1) 触及 0.9474 峰值
#         但 m4 后续下滑 (0.632 → 0.487), best/mean gap 大.
#
#   假设: "GroupAdv 提升 m4" + "no_farm 提升 b4 峰值" 两个独立信号可叠加.
#   做法: 把 v21_groupadv 的 N=4 + 缩 batch 与 no_farm yaml 组合, 同时启 OOD val
#         以观察泛化是否被 N=4 改善 (4 条 rollout 的多样性可能减缓单场景过拟合).
#
# 与 v21_groupadv 的差异:
#   ★ ENV_CONFIG: env_config_v19_no_farm_dual.yaml -> env_config_v19_no_farm.yaml
#   + OOD val (Plan A)
#
# 与 v21_no_farm_only_stab 的差异:
#   ★ N_TRAJECTORY: 1 -> 4
#   ★ TRAIN_BATCH_SIZE: 24 -> 12
#   ★ PPO_MINI_BATCH_SIZE: 12 -> 8
#   + OOD val
#
# 期望:
#   - b4 触及 0.9474 (no_farm 的特征)
#   - m4 在 s80+ 突破 0.60 并持续上升 (GroupAdv 的特征)
#   - OOD b4 不像 v22_cosine_no_farm 那样从 0.895 单调跌到 0.632
#     (期望 N=4 的多样性能缓解 in-domain 专化)
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v23_groupadv_no_farm.sh \
    > v23_groupadv_no_farm.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v23_groupadv_no_farm"
ENV_CONFIG="env_config_v19_no_farm.yaml"      # ★ no_farm (vs v21_groupadv 的 dual)
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (= v21_no_farm_only_stab / v21_groupadv) ===
ENTROPY_COEFF="0.002"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"
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
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (★ GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4                  # ★ 1 -> 4
TRAIN_BATCH_SIZE=12             # ★ 24 -> 12 (12*4=48 轨迹/step)
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8           # ★ 12 -> 8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
