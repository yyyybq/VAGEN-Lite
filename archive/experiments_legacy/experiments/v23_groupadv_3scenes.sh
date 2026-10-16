# =============================================================================
# v23_groupadv_3scenes — GroupAdv (N=4) + 3-scene curriculum + OOD val
# =============================================================================
# 动机:
#   v22_3scenes 的 step 30→120 显示出独特的"b4↓ 而 OOD m4↑"信号:
#     - b4: 0.842 → 0.737 (in-domain best 在下降)
#     - OOD b4: 0.895 → 0.737 (OOD 也在退化, 与 v22_cosine_no_farm 类似)
#     - OOD m4: 0.368 → 0.460 (但平均一致性在提升, 独有)
#
#   假设: "GroupAdv 收紧 best/mean gap" + "3 场景训练扩大几何先验" 可叠加,
#         有望同时:
#           (a) 抬高 OOD m4 (GroupAdv 直接优化 mean)
#           (b) 阻止 OOD b4 单调退化 (多场景训练 + N=4 多样性双重约束)
#           (c) 让 in-domain b4 不再单调下降
#
#   做法: 把 v21_groupadv 的 N=4 套到 3 场景训练上.
#
# 与 v22_3scenes 的差异:
#   ★ N_TRAJECTORY: 1 -> 4
#   ★ TRAIN_BATCH_SIZE: 24 -> 12
#   ★ PPO_MINI_BATCH_SIZE: 12 -> 8
#
# 与 v23_groupadv_no_farm 的差异:
#   ★ ENV_CONFIG: env_config_v19_no_farm.yaml -> env_config_v22_3scenes.yaml
#   ★ TEST_FREQ: 20 -> 30 (val 现在 38 prompts × 4 rollouts = 152 records)
#   ★ TOTAL_STEPS: 2000 -> 1000
#
# 期望:
#   - OOD b4 在 s60+ 仍保持 ≥ 0.80 (vs v22_3scenes 在 s90 已跌到 0.842)
#   - OOD m4 持续上升突破 0.50 (vs v22_3scenes 至 s120 = 0.460)
#   - in-domain b4 不再单调下降, 至少稳定在 0.842
#   - 若 (a)(b) 同时达成, 则 v23_groupadv_3scenes 成为下游 SFT 候选权重源
#
# 注意:
#   每 step 涉及 N=4 rollout × 3 场景 ≈ 4-6 次 set_scene 切换,
#   PLY 重载会进一步拖慢 step time. 若每步 > 90s 考虑回退.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v23_groupadv_3scenes.sh \
    > v23_groupadv_3scenes.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v23_groupadv_3scenes"
ENV_CONFIG="env_config_v22_3scenes.yaml"     # ★ 3 场景合并 jsonl
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
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
TRAIN_BATCH_SIZE=12             # ★ 24 -> 12
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8           # ★ 12 -> 8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=80                    # ★ 改为 80
TEST_FREQ=30                    # ★ val 现在 38 prompts, 慢一些
TOTAL_STEPS=1000                # ★ 短期实验, 3 epoch
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
