# =============================================================================
# v23_groupadv_3scenes_7types — GroupAdv (N=4) + 3-scene + 7 task types + OOD val
# =============================================================================
# 动机:
#   v23_groupadv_3scenes (156 steps, step 150 checkpoint) 使用的是 9 种任务类型.
#   其中 screen_occupancy (30%) / size_distance_invariance (5.6%) 在 OOD val 中
#   完全缺失, 导致训练分布与验证分布存在显著不匹配.
#
#   本实验去除这两种任务, 使训练集与 OOD val 任务类型完全对齐:
#     absolute_positioning, delta_control, projective_relations, equidistance,
#     occlusion_alignment, fov_inclusion, centering
#
# 与 v23_groupadv_3scenes 的差异:
#   ★ ENV_CONFIG: env_config_v22_3scenes.yaml -> env_config_v23_3scenes_7types.yaml
#   ★ train_size: 752 -> 497  (去除 screen_occupancy + size_distance_invariance)
#   ★ EXPERIMENT_NAME: v23_groupadv_3scenes -> v23_groupadv_3scenes_7types
#   (所有超参数完全保持一致)
#
# 期望:
#   - 任务类型对齐后 OOD val 指标更稳定
#   - GroupAdv + 3场景 的稳定性优势继续保持
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v23_groupadv_3scenes_7types.sh \
    > v23_groupadv_3scenes_7types.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v23_groupadv_3scenes_7types"
ENV_CONFIG="env_config_v22_3scenes.yaml"     # ★ 已更新为 7-type filtered jsonl
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

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=12             # 12 * 4 = 48 轨迹/step
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=80
TEST_FREQ=30
TOTAL_STEPS=1000
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
