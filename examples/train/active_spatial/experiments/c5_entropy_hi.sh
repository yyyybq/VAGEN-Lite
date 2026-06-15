# =============================================================================
# c5_entropy_hi — Cambrian-S 7B + 高熵正则（直接对抗 turn_left 坍缩）
# =============================================================================
# 动机（基于 c1 rollout 分析）:
#
#   c1 最核心的失败表现: turn_left=43.2%，模型行为高度集中在单一动作，
#   action distribution entropy 极低 → 这是 entropy collapse 的典型症状。
#
#   ENTROPY_COEFF=0.005 (c1) 的熵正则太弱，无法阻止 Cambrian 的 turn_left 固化。
#   c5 将 ENTROPY_COEFF 提高 4x 至 0.02，直接在 PPO 目标中惩罚低熵分布，
#   强迫模型在 policy 更新时保持动作多样性。
#
# 相对 c3（基线）的差异:
#   ★ ENTROPY_COEFF: 0.005 → 0.02   (直接对抗 turn_left 坍缩)
#   其他参数与 c3 一致 (ACTOR_LR=5e-7, 标准 free_think prompt)
#
# 科学假说:
#   turn_left 43% 的主导行为是 PPO entropy 坍缩导致的 action distribution 退化。
#   若 c5 在 step 50-100 时 turn_left 占比 < 30% 且 move_forward > 30%，
#   说明熵正则是 Cambrian 的关键缺失。
#   若 c5 行为与 c3 相同，说明 turn_left 是 Cambrian 的固有先验（需要 SFT 冷启动）。
#
# 对比矩阵（相对 c1 基线）:
#   c3: LR 5e-7  (诊断 LR 影响)
#   c4: fwd-first prompt (诊断 prompt 策略影响)
#   c5: entropy 0.02 (诊断熵正则影响) ← 本实验
#   c6: no_think format (诊断 think-format 合规障碍)
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c5_entropy_hi.sh \
    > c5_entropy_hi.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="c5_entropy_hi"
ENV_CONFIG="env_config_v24_100scenes.yaml"   # 标准 free_think prompt（与 c3 相同）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.02"         # ★ 0.005 → 0.02 (4x，直接对抗 turn_left entropy 坍缩)
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=2
GPU_MEM_UTIL=0.20

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
ACTOR_LR="5e-7"              # 与 c3/c4 一致

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=4

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=4
MINI_BATCH_SIZE=4

# === Trainer ===
SAVE_FREQ=80
TEST_FREQ=50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Model: Cambrian-S 7B ===
MODEL_PATH="/scratch/by2593/hf_cache/cambrian-s-7b"

EXTRA_OVERRIDES="\
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.cambrian_register \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
