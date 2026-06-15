# =============================================================================
# c3_lr5e7_cambrian — Cambrian-S 7B + GroupAdv (N=4) + LR 降低诊断
# =============================================================================
# 动机（相对 c1 的改动）:
#   c1 用 LR=1e-6 完全没有收益（step 70 时 turn_left=43.2%，模型没有在学习）。
#   诊断假说：LR=1e-6 对 7B 模型过大，导致策略网络震荡而无法稳定改进。
#   c3 降低 LR 至 5e-7（与 v28 Qwen3B 相同），同时延长 critic warmup（120步）
#   为 7B 模型提供更稳健的 critic baseline。
#
# 相对 c1 的差异:
#   ★ ACTOR_LR: 1e-6 → 5e-7
#   ★ CRITIC_WARMUP: 60 → 120
#
# 科学问题:
#   LR=1e-6 是否是 Cambrian turn_left 固化、无法学习的根本原因？
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c3_lr5e7_cambrian.sh \
    > c3_lr5e7_cambrian.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="c3_lr5e7_cambrian"
ENV_CONFIG="env_config_v24_100scenes.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"
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
CRITIC_WARMUP=120            # ★ 延长 warmup（c1=60），7B 需要更久稳定 critic
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"              # ★ 降低（c1=1e-6），诊断 LR 是否是不学习的原因

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
