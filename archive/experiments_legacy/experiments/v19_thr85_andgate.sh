# =============================================================================
# v19_thr85_andgate — SFT 级 AND-gate, pos_thr=0.85, ori_thr=0.80
# =============================================================================
# yaml 改动 (相对 v18_andgate):
#   success_position_threshold:    0.45 -> 0.85
#   success_orientation_threshold: 0.45 -> 0.80
#   near_success_threshold:        0.35 -> 0.70  (sigmoid)
#   near_success_bonus:            0.2  -> 0.5
#   near_success_mode:             constant -> sigmoid (steepness=10)
#   potential_field_reward_scale:  0.5  -> 0.8
#
# 注: 此 yaml 信号最稀, 建议先做 SFT cold-start, 否则 from-scratch RL 起步慢.
# 此 sh 仍 from-scratch — 当 baseline 比对; SFT-init 由另一个 sh 启动.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v19_thr85_andgate.sh \
    > v19_thr85_andgate.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v19_thr85_andgate"
ENV_CONFIG="env_config_v19_thr85_andgate.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
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
CRITIC_WARMUP=30
CLIPRANGE_VALUE="0.8"

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
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
