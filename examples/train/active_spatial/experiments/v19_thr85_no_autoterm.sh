# =============================================================================
# v19_thr85_no_autoterm — SFT 级门槛 + agent 自决 done
# =============================================================================
# yaml 改动 (相对 v19_no_autoterm):
#   success_position_threshold:    0.70 -> 0.85
#   success_orientation_threshold: 0.70 -> 0.80
#   near_success_*: constant 0.55/0.2 -> sigmoid 0.70/0.5 (steepness=10)
#   potential_field_reward_scale:  0.5  -> 0.8
#
# 奖励系统三处结构性修复 (only this yaml, 另 3 个 v19 已启动不动):
#   ☆ success_reward:  30.0  -> 10.0    # 治 cliff (#1)
#   ☆ format_reward:   0.02  -> 0.0     # 治 format-farm (#2)
#   ☆ step_penalty:    -0.01 -> -0.05   # 治磨蹭 (#3)
#
# 注: 此 yaml 信号最稀, 强烈建议配 SFT cold-start. 此 sh 仍 from-scratch 作 baseline.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v19_thr85_no_autoterm.sh \
    > v19_thr85_no_autoterm.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v19_thr85_no_autoterm"
ENV_CONFIG="env_config_v19_thr85_no_autoterm.yaml"
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
