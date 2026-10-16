# v17 ablation B: reward-shape fix only (legacy action space)
# vs v17 (full):  reward redesign kept; action space reverted to v16 legacy
# Use this to answer: "Did fixing the reward magnitude alone solve it?"
# Note: invalid_action_count is expected to stay high (model still hallucinates move_left).
#
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v17_reward_only.sh \
    > v17_reward_only.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v17_reward_only"
ENV_CONFIG="env_config_v17_reward_only.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# Actor: legacy action space -> can use lower entropy (no new space to explore)
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.04"
TEMPERATURE="0.7"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

CRITIC_LR="2e-5"
CRITIC_WARMUP=30
CLIPRANGE_VALUE="0.8"

GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
