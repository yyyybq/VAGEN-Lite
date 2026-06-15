# v17 ablation C: "guaranteed early win" curriculum
# success_score_threshold=0.20 -> any reasonable approach triggers success.
# If this still gives 0 traj_success, the issue is NOT the env — it's the
# RL stack (critic, GAE, KL, PPO ratio clipping) and we need to pivot.
#
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v17_easy.sh \
    > v17_easy.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v17_easy"
ENV_CONFIG="env_config_v17_easy.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.02"
TEMPERATURE="0.8"
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
