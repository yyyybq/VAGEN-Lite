# =============================================================================
# v19_no_farm_dual — Triple winner merger
#   = v19_thr70_andgate (AND-gate, thr=0.70)
#   + v19_no_farm        (format=0, step_penalty=-0.02)
#   + v19_dual           (progress_mode=dual, α_pos=0.3, α_ori=0.7)
# yaml: env_config_v19_no_farm_dual.yaml
# sh   部分完全沿用 v19_thr70_andgate (无 sh 级改动, 纯 yaml 合并).
#
# 与 v20_winner 的区别:
#   v19_no_farm_dual = v19_thr70_andgate sh + 三路 yaml 合并.
#   v20_winner       = 同 yaml, 但 sh 加稳定性补丁 (KL_LOSS_COEF↑, CRITIC_WARMUP↑,
#                      CLIPRANGE_VALUE↓), 用于诊断 v19_thr70_andgate 的后期崩塌.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v19_no_farm_dual.sh \
    > v19_no_farm_dual.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v19_no_farm_dual"
ENV_CONFIG="env_config_v19_no_farm_dual.yaml"
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
