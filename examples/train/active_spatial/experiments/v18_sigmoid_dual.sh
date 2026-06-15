# =============================================================================
# v18_sigmoid_dual — dual progress × sigmoid near-bonus (激进组合)
# =============================================================================
# yaml 核心 (混合 v18_sigmoid2 + v18_dual2):
#   progress_mode             = "dual"            (来自 dual)
#   α_pos = 0.4, α_ori = 0.6                       (来自 dual2)
#   near_success_mode         = "sigmoid"          (来自 sigmoid2)
#   near_success_threshold    = 0.20
#   near_success_bonus        = 0.5, steepness=10
#   success_score_threshold   = 0.45
#   success_reward            = 20.0               (sigmoid 家族小阶跃)
#   success_require_both      = false              (这版不叠 AND-gate)
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_sigmoid_dual.sh \
    > v18_sigmoid_dual.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_sigmoid_dual"
ENV_CONFIG="env_config_v18_sigmoid_dual.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (沿用 v18_sigmoid2) ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling (F2) ===
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
