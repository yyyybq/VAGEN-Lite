# =============================================================================
# v20_sft_no_autoterm — SFT cold-start × v19_thr85_no_autoterm yaml
# =============================================================================
# 背景:
#   v19_thr85_no_autoterm (from-scratch, autoterm OFF + thr=0.85/0.80 AND-gate)
#   已证实信号过稀, val best@4 峰值仅 0.036, 出现 entropy=0.73 模式崩溃.
#   本实验同 yaml + SFT cold-start, 让 agent 从 SFT 习得的 done 行为出发,
#   验证 "SFT cold-start 是否解锁 agent-自决-done 路线" 假设.
#
# 前置依赖: 同 v20_sft_thr85_andgate (SFT ckpt 必须存在).
#
# 启动:
: <<'RUN'
SFT_CKPT="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/checkpoints/sft_qwen25vl_3b_5k"
ls -d "$SFT_CKPT"/checkpoint-* 2>/dev/null | tail -1
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v20_sft_no_autoterm.sh \
    > v20_sft_no_autoterm.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v20_sft_no_autoterm"
ENV_CONFIG="env_config_v19_thr85_no_autoterm.yaml"  # 复用 v19_thr85_no_autoterm yaml
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

# ★ SFT cold-start
MODEL_PATH="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/checkpoints/sft_qwen25vl_3b_5k"

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
