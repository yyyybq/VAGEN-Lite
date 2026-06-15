# =============================================================================
# v20_sft_thr85_andgate — SFT cold-start × v19_thr85_andgate yaml
# =============================================================================
# 背景:
#   v19_thr85_andgate (from-scratch RL, SFT 级 thr=0.85/0.80 AND-gate) 已证实
#   val best@4 峰值 0.623 后崩塌, 信号太稀难以从零起.
#   本实验同 yaml + Qwen2.5-VL-3B SFT-on-active_spatial-5k checkpoint 作 actor 初始,
#   预期能稳定收敛到 ≥ 0.85, 验证 "SFT cold-start 解锁 SFT-级稀疏奖励" 假设.
#
# 前置依赖:
#   1. data_gen/active_spatial_sft 5k 数据生成完成 (parquet 已 ready).
#   2. run_sft_qwen25vl_3b.sh 已跑完, ckpt 落在
#      /scratch/by2593/project/Active_Spatial/VAGEN-Lite/checkpoints/sft_qwen25vl_3b_5k/
#   3. 该 ckpt 是 HF 格式 (LLaMA-Factory 默认), 可直接被 verl 加载.
#   4. ckpt 目录下应有 config.json + tokenizer + preprocessor_config.json + safetensors.
#      若 LLaMA-Factory 输出按 step 分片 (checkpoint-XXXX/), 把 SFT_CKPT 指向最终
#      step 的子目录, 或先合并到一个 final/ 目录.
#
# 启动 (确认 SFT ckpt 已存在后):
: <<'RUN'
SFT_CKPT="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/checkpoints/sft_qwen25vl_3b_5k"
ls -d "$SFT_CKPT"/checkpoint-* 2>/dev/null | tail -1   # 确认最终 ckpt 路径
# 如果 LLaMA-Factory 输出在 checkpoint-N 下, 先把 SFT_CKPT 改到该子目录再启动.
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v20_sft_thr85_andgate.sh \
    > v20_sft_thr85_andgate.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v20_sft_thr85_andgate"
ENV_CONFIG="env_config_v19_thr85_andgate.yaml"   # 复用 v19_thr85_andgate yaml
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

# ★ SFT cold-start: 把 actor 初始 model 从 Qwen2.5-VL-3B-Instruct 换成 SFT ckpt
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
