# =============================================================================
# v29_grpo_resume — v29_grpo 续训（在 7 卡 H200 上恢复，沿用 5 张 GPU）
# =============================================================================
# 背景:
#   v29_grpo 因 job 到期停止，最新 step=77，最新保存 checkpoint=global_step_50。
#   由于原始 checkpoint 以 world_size=4 (4 分片) 保存，
#   FSDP resume 必须使用相同 world_size，因此保持 NUM_TRAIN_GPUS=4 不变。
#   7 卡节点中 GPU 0-3 训练，GPU 4 渲染，GPU 5-6 空闲。
#
# 与 v29_grpo.sh 的唯一差异:
#   ★ RESUME_MODE: disable → auto
#
# 启动 (7卡 H200 节点，使用 GPU 0-4):
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v29_grpo_resume.sh \
    > v29_grpo_resume.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v29_grpo"          # ★ 保持相同实验名（续写同一个 exp 目录）
ENV_CONFIG="env_config_v24_100scenes_lm.yaml"
NUM_TRAIN_GPUS=4                    # 保持 4（与保存 checkpoint 的 world_size 一致）
RENDERING_GPU=4

RESUME_MODE="auto"                  # ★ disable → auto（从 global_step_50 恢复）

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
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
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=12
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=50
TEST_FREQ=50
TOTAL_STEPS=1000
VAL_BEFORE_TRAIN="False"    # resume 时不重复 step0 val

# === Algorithm: GRPO ===
ADV_ESTIMATOR="grpo"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === cosine LR decay ===
export EXTRA_OVERRIDES="\
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
