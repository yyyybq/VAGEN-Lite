# =============================================================================
# c1_groupadv_100scenes — Cambrian-S 7B + GroupAdv (N=4) + 94 场景
# =============================================================================
# 动机:
#   基于 v25_groupadv_100scenes_klhi 的参数体系，切换骨干模型至 Cambrian-S (7B):
#     - 模型: /scratch/by2593/hf_cache/cambrian-s-7b  (CambrianQwenForCausalLM)
#     - 视觉编码: SigLIP-so400m-patch14-384，每张图 756 tokens
#     - 参数量 7B → TP=2（单节点 4 GPU），GPU_MEM_UTIL 适当提高
#     - TRAIN_BATCH_SIZE/PPO_MINI_BATCH_SIZE 缩小以适应显存
#   目标：验证 Cambrian-S 在 Active-Spatial navigation 任务上的基线性能
#
# 相对 v25_groupadv_100scenes_klhi 的差异:
#   ★ MODEL_PATH: Qwen2.5-VL-3B → /scratch/by2593/hf_cache/cambrian-s-7b
#   ★ TP_SIZE: 4 → 2  （7B 模型，TP=2 在 4 张 GPU 上各占 2 张）
#   ★ GPU_MEM_UTIL: 0.4 → 0.65
#   ★ TRAIN_BATCH_SIZE: 12 → 8
#   ★ PPO_MINI_BATCH_SIZE / MINI_BATCH_SIZE: 8 → 4
#   ★ VAL_BATCH_SIZE: 8 → 4
#   ★ EXTRA_OVERRIDES: 注册 vagen.models.cambrian_register + trust_remote_code
#   KL/entropy/LR/轨迹参数与 v25_klhi 相同
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c2_nfp_groupadv.sh \
    > c2_nfp_groupadv.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="c2_nfp_groupadv"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # 94 scenes, 28662 tasks（与 v24/v25 共用）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=2                    # 7B 模型，TP=2，每对 GPU 分别承载一个 shard
GPU_MEM_UTIL=0.20            # 7B 模型需要更多显存; reduced to 0.20: FSDP _padded_unsharded_flat_param buffers (~48 GiB for 3 models) leave only ~31.2 GiB free; need GPU_MEM_UTIL*140 <= 31.2 GiB → max 0.223

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
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=8           # 7B 模型显存更大，缩小 batch
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
    # NOTE: use_remove_padding=False is required for NFP training.
    # With rmpad=True, input_ids is flattened from (bs, seq_len) to (1, total_nnz),
    # stripping all padding tokens and shifting positions.  The nfp_loss_mask is
    # computed in the padded (bs, seq_len) coordinate system, so position indices
    # become invalid under rmpad and the wrong tokens get masked.

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
