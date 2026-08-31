# =============================================================================
# u1_fwdfirst_rewscale_i2i_smoke — SenseNova-U1 Plan B phase-1 (open-loop FM aux)
# =============================================================================
# Goal:
#   Active Spatial PPO smoke with teacher-forced next-frame FM aux:
#   rollout text-only <action>...</action>; train appends <image> gen span;
#   obs always from env (open-loop). image_gen_loss_coef=0.1.
#
# Full 8-GPU node: 7 train + 1 local render (GPU7). Parallel with other jobs is OK.
#   bash examples/train/active_spatial/run_experiment.sh \
#     examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke.sh
#
# Launch:
#   cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
#   export SENSENOVA_U1_SRC=/mnt/umm/users/yinbaiqiao/SenseNova-U1/src
#   export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH}"
#   bash examples/train/active_spatial/run_experiment.sh \
#     examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_i2i_smoke.sh

EXPERIMENT_NAME="u1_fwdfirst_rewscale_i2i_smoke"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst_rewscale_smoke_server.yaml"
NUM_TRAIN_GPUS="${U1_NUM_TRAIN_GPUS:-7}"
RENDERING_GPU="${U1_RENDERING_GPU:-7}"
USE_GPU_HOLDER=false

# Optional remote GS render (U1_RENDER_MODE=remote)
RENDER_MODE="${U1_RENDER_MODE:-local}"
RENDER_PROTOCOL="${U1_RENDER_PROTOCOL:-http}"
RENDER_HOST="${U1_RENDER_HOST:-}"
RENDER_PORT="${U1_RENDER_PORT:-8767}"

RESUME_MODE="disable"

# === Model ===
MODEL_PATH="${SENSENOVA_U1_MODEL_PATH:-/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT}"
export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# === Conservative rollout / memory ===
TP_SIZE="${U1_TP_SIZE:-1}"
GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.35}"
TEMPERATURE="0.8"
TOP_P="0.92"
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="1"

# === Optim ===
ACTOR_LR="5e-7"
CRITIC_LR="2e-5"
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
GRAD_CLIP="0.3"
CRITIC_WARMUP="0"
CLIPRANGE_VALUE="0.5"

# === Episode / batch: smoke only ===
MAX_TURNS="2"
MAX_TRAJECTORY_LENGTH="12000"
MAX_RESPONSE_LENGTH="512"
MAX_PROMPT_LENGTH="2048"
N_TRAJECTORY="1"
TRAIN_BATCH_SIZE="${U1_TRAIN_BATCH_SIZE:-7}"
VAL_BATCH_SIZE="1"
PPO_MINI_BATCH_SIZE="${U1_PPO_MINI_BATCH_SIZE:-7}"
MINI_BATCH_SIZE="${U1_MINI_BATCH_SIZE:-7}"

# === Schedule ===
SAVE_FREQ="1"
TEST_FREQ="1"
TOTAL_STEPS="1"
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

EXTRA_OVERRIDES="\
    data.apply_chat_template_kwargs.enable_thinking=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=vagen.models.u1_neo_compat \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.u1_neo_compat \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False \
    actor_rollout_ref.actor.image_gen_loss_coef=0.1 \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra] \
    actor_rollout_ref.actor.nfp_loss_coef=0.1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_TRAIN_GPUS} \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,NEOVisionModel] \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,NEOVisionModel] \
    +critic.model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,NEOVisionModel]"
