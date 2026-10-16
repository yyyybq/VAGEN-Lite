# =============================================================================
# Phase 4B: protocol-only pre/post PPO, fixed 32 optimizer steps.
#
# This is intentionally isomorphic to u1_fwdfirst_rewscale_i2i_short.sh except
# for the isolated experiment path, fixed checkpoint cadence, remote renderer,
# and U1_FM_BACKPROP=0 (set by the launch/entry script).  It adds no SFT, CE,
# protocol auxiliary objective, or other loss.
# =============================================================================

EXPERIMENT_NAME="protocol_only_prepost"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst_rewscale_u1_server.yaml"

NUM_TRAIN_GPUS="${U1_NUM_TRAIN_GPUS:-8}"
RENDER_MODE="remote"
RENDER_PROTOCOL="http"
RENDER_HOST="${U1_RENDER_HOST:-10.119.27.217}"
RENDER_PORT="${U1_RENDER_PORT:-8768}"
RENDERING_GPU=0
USE_GPU_HOLDER=false

# This controlled run must never attach to a historical run.
RESUME_MODE="disable"

MODEL_PATH="${SENSENOVA_U1_MODEL_PATH:-/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT}"
export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export U1_FM_BACKPROP=0
export U1_FM_USE_UND_KV=0

TP_SIZE="${U1_TP_SIZE:-1}"
GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.28}"
TEMPERATURE="0.8"
TOP_P="0.92"
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="1"

ACTOR_LR="5e-7"
CRITIC_LR="2e-5"
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
GRAD_CLIP="0.3"
CRITIC_WARMUP="0"
CLIPRANGE_VALUE="0.5"

MAX_TURNS="2"
MAX_TRAJECTORY_LENGTH="12000"
MAX_RESPONSE_LENGTH="512"
MAX_PROMPT_LENGTH="2048"
N_TRAJECTORY="1"
TRAIN_BATCH_SIZE="${U1_TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="1"
PPO_MINI_BATCH_SIZE="${U1_PPO_MINI_BATCH_SIZE:-8}"
MINI_BATCH_SIZE="${U1_MINI_BATCH_SIZE:-8}"

# Fixed before submission; no adaptive extension is allowed.
SAVE_FREQ="8"
TEST_FREQ="10"
TOTAL_STEPS="32"
VAL_BEFORE_TRAIN="False"

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
    actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,optimizer,extra] \
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
    +critic.model.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3DecoderLayer,NEOVisionModel] \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.max_critic_ckpt_to_keep=5 \
    +trainer.ckpt_milestones=[8,16,24,32]"
