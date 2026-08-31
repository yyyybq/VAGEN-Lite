# =============================================================================
# u1_fwdfirst_rewscale_smoke — SenseNova-U1 8B + Active Spatial 1-step smoke
# =============================================================================
# Goal:
#   Wire SenseNova-U1 into the same Active Spatial PPO path used by Cambrian-S:
#   per-turn image observation -> model emits <think>prediction</think><action>...</action>.
#
# This script is intentionally tiny.  Use it to validate:
#   1. tokenizer/config load
#   2. U1 processor wrapper expands <image> into <img><IMG_CONTEXT>*N</img>
#   3. actor/ref logprob forward runs through vagen.models.sensenova_u1_register
#   4. critic/value model initializes
#   5. rollout server can serve the U1 checkpoint, or clearly fails at vLLM support
#
# Launch:
#   cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite
#   export SENSENOVA_U1_SRC=/mnt/umm/users/yinbaiqiao/SenseNova-U1/src
#   export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH}"
#   bash examples/train/active_spatial/run_experiment.sh \
#     examples/train/active_spatial/experiments/u1_fwdfirst_rewscale_smoke.sh

EXPERIMENT_NAME="u1_fwdfirst_rewscale_smoke"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst_rewscale.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4
USE_GPU_HOLDER=false

RESUME_MODE="disable"

# === Model ===
MODEL_PATH="${SENSENOVA_U1_MODEL_PATH:-/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT}"
export SENSENOVA_U1_SRC="${SENSENOVA_U1_SRC:-/mnt/umm/users/yinbaiqiao/SenseNova-U1/src}"
export PYTHONPATH="${SENSENOVA_U1_SRC}:${PYTHONPATH}"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# === Conservative rollout / memory ===
TP_SIZE="${U1_TP_SIZE:-4}"
GPU_MEM_UTIL="${U1_GPU_MEM_UTIL:-0.25}"
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
MAX_RESPONSE_LENGTH="256"
MAX_PROMPT_LENGTH="3072"
N_TRAJECTORY="1"
TRAIN_BATCH_SIZE="1"
VAL_BATCH_SIZE="1"
PPO_MINI_BATCH_SIZE="1"
MINI_BATCH_SIZE="1"

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
    data.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=vagen.models.sensenova_u1_register \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.sensenova_u1_register \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False"
