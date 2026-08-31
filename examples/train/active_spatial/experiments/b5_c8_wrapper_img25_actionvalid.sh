# =============================================================================
# B5: c8 recipe + wrapper limits + stronger action-valid
# Unique vs c8_fwdfirst_rewscale_lfp_server_v19:
#   1) ENV_CONFIG -> env_config_b5_c8_actionvalid.yaml (format/invalid penalties)
#   2) EXTRA_OVERRIDES += limit_images=25, max_model_len=16384
# Eval-side: export VAGEN_CAMBRIAN_LIMIT_IMAGES / VAGEN_CAMBRIAN_MAX_MODEL_LEN
# =============================================================================
EXPERIMENT_NAME="b5_c8_wrapper_img25_actionvalid"
export VAGEN_VISION_SANITY_DIR="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/vision_sanity"
export VAGEN_VISION_SANITY_MAX=16
ENV_CONFIG="env_config_b5_c8_actionvalid.yaml"
NUM_TRAIN_GPUS=8
RENDER_MODE="remote"
RENDER_HOST="10.119.30.223"
RENDER_PORT="8767"
RENDER_PROTOCOL="http"
RENDERING_GPU=0
RESUME_MODE="auto"

ENTROPY_COEFF="0.0"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
TEMPERATURE="0.6"
TOP_P="0.90"
TP_SIZE=2
GPU_MEM_UTIL=0.35
VAL_TEMPERATURE="0.0"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="False"
VAL_N="1"
CRITIC_LR="2e-6"
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8
SAVE_FREQ=50
TEST_FREQ=50
TOTAL_STEPS=1000
VAL_BEFORE_TRAIN="True"
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
MODEL_PATH="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP"

EXTRA_OVERRIDES="\
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.cambrian_register \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False \
    actor_rollout_ref.actor.nfp_loss_coef=1.0 \
    +actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.logprob_temperature=1.0 \
    algorithm.rollout_correction.bypass_mode=True \
    +actor_rollout_ref.rollout.limit_images=25 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    hydra.run.dir=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/${EXPERIMENT_NAME}/hydra_run \
  +ray_kwargs.ray_init.include_dashboard=False"

export ID_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export ID_VAL_N_ENVS=19
export OOD_VAL_JSONL="/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
export OOD_VAL_N_ENVS=25
