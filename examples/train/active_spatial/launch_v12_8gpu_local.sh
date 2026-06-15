#!/bin/bash
# =============================================================================
# Active Spatial — v12 config, 8 GPU, local rendering
#
# GPU 分配:
#   GPU 0-6 (7张) — 训练 (actor/critic/ref FSDP + SGLang rollout TP=4)
#   GPU 7          — Active Spatial 环境服务器 (含本地 GS 渲染)
#
# 用法:
#   cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite
#   bash examples/train/active_spatial/launch_v12_8gpu_local.sh
# =============================================================================

set -e
set -x

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
BASEDIR=$(cd "$(dirname "$0")/../../.." && pwd)   # → VAGEN-Lite/
SCRIPTDIR="$BASEDIR/examples/train/active_spatial"
PROJECT_NAME="vagen_active_spatial"
EXPERIMENT_NAME="v12_max_turns15_8gpu_local"

EXPERIMENT_DIR="${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}"
SAVE_CHECKPOINT_DIR="${EXPERIMENT_DIR}/verl_checkpoints"
ENV_SERVER_LOG="${EXPERIMENT_DIR}/env_server.log"

mkdir -p "${EXPERIMENT_DIR}"

# ---------------------------------------------------------------------------
# v12 超参 (与 experiments/v12_max_turns50.sh 对齐)
# ---------------------------------------------------------------------------
ACTOR_LR="1e-6"
CRITIC_LR="2e-5"
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
GRAD_CLIP="0.5"
CRITIC_WARMUP=40
CLIPRANGE_VALUE="0.8"

MAX_TURNS=15
MAX_TRAJECTORY_LENGTH=26000
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=2048

N_TRAJECTORY=2
TRAIN_BATCH_SIZE=14       # 14×2=28 总轨迹, 28%7=0 ✓
VAL_BATCH_SIZE=4
PPO_MINI_BATCH_SIZE=14
MINI_BATCH_SIZE=8

SAVE_FREQ=100
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="True"

ADV_ESTIMATOR="no_concat_gae"   # VAGEN-Lite 用 no_concat_gae
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
REF_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
DATASET_TRAIN="${SCRIPTDIR}/train_active_spatial_local.yaml"
DATASET_VAL="${SCRIPTDIR}/val_active_spatial_local.yaml"

# Agent loop (no-concat)
AGENT_LOOP_CONFIG="${BASEDIR}/vagen/configs/agent_no_concat_active_spatial.yaml"

# ---------------------------------------------------------------------------
# 启动 Active Spatial 环境服务器 (GPU 7, 本地渲染)
# ---------------------------------------------------------------------------
echo "[launch] Starting Active Spatial env server on GPU 7..."

CUDA_VISIBLE_DEVICES=7 \
    "${BASEDIR}/../../../miniconda3/envs/vagen/bin/python" \
    -m vagen.envs.active_spatial.serve \
    --devices='[0]' \
    --port=8001 \
    --thread_pool_size=64 \
    > "${ENV_SERVER_LOG}" 2>&1 &

ENV_SERVER_PID=$!
echo "[launch] Env server PID: ${ENV_SERVER_PID}"

# 等待环境服务器就绪 (最多 120 秒)
echo "[launch] Waiting for env server at http://localhost:8001 ..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        echo "[launch] Env server is ready (${i}s)"
        break
    fi
    if ! kill -0 "${ENV_SERVER_PID}" 2>/dev/null; then
        echo "[ERROR] Env server process died. Check ${ENV_SERVER_LOG}"
        cat "${ENV_SERVER_LOG}"
        exit 1
    fi
    sleep 1
done

# 清理函数: 训练结束时关闭环境服务器
cleanup() {
    echo "[launch] Shutting down env server (PID ${ENV_SERVER_PID})..."
    kill "${ENV_SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 启动训练 (GPU 0-6, 7张卡)
# ---------------------------------------------------------------------------
echo "[launch] Starting PPO training on GPUs 0-6..."

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
PYTHONUNBUFFERED=1 \
    "${BASEDIR}/../../../miniconda3/envs/vagen/bin/python" \
    -m vagen.main_ppo \
    --config-path="${BASEDIR}/vagen/configs" \
    --config-name='vagen_multiturn' \
    \
    data.train_files="${DATASET_TRAIN}" \
    data.val_files="${DATASET_VAL}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_TRAJECTORY_LENGTH} \
    \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${N_TRAJECTORY} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.top_p=${TOP_P} \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.optim.lr=${CRITIC_LR} \
    critic.model.path="${REF_MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.cliprange_value=${CLIPRANGE_VALUE} \
    \
    trainer.concat_multi_turn=False \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
    trainer.validation_data_dir=${EXPERIMENT_DIR}/validation \
    trainer.rollout_data_dir=${EXPERIMENT_DIR}/rollout_data \
    trainer.log_val_generations=8 \
    \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"
