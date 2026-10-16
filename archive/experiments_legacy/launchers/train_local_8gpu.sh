#!/bin/bash
# =============================================================================
# Active Spatial — PPO training (8 GPU, local rendering, no-concat mode)
#
# 基于 v12 超参（max_turns=15, no_concat_gae），本地 GS 渲染（不需要单独的渲染服务器）
#
# 用法：
#   cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite
#   bash examples/train/active_spatial/train_local_8gpu.sh
#
# GPU 分配：
#   GPU 0-6: 训练 (trainer.n_gpus_per_node=7)
#   GPU 7:   env 服务进程 (serve.py, 本地 GS 渲染)
#   注：如果希望全 8 卡训练，可把 env serve 放到 CPU/另一机器
# =============================================================================

set -e
set -x

# ---------------------------------------------------------------------------
# Paths — 按需修改
# ---------------------------------------------------------------------------
BASEDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)   # VAGEN-Lite root
SCRIPTDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial/train_data_0267_840790.jsonl"
GS_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
REF_MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

EXPERIMENT_NAME="v12_8gpu_local_render"
PROJECT_NAME="vagen_active_spatial"
EXPERIMENT_DIR="${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_DIR}"

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------
export ACTIVE_SPATIAL_VAGEN_PATH="${BASEDIR}/../VAGEN"  # 旧 VAGEN 路径（env.py 依赖）

export VLLM_ATTENTION_BACKEND=XFORMERS
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export PYTHONHASHSEED=0
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0
export PYTHONUNBUFFERED=1

# 全部 8 卡均可见（serve 进程会抢占 GPU 7，训练进程用 0-6）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------------------------------------------------------------------------
# 启动 Active Spatial env serve（GPU 7，本地渲染）
# serve.py 会侦听 :8001，由训练的 RemoteEnv 客户端连接
# ---------------------------------------------------------------------------
ENV_PORT=8001
ENV_LOG="${EXPERIMENT_DIR}/env_serve.log"

echo "[INFO] Starting Active Spatial env server on GPU 7 (port ${ENV_PORT})..."
CUDA_VISIBLE_DEVICES=7 \
    ${PYTHON} -m vagen.envs.active_spatial.serve \
        --devices="[0]" \
        --port=${ENV_PORT} \
        --gs_root="${GS_ROOT}" \
        --render_backend=local \
    > "${ENV_LOG}" 2>&1 &
ENV_SERVE_PID=$!
echo "[INFO] env serve PID=${ENV_SERVE_PID}, log: ${ENV_LOG}"

# 等待 serve 就绪
echo "[INFO] Waiting for env server to become ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${ENV_PORT}/health" >/dev/null 2>&1; then
        echo "[INFO] Env server ready after ${i}s"
        break
    fi
    sleep 2
done

# ---------------------------------------------------------------------------
# 训练数据 YAML（inline 覆盖真实路径）
# ---------------------------------------------------------------------------
TRAIN_YAML="${SCRIPTDIR}/train_active_spatial_local.yaml"
VAL_YAML="${SCRIPTDIR}/val_active_spatial_local.yaml"

cat > "${TRAIN_YAML}" <<EOF
envs:
  - name: RemoteEnv
    n_envs: 1150
    data_source: active_spatial
    seed: [0, 1000]
    max_turns: 15
    response_length_per_turn: 512
    config:
      base_urls: "http://localhost:${ENV_PORT}"
      timeout: 120
      retries: 3
      jsonl_path: "${DATA_JSONL}"
      dataset_root: ""
      render_backend: "local"
      gs_root: "${GS_ROOT}"
      step_translation: 0.3
      step_rotation_deg: 30.0
      enable_potential_field: true
      success_score_threshold: 0.95
      enable_auto_termination: true
      format_reward: 0.2
      success_reward: 1.0
      enable_collision_detection: true
      collision_penalty: -0.15
      enable_step_penalty: true
      step_penalty: -0.01
      max_episode_steps: 50
      max_actions_per_step: 1
      action_sep: "|"
      prompt_format: "free_think"
EOF

cat > "${VAL_YAML}" <<EOF
envs:
  - name: RemoteEnv
    n_envs: 50
    data_source: active_spatial
    seed: [1000, 1050]
    max_turns: 15
    response_length_per_turn: 512
    config:
      base_urls: "http://localhost:${ENV_PORT}"
      timeout: 120
      retries: 3
      jsonl_path: "${DATA_JSONL}"
      dataset_root: ""
      render_backend: "local"
      gs_root: "${GS_ROOT}"
      step_translation: 0.3
      step_rotation_deg: 30.0
      enable_potential_field: true
      success_score_threshold: 0.95
      enable_auto_termination: true
      format_reward: 0.2
      success_reward: 1.0
      enable_collision_detection: true
      collision_penalty: -0.15
      enable_step_penalty: true
      step_penalty: -0.01
      max_episode_steps: 50
      max_actions_per_step: 1
      action_sep: "|"
      prompt_format: "free_think"
EOF

# ---------------------------------------------------------------------------
# 清理函数：训练结束后自动 kill env serve
# ---------------------------------------------------------------------------
cleanup() {
    echo "[INFO] Stopping env server (PID=${ENV_SERVE_PID})"
    kill "${ENV_SERVE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 训练（GPU 0-6，7 卡）
# 继承 v12 超参：no_concat_gae, max_turns=15, 单动作模式
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
${PYTHON} -m vagen.main_ppo \
    --config-path="${BASEDIR}/vagen/configs" \
    --config-name="vagen_multiturn" \
    data.train_files="${TRAIN_YAML}" \
    data.val_files="${VAL_YAML}" \
    data.train_batch_size=14 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    \
    algorithm.adv_estimator=no_concat_gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=14 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.015 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.008 \
    actor_rollout_ref.actor.clip_grad=0.5 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.92 \
    actor_rollout_ref.rollout.max_num_batched_tokens=12000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${BASEDIR}/vagen/configs/agent_no_concat.yaml" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.optim.lr=2e-5 \
    critic.model.path="${REF_MODEL_PATH}" \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.cliprange_value=0.8 \
    \
    trainer.concat_multi_turn=False \
    trainer.critic_warmup=40 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=20 \
    trainer.total_training_steps=2000 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${EXPERIMENT_DIR}/checkpoints" \
    trainer.validation_data_dir="${EXPERIMENT_DIR}/validation" \
    trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
    trainer.log_val_generations=8 \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"
