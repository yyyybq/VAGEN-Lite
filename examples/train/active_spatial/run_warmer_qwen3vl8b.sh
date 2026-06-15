#!/bin/bash
set -x

# =============================================================================
# 4-GPU Training + GPU Warmer - Qwen3-VL-8B-Instruct
# =============================================================================
# Based on run_warmer.sh, adapted for Qwen3-VL-8B-Instruct (8B model)
# Changes from 3B version:
#   - Model: Qwen/Qwen3-VL-8B-Instruct (both actor and critic)
#   - train_batch_size: 32 -> 16 (larger model needs less batch per step)
#   - ppo_mini_batch_size: 16 -> 8
#   - gpu_memory_utilization: 0.6 (with free_cache_engine+chunked_prefill)
#   - Learning rate: 1e-6 -> 5e-7 (lower for larger model)
#   - FSDP offload: enabled for both actor and critic
# =============================================================================

# Set CUDA devices - 4 GPUs for training (0-3), 1 GPU for rendering (4)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

# Ray configuration
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Disable verbose logging
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0

# Reserve GPU 4 for rendering
export RENDERING_GPU_ID=4

# Ensure ninja is in PATH for Ray workers
export PATH="/scratch/by2593/miniconda3/envs/vagen/bin:$PATH"

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME="active_spatial_ppo_qwen3vl8b_0321"

# 清理函数
cleanup() {
    echo "Cleaning up..."
    kill $GPU_HOLDER_PID 2>/dev/null || true
    # 杀掉所有 gpu_holder.py 子进程
    pkill -P $$ -f "gpu_holder.py" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Model: Qwen/Qwen3-VL-8B-Instruct"
echo "Training GPUs: 0,1,2,3 (4 GPUs)"
echo "Rendering GPU: 4"
echo "Strategy:"
echo "  - GPU 4: 动态补齐到 75% (与渲染共存)"
echo "  - GPU 0-3: 训练间隙填充"
echo "=============================================="

# Create training dataset
$PYTHON -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config_balanced.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

# GPU 4 (渲染卡): 占空比 + SM满载控制
HOLDER_GPU=4 HOLDER_MEM_FRAC=0.75 HOLDER_TARGET=75 \
    $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
GPU_HOLDER_PID=$!
echo "GPU Holder started for GPU 4, target=75%, PID=$GPU_HOLDER_PID"

# GPU 0-3 (训练卡): SM满载占空比控制
for GPU_ID in 0 1 2 3; do
    HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
        $PYTHON "$SCRIPT_DIR/gpu_holder.py" &
    eval "GPU_WARMER_PID_${GPU_ID}=$!"
    echo "GPU Holder started for GPU $GPU_ID, target=75%, PID=$!"
done
GPU_WARMER_PID=$GPU_WARMER_PID_0

# Run PPO training with Qwen3-VL-8B-Instruct
$PYTHON -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.max_trajectory_length=14000 \
    data.image_key=images \
    data.truncation=left \
    +data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-8B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=14000 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.limit_mm_per_prompt=15 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=5e-6 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen3-VL-8B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_training_steps=2000 \
    rollout_manager.max_turns=12 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=False \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=False \
    +rollout_manager.mini_batch_size=4 \
    2>&1 | tee $EXPERIMENT_NAME.log
