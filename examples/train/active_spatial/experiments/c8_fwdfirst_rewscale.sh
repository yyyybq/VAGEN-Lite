# =============================================================================
# c8_fwdfirst_rewscale — Cambrian-S 7B + fwd-first 提示 + 低方差奖励结构
# =============================================================================
# 动机（基于 c4/v30 结果，2026-06-08）:
#
#   c4_fwdfirst 结果:
#     - step100 达到 Cambrian 历史最佳 ID_m4=0.500（fwd-first 提示有效）
#     - step80+ entropy=1.06 → step140 entropy=2.05（崩塌，与 Qwen 实验模式一致）
#     - 根因假设：success_reward=50 导致高方差（同 v28 崩塌机制）
#
#   v30 (Qwen) 验证：success_reward=5 + scale=1.0 后 entropy 自愈，step194 stable
#
#   c8 假设:
#     将 v30 的奖励重设计移植到 Cambrian c4 配置，
#     若奖励方差是 Cambrian 崩塌的主因（而非模型规模），
#     则 c8 应在 step150+ 仍保持 entropy < 1.0。
#
# 相对 c4_fwdfirst 的唯一改动（env_config 层面）:
#   ★ ENV_CONFIG: env_config_v24_100scenes_fwdfirst.yaml
#              → env_config_v24_100scenes_fwdfirst_rewscale.yaml
#     · potential_field_reward_scale: 0.5 → 1.0
#     · near_success_bonus:           0.2 → 0.5
#     · success_reward:               50  → 5
#   所有训练超参（LR、算法、batch、Cambrian-specific）与 c4 完全相同
#
# GPU: 5卡 H200 (4 train + 1 render)，与 c4 一致
#
# 启动 (7卡 H200):
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c8_fwdfirst_rewscale.sh \
    > c8_fwdfirst_rewscale.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="c8_fwdfirst_rewscale"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst_rewscale.yaml"  # ★ fwd-first + rewscale
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (与 c4 完全相同) ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=2               # Cambrian-7B: TP=2（vLLM KV cache 更大）
GPU_MEM_UTIL=0.20       # Cambrian-7B: 保守内存分配

# === Validation sampling ===
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic (与 c4 完全相同) ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim (与 c4 完全相同) ===
GRAD_CLIP="0.3"
ACTOR_LR="5e-7"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=8
VAL_BATCH_SIZE=4

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=4
MINI_BATCH_SIZE=4

# === Trainer ===
SAVE_FREQ=50
TEST_FREQ=50
TOTAL_STEPS=1000
VAL_BEFORE_TRAIN="True"

# === Algorithm (PPO + no_concat_gae，与 c4 相同) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Model: Cambrian-S 7B (与 c4 完全相同) ===
MODEL_PATH="/scratch/by2593/hf_cache/cambrian-s-7b"

EXTRA_OVERRIDES="\
    actor_rollout_ref.model.external_lib=vagen.models.cambrian_register \
    actor_rollout_ref.model.trust_remote_code=True \
    critic.model.external_lib=vagen.models.cambrian_register \
    critic.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    critic.model.use_remove_padding=False"

# === OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
