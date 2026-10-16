# =============================================================================
# c6_no_think — Cambrian-S 7B + no_think 格式（去掉 think 标签合规障碍）
# =============================================================================
# 动机（基于 c1 rollout 分析）:
#
#   c1 的两类格式失败：
#     ① 16.6% 响应无 <action> 标签（纯英文散文，format penalty -0.200）
#     ② 5.8% 响应 action 名称拼写错误（`move Forward`、`move.forward`等）
#
#   c1~c5 均使用 free_think 格式，要求模型输出：
#     <think>推理过程</think><action>action1|action2|</action>
#   Cambrian 可能在长推理过程中"忘记"添加 <action> 标签，或将推理直接写成动作指令。
#
#   c6 切换到 no_think 格式，只需要输出：
#     <action>action1|action2|</action>
#   彻底消除 think 格式的合规障碍，最小化格式复杂度，
#   让模型把学习资源集中在 action 选择上而不是格式合规上。
#
#   副作用：响应更短（从 ~200-300 tokens → ~10-30 tokens），
#           throughput 大幅提升，可能加速 reward 学习。
#
# 相对 c3（基线）的差异:
#   ★ ENV_CONFIG: env_config_v24_100scenes.yaml → env_config_v24_100scenes_no_think.yaml
#                 prompt_format: free_think → no_think
#   其他参数与 c3 一致 (ACTOR_LR=5e-7)
#
# 科学假说:
#   若 Cambrian 的格式失败（16.6%）是 think-format 的认知负担导致的，
#   则 no_think 格式下格式合规率应 > 95%，且 reward 曲线应更快上升。
#   若 no_think 仍然有大量格式失败，说明 Cambrian 根本不理解 action 输出格式，
#   需要 SFT 冷启动来建立格式先验。
#
#   额外观察指标：turn_left 偏好是否在 no_think 下减轻？
#   （如果 think 过程中的推理强化了 turn 策略，no_think 可能打破这个循环）
#
# 对比矩阵（相对 c1 基线）:
#   c3: LR 5e-7  (诊断 LR 影响)
#   c4: fwd-first prompt (诊断 prompt 策略影响)
#   c5: entropy 0.02 (诊断熵正则影响)
#   c6: no_think format (诊断 think-format 合规障碍) ← 本实验
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c6_no_think.sh \
    > c6_no_think.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="c6_no_think"
ENV_CONFIG="env_config_v24_100scenes_no_think.yaml"  # ★ no_think 格式（无 think 标签）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=2
GPU_MEM_UTIL=0.20

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
ACTOR_LR="5e-7"              # 与 c3/c4/c5 一致

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=128      # ★ no_think 响应极短（仅 <action>...</action>），128 足够
MAX_PROMPT_LENGTH=2048

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=8
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

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
