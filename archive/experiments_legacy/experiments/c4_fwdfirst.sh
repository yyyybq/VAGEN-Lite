# =============================================================================
# c4_fwdfirst — Cambrian-S 7B + forward-first 提示 + LR=5e-7
# =============================================================================
# 动机 (基于 c1/c2 rollout 分析, 2026-06-03):
#
#   c1/c2 的根本失败原因:
#     ① turn_left 行为偏好过强: 43.2% 的动作是 turn_left，仅 24.5% 是 move_forward
#     ② 系统提示的 Hint 4 ("Look around if you're unsure") 和 Hint 5
#        ("FIRST rotate...THEN translate") 对 Cambrian 被过度执行 → 持续旋转不前进
#     ③ LR=1e-6 在 Qwen 实验中已验证不稳定（v25 在 step~180 entropy 爆炸）
#     ④ 16.6% 的响应无 <action> 标签（纯英文）；5.8% 动作名拼写错误（大小写/标点）
#
#   c3 的修复策略 (相对 c1):
#     ★ ENV_CONFIG: env_config_v24_100scenes.yaml → env_config_v24_100scenes_fwdfirst.yaml
#                   prompt_format: free_think → free_think_fwd_first
#                   新 Hint 4: "DEFAULT to move_forward; only turn when needed; max 2 consecutive turns"
#                   新 Hint 5: "AVOID spinning; if reward not improving after turns, move_forward"
#                   新增: 显式列出有效动作名（精确拼写）
#     ★ ACTOR_LR: 1e-6 → 5e-7  (与 v26_klhi_lr5e7 一致，减少初期 entropy 爆炸风险)
#     其他参数与 c1 完全相同（TP_SIZE=2, batch=8, critic_warmup=60, KL=0.20）
#
#   科学假设:
#     若 turn_left 偏好源于提示策略（"先转后走"），则改为 forward-first 提示后，
#     Cambrian 应在 step 50-100 时展现 move_forward 主导行为，且 t_succ% 应上升到 >2%。
#     若 turn_left 偏好是 Cambrian 的内在先验（与提示无关），则 c3 仍会失败，
#     说明需要 SFT 冷启动（c4 待定）。
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c4_fwdfirst.sh \
    > c4_fwdfirst.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="c4_fwdfirst"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst.yaml"  # ★ 使用 forward-first 提示格式
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
ACTOR_LR="5e-7"              # ★ 1e-6 → 5e-7（与 v26_klhi 一致，更稳定）

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
SAVE_FREQ=80
TEST_FREQ=50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="True"      # 记录 forward-first 提示下的起点性能（与 c1 step0 对比）

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
