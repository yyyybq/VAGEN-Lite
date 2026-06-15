# =============================================================================
# c7_fwdfirst_ehi — Cambrian-S 7B + fwd-first 提示 + 适度 entropy 提升
# =============================================================================
# 动机（基于 c3~c6 消融结论，2026-06-03）:
#
#   c4_fwdfirst 确立了核心结论：
#     ✅ fwd-first 提示将 move_forward 从 24% 提升至 53-57%，解决 turn_left 偏好
#     ✅ val step50: ID_m4=0.405, OOD_m4=0.382（显著优于其他所有实验）
#
#   c4 残留的两个问题：
#     ① no_tag 率仍有 13-18%（响应中缺失 <action> 标签，浪费样本效率）
#     ② ID_m4=0.405 尚未达到 Qwen3B 峰值 0.607，有进一步提升空间
#
#   c7 的核心假设：
#     适度提升 entropy_coeff（0.005→0.01）可在维持 fwd-first 方向偏好的前提下：
#     1. 鼓励更多动作多样性，减少 no_tag 率（模型被强制输出更多有效 token）
#     2. 降低早期策略坍缩风险，延长有效学习窗口
#     注意：c5 用 entropy=0.02 但早期（5步）仍见 turn_left 44%，
#           说明 entropy 本身不能修复 prompt 策略问题；
#           c7 的改善依赖于 fwd-first prompt + entropy 双重作用。
#
# 相对 c4 的差异：
#   ★ ENTROPY_COEFF: 0.005 → 0.01  (适度 2x，目标降低 no_tag 率)
#   ★ SAVE_FREQ:     80    → 40    (更密集 checkpoint，追踪 step80/120/160 峰值)
#   其他参数与 c4 完全相同
#
# 对比矩阵：
#   c4: fwd-first, entropy=0.005  ← 已证明有效，当前最优基线
#   c7: fwd-first, entropy=0.01   ← 本实验：测试适度 entropy 是否进一步改善
#   c8: fwd-first, entropy=0.02   ← 待定：高熵组合（若 c7 有效再考虑）
#
# 启动（c6 停止后可用该 GPU 槽位）：
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/c7_fwdfirst_ehi.sh \
    > c7_fwdfirst_ehi.log 2>&1 &
echo "PID: $!"
RUN

# ---------------------------------------------------------------------------
# 可选：从 c4 的 step 80 checkpoint 续训（待 c4 保存后启用）
# 将下面 RESUME_MODE 和 EXTRA_OVERRIDES 中的 resume_from_path 取消注释：
#
# RESUME_MODE="resume_path"
# C4_CKPT="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/exps/vagen_active_spatial/c4_fwdfirst/checkpoints/global_step_80"
# EXTRA_OVERRIDES="... trainer.resume_from_path=${C4_CKPT} ..."
# ---------------------------------------------------------------------------

EXPERIMENT_NAME="c7_fwdfirst_ehi"
ENV_CONFIG="env_config_v24_100scenes_fwdfirst.yaml"   # ★ fwd-first 提示（与 c4 相同）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.01"         # ★ 0.005 → 0.01（2x 适度提升，目标降低 no_tag 率）
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
ACTOR_LR="5e-7"              # 与 c4 一致

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
SAVE_FREQ=40                 # ★ 80 → 40（更密集 checkpoint，追踪峰值位置）
TEST_FREQ=50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="True"      # 记录 c7 初始性能（应与 c4 step0 ≈ 一致）

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

# === OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
