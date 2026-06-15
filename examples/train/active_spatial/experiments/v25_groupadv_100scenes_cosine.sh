# =============================================================================
# v25_groupadv_100scenes_cosine — GroupAdv (N=4) + 94 场景 + cosine LR decay
# =============================================================================
# 动机:
#   基于 v21-v24 系列分析（2026-05-24）:
#
#   核心发现:
#     - 所有能训起来的实验（v19-v24）均在 step ~40 出现 best@4 峰值，之后开始衰减
#       → 常数 ACTOR_LR=1e-6 在 sweet spot 之后偏大，是峰后衰减的共同原因
#     - v21_lrdecay 试验过 cosine decay，但配置是 v20_winner（单场景 + no_farm_dual），
#       本身就存在崩塌风险；v21_lrdecay 最终在 step 140 跌至 0.7368
#       → v21_lrdecay 的失败可能源于"单场景 + 不稳定的 yaml"，而非 cosine 本身无效
#     - v24_groupadv_100scenes 已验证在 100 场景 + GroupAdv 下稳定无崩塌（step 58）
#       → 在更稳定的底座上重试 cosine，排除之前 yaml 不稳定的干扰
#
#   本实验假设:
#     cosine LR decay（warmup 10% steps, min_lr=0.1×base）在 v24 的稳定底座上：
#       1. 峰值 step 从 ~40 推迟至 ~80-120（LR 下降后梯度更保守）
#       2. 峰后衰减放缓，best@4 在更长时间窗口内维持高位
#       3. OOD 退化速度减慢（更保守的更新减少 in-domain 过拟合）
#
#   消融位置 (相对 v24_groupadv_100scenes):
#     ★ EXTRA_OVERRIDES: actor_rollout_ref.actor.lr_scheduler_type=cosine
#                        actor_rollout_ref.actor.warmup_steps_ratio=0.1
#                        actor_rollout_ref.actor.min_lr_ratio=0.1
#     其他全部不变
#
# 注意:
#   - run_experiment.sh 已在 §8.4 更新，支持 ${EXTRA_OVERRIDES:-} 追加 hydra override.
#   - TOTAL_STEPS=500 以观察完整的 cosine curve（与 v21_lrdecay 保持同等观察窗口）.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v25_groupadv_100scenes_cosine.sh \
    > v25_groupadv_100scenes_cosine.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v25_groupadv_100scenes_cosine"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # 94 scenes, 28662 tasks（与 v24 共用）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.002"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

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
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4
TRAIN_BATCH_SIZE=12
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=80
TEST_FREQ=50
TOTAL_STEPS=500                    # ★ 观察完整 cosine curve（同 v21_lrdecay 窗口）
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === ★ cosine LR decay（核心改动）===
# lr_warmup_steps_ratio=0.1 → 前 50 steps 线性 warmup（TOTAL_STEPS=500）
# min_lr_ratio=0.1          → 最小 LR = 1e-6 × 0.1 = 1e-7
# 字段位于 actor.optim 下（verl OptimizerConfig），非 actor 直接下
export EXTRA_OVERRIDES="\
  actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
  actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
