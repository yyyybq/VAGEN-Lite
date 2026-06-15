# =============================================================================
# v25_groupadv_100scenes_klhi — GroupAdv (N=4) + 94 场景 + KL↑ + entropy↑
# =============================================================================
# 动机:
#   基于 v21-v24 系列分析（2026-05-24）:
#
#   核心发现:
#     - v22_kl_ablation 证实 KL=0.05 是灾难性崩塌诱因（步 180 ID/OOD 同归零）
#       → KL_LOSS_COEF ≥ 0.10 是硬性底线
#     - v21_no_farm_only_stab 等单场景实验普遍在 step ~200 出现 PPO 崩塌；
#       entropy 低（<0.46）是崩塌前的 leading indicator
#     - v24_groupadv_100scenes（KL=0.10, entropy=0.002）当前稳定，但
#       "KL 是否已足够强"尚未探索
#
#   本实验假设:
#     KL_LOSS_COEF: 0.10 → 0.20  更强 anchor 约束 → 防止 actor 离 ref policy 漂移
#     ENTROPY_COEFF: 0.002 → 0.005  更强探索奖励 → 减缓策略固化 + 缓解 OOD 退化
#     两者协同: 探索更充分（entropy↑）+ 固定更稳（KL↑），理论上延迟崩塌时间
#
#   消融位置 (相对 v24_groupadv_100scenes):
#     ★ KL_LOSS_COEF: 0.10 → 0.20
#     ★ ENTROPY_COEFF: 0.002 → 0.005
#     其他全部不变
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v25_groupadv_100scenes_klhi.sh \
    > v25_groupadv_100scenes_klhi.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v25_groupadv_100scenes_klhi"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # 94 scenes, 28662 tasks（与 v24 共用）
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.005"              # ★ 0.002 → 0.005：更强探索奖励
USE_KL_LOSS="True"
KL_LOSS_COEF="0.20"                # ★ 0.10 → 0.20：更强 KL 约束
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
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
