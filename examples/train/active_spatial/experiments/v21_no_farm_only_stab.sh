# =============================================================================
# v21_no_farm_only_stab — v19_no_farm yaml × v20_winner sh 稳定补丁
# =============================================================================
# 动机:
#   横向对比:
#     v19_no_farm        峰值 best@4=47.45 / tsucc_b4=0.880 @ step40, 崩在 step 180
#     v19_thr70_andgate  峰值 best@4=46.59 / tsucc_b4=0.859 @ step40, 崩在 step 120
#     v20_winner (上)    峰值 best@4=44.95 / tsucc_b4=0.819 @ step40, 未崩 (still early)
#     v19_no_farm_dual   峰值 best@4=39.57 / tsucc_b4=0.715 @ step20, only 2 ckpt
#
#   v20_winner 用 "no_farm + dual + thr70_andgate" 三路 yaml 合并 + 4 个 sh 补丁,
#   但峰值落到 v19_no_farm 之下. 假设:dual reward 与 no_farm 的奖励整形目标互相
#   冲突 (no_farm 偏稀疏, dual 偏 dense ori 信号), 拖低了峰值.
#
#   本实验做一个干净的消融: 只保留 no_farm 的 yaml, 叠上 v20 的稳定 sh 补丁,
#   看能否拿到 "v19_no_farm 峰值 (~0.88) + v20 的不崩" 组合.
#
# yaml: env_config_v19_no_farm.yaml  (与 v19_no_farm 完全一致)
# sh:   v19_no_farm 基线 + 全部 v20_winner 的 4 个稳定性改动
#
# 与 v19_no_farm 的 sh 差异 (与 v20_winner 相同的 4 个补丁):
#   ★ KL_LOSS_COEF:      0.05 -> 0.10
#   ★ CLIPRANGE_VALUE:   0.8  -> 0.5
#   ★ CRITIC_WARMUP:     30   -> 60
#   ★ ENTROPY_COEFF:     0.001 -> 0.002
#
# 与 v20_winner 的差异:
#   只有 ENV_CONFIG (no_farm vs no_farm_dual). 这样 v21_no_farm_only_stab vs
#   v20_winner 的 delta 就唯一归因于 dual reward 是否值得加.
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v21_no_farm_only_stab.sh \
    > v21_no_farm_only_stab.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v21_no_farm_only_stab"
ENV_CONFIG="env_config_v19_no_farm.yaml"   # ★ 与 v20 不同:只用 no_farm, 不加 dual
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.002"          # ★ v19_no_farm 是 0.001
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"            # ★ v19_no_farm 是 0.05
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
CRITIC_WARMUP=60               # ★ v19_no_farm 是 30
CLIPRANGE_VALUE="0.5"          # ★ v19_no_farm 是 0.8

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data ===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
