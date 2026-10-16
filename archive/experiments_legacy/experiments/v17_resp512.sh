# v17_resp512: v17 (full) 的 response-length ablation
# 唯一变化：MAX_RESPONSE_LENGTH 384 → 512 (+ MAX_TRAJECTORY_LENGTH 18000 → 20000 适配)
#
# 动机：v17 rollout 显示
#   - response_length/clip_ratio = 0.33 (33% 输出被 384 token 截断)
#   - 46% rollout 没有合法 <action> 块 (think 段吃完 budget)
#   - rollout 里 turn_left/turn_right 占比 = 0% (可能是 think 写完才轮到 action 时已被截)
#
# 假设：384 太短 → think 段占满后 <action> 没生成 → format 失败 + 模型偏向更短的
# action 序列 (纯平移 < 包含 turn 的多动作组合)。给到 512 是否能恢复 turn 使用 +
# 降低 clip_ratio 是这个版本要回答的。
#
# 与 v17 完全相同的其他参数：env_config_v17.yaml, action_space=strafe, success_reward=50,
# scale=0.5, threshold=0.30, near_success 0.35/0.2, entropy=0.005, KL=0.02, temp=0.8
#
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v17_resp512.sh \
    > v17_resp512.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v17_resp512"
ENV_CONFIG="env_config_v17.yaml"      # 完全复用 v17 的环境
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (与 v17 完全一致) ===
ENTROPY_COEFF="0.005"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.02"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30
CLIPRANGE_VALUE="0.8"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === ★ 唯一变化：response 长度从 384 → 512 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=20000   # ★ 18000 → 20000 (12 turn × +128 tok ≈ +1500)
MAX_RESPONSE_LENGTH=512       # ★ 384 → 512
MAX_PROMPT_LENGTH=2048

# === 数据 ===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === 训练 ===
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
