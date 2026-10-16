# v17_n4: v17 (full) 的 N_TRAJECTORY ablation
# 唯一变化：N_TRAJECTORY 1 → 4 (每个 env 采 4 条 rollout 而非 1 条)
#
# 动机：
#   - 当前 v17 critic/score mean=33 / max=50，advantage 估计噪声大
#   - n=1 时 PPO advantage 完全靠 critic baseline，critic 噪声会直接传给 actor
#   - n=4 让同一个 env 的 4 条轨迹形成"小 group"，advantage 内部相对，方差降低
#     (这是 GRPO 的思想，但仍用 GAE+critic，不是真正的 GRPO；真正的 GRPO 需要
#      实现 no_concat_grpo estimator，工作量大，先用 n=4 试)
#
# 资源影响：
#   - rollout 阶段：24 env × 4 traj = 96 个 rollout/step (v17 是 24)
#     vLLM async 串行 → 单 step 生成时间约 4×；v17 单 step ~9 min → n=4 约 30+ min
#   - 训练阶段：96 traj × episode_length ~3 = ~290 transitions，mini-batch 不变
#   - 内存：rollout token 数 4×，但都过 vLLM 不在 actor 上，应该 OK
#   - VAL_BATCH_SIZE 不变 (val 仍 n=1)
#
# 与 v17 完全相同的其他参数：env_config_v17.yaml, action_space=strafe, success_reward=50,
# scale=0.5, threshold=0.30, response=384, entropy=0.005, KL=0.02, temp=0.8
#
# ⚠️ 训练速度会显著变慢，若想保持步数预算可同时把 TOTAL_STEPS 降一半，但建议保持
# 不变以观察 advantage 改善是否抵消速度损失。
#
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v17_n4.sh \
    > v17_n4.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v17_n4"
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

# === 轨迹 (与 v17 完全一致) ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === ★ 唯一变化：N_TRAJECTORY 1 → 4 ===
N_TRAJECTORY=4                # ★ 每个 env 采 4 条 rollout
TRAIN_BATCH_SIZE=24           # env 数不变；总 rollout = 24 × 4 = 96
VAL_BATCH_SIZE=8

# === PPO mini-batch (按总 rollout=96 略调) ===
PPO_MINI_BATCH_SIZE=24        # ★ 12 → 24 (96 / 4 = 24，4 个 PPO update per step)
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
