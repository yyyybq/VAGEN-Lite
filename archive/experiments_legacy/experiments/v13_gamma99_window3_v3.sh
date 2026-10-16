# v13_v3: 修复 episode 提前终止 bug（gym_agent_loop_no_concat.py）
#
# ============================================================
# v13_v2 → v13_v3 修复
# ============================================================
# [Bug 修复] gym_agent_loop_no_concat.py 中 episode 提前终止
#   问题：_handle_env_state() 中存在以下判断：
#     if len(agent_data.turn_response_mask) >= self.response_length:
#         last_turn = True
#   其中 self.response_length = MAX_RESPONSE_LENGTH = 512
#   no_concat 模式下 turn_response_mask 仅存当前轮 token（非累积）
#   → 只要模型当轮生成满 512 token（无论内容），episode 立即终止
#
#   影响：v11~v15 所有实验都使用该代码路径
#   v13_v2 中 response_length/clip_ratio 从 step 15 起达到 1.0，
#   意味着每个 episode 在第 1 轮就被强制终止：
#     episode_length = 2（init obs + 1 action）
#     reward_variance → 0 → PPO 梯度消失 → 训练死锁
#
#   修复：删除该 if 块，episode 终止由以下两个正确机制负责：
#     1. env.py: max_episode_steps=50 → done=True → last_turn=True
#     2. agent_loop: env_max_turns=50 → last_turn=True
#
# [其余全部继承 v13_v2]
#   gamma=0.99, MAX_TURNS=50, MAX_PROMPT_LENGTH=4096
#   N_TRAJECTORY=2, TRAIN_BATCH_SIZE=12
#   所有奖励参数、稳定性超参不变
#
# ============================================================
# v13_v3 参数总结（相对 v13_v2 的变化用 ★ 标注）
# ============================================================
#   ★ gym_agent_loop_no_concat.py: 删除 response_length 触发 last_turn 的判断
#   其余与 v13_v2 完全相同
# ============================================================
#
# 启动命令：
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v13_gamma99_window3_v3.sh \
    > v13_gamma99_window3_v3.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v13_gamma99_window3_v3"
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === 继承 v13_v2 稳定性参数 ===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.3

# === Critic（继承 v13_v2）===
CRITIC_LR="2e-5"
CRITIC_WARMUP=10
CLIPRANGE_VALUE="0.8"

# === 梯度（继承 v13_v2）===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 轨迹参数（继承 v13_v2）===
MAX_TURNS=50
WINDOW_SIZE=3
MAX_TRAJECTORY_LENGTH=60000
MAX_RESPONSE_LENGTH=512
MAX_PROMPT_LENGTH=4096

# === N=2（继承 v13_v2）===
N_TRAJECTORY=2
TRAIN_BATCH_SIZE=12
VAL_BATCH_SIZE=4

# === PPO mini-batch（继承 v13_v2）===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === 训练参数 ===
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数（继承 v13_v2）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.99"
KL_COEF="0.001"
LAM="0.95"
