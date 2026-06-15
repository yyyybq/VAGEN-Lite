# =============================================================================
# v18_potential (S4 真 Ng1999 γ-discounted + E1 + E4 + E5)
# =============================================================================
# 在测什么:
#   S4 (真实现, γ=0.99): env.py 新增 progress_mode="potential",
#       r_t = scale · (γ · Φ_t − Φ_{t-1})
#   定理(Ng et al. 1999): 当 γ 与 MDP 折扣一致时, potential-based shaping
#   不改变最优策略, 但密集化 reward 信号。与 v17 "delta" 的区别:
#     - delta:     r_t = Φ_t − Φ_{t-1}     (γ=1, 简单 telescoping)
#     - potential: r_t = γΦ_t − Φ_{t-1}    (γ<1, 抑制 oscillation cycle)
#   "刷分循环"(绕远再回来反复拿正 Δ)累计 reward 会因 γ<1 收敛到 0,
#   这是 v17 后期 entropy 0.41 -> 9.92 崩溃的疑似根因之一。
#
#   γ=0.99 (yaml 里设 potential_field_gamma: 0.99)。注意训练侧
#   HIGH_LEVEL_GAMMA=0.95, 严格匹配应改为 0.99 或把 yaml γ 改 0.95，
#   但 0.99 是 Ng1999 论文与多数 shaping 文献的标准选择, 60-step
#   episode 内 0.04 的 γ 失配影响很小。
#
#   E1: KL=0.05, ent=0.001  E4: step 0.30/20°  E5: prompt 已 fix
#
# 启动命令:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_potential.sh \
    > v18_potential.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_potential"
ENV_CONFIG="env_config_v18_potential.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor: E1 ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
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
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
