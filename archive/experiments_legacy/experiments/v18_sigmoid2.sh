# =============================================================================
# v18_sigmoid2 — v18_sigmoid 续作 (3 大修复)
# =============================================================================
# 修复内容:
#   F1 (env.py)    info["success"]/["is_success"] 现在与 traj_metrics 同步,
#                  解锁 val-aux/active_spatial/traj_success/mean@1 (此前永远 0).
#   F2 (run_*.sh)  val_kwargs.{temperature,top_p,do_sample,n} 注入,
#                  打破 val_reward 卡常数的 greedy 确定性.
#   F3 (yaml)      success_score_threshold 0.20 -> 0.45,
#                  避免 v18_sigmoid 出现的 env_turns≈1.6 episode 塌缩.
#
#   AND-gate 字段已加好但默认关闭 (success_require_both=false) — 留给下一版
#   v18_andgate 实验切换 (success_require_both=true,pos_thr=0.5,ori_thr=0.5).
#
# 其余 actor / critic / optim 超参 全部沿用 v18_sigmoid, 便于直接对照.
#
# 启动命令:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v18_sigmoid2.sh \
    > v18_sigmoid2.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v18_sigmoid2"
ENV_CONFIG="env_config_v18_sigmoid2.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (沿用 v18_sigmoid) ===
ENTROPY_COEFF="0.001"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.05"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling (F2 修复) ==========================================
# 之前未设置 → 默认 greedy (do_sample=False, n=1, temp=0) → 与固定 val seeds
# 组合得到完全确定的轨迹 → val_reward 一直卡常数. 这里强制 stochastic + n=4.
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic (沿用 v18_sigmoid) ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30
CLIPRANGE_VALUE="0.8"

# === Optim (沿用 v18_sigmoid) ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === 轨迹 (沿用 v18_sigmoid) ===
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
