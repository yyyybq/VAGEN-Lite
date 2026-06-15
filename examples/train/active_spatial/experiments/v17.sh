# =============================================================================
# v17: 基础设计修复版 (相对 v16)
# =============================================================================
# 核心目的:
#   v11..v16 在 240+ training step 后 traj_success 始终为 0。8-question 诊断
#   定位到 4 个根本性设计缺陷:
#     1. 模型大量幻觉 `move_left` (~75% action invalid) 但 action_space 没有
#     2. success_reward=5 与 dense 累积 ~5 同量级 -> 完成与接近无差异
#     3. 显式 done 惩罚 -0.3 vs 自动 success +5 不对称 -> 模型学到不要 done
#     4. delta-only shaping 在 score 附近高原存在对称局部最优
#
#   v17 不再做参数微调，而是动手术:
#     ★ action_space=strafe  (加 move_left/right, 删 look_up/down)
#     ★ enable_explicit_done=false (只允许自动 success)
#     ★ success_reward 5 -> 50, reward_scale 5 -> 0.5 (10:1 完成 vs 接近)
#     ★ success_score_threshold 0.50 -> 0.30 (curriculum 起点)
#     ★ near-success bonus +0.2/step (state reward, 打破对称震荡)
#     ★ 删除 collision/visibility (简化奖励, 降低噪声)
#
# ============================================================
# 诊断: 新增 traj_metrics 字段 (env.py)
# ============================================================
#   - invalid_action_count        : 整 episode 非法 action 次数
#   - best_score / final_score    : 轨迹最高/终止分数
#   - final_position_score        : 位置分量 (诊断"靠得近但角度错"等)
#   - final_orientation_score     : 朝向分量
#   - near_success_step_count     : 在 near-success 阈值上方的步数
#   - near_success_bonus_total    : 累计的 near-success 奖励
#   - success_by_done / by_auto / by_max_steps : 成功来源分布
#   - action_count/<name>         : 每个动作的调用次数 (wandb 直接看分布)
#
# ============================================================
# 训练参数 (相对 v16 的变化用 ★ 标注)
# ============================================================
#   ENV_CONFIG:            ★ env_config_v17.yaml
#   MAX_RESPONSE_LENGTH:   ★ 768 -> 384  (阻断 100+ token hallucination)
#   ENTROPY_COEFF:         ★ 0.001 -> 0.005 (新 action space, 需重新探索)
#   KL_LOSS_COEF:          ★ 0.04 -> 0.02 (action space 变了, 允许更大漂移)
#   TEMPERATURE:           ★ 0.7 -> 0.8 (探索新空间)
#   MAX_TURNS:             15 -> 12     (与 max_episode_steps=60 对齐)
#   MAX_TRAJECTORY_LENGTH: 24000 -> 18000 (response 短了 -> 总长降)
#   其余: per-5-action, N=1, TRAIN_BATCH_SIZE=24, gamma=0.95
#
# ============================================================
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v17.sh \
    > v17.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v17"
ENV_CONFIG="env_config_v17.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"          # action space changed -> 不可 resume v16

# === Actor: 新 action space 需要重新探索 ===
ENTROPY_COEFF="0.005"          # ★ 0.001 -> 0.005
USE_KL_LOSS="True"
KL_LOSS_COEF="0.02"            # ★ 0.04 -> 0.02 (允许策略漂移到新空间)
TEMPERATURE="0.8"              # ★ 0.7 -> 0.8
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

# === ★ 轨迹参数 ===
MAX_TURNS=12                   # ★ 15 -> 12 (= max_episode_steps 60 / 5)
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000    # ★ 24000 -> 18000 (response 缩短)
MAX_RESPONSE_LENGTH=384        # ★ 768 -> 384 (阻断 hallucinated repetition)
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
