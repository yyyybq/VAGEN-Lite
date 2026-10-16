# v15_v3_test: 修复 episode 提前终止 bug（gym_agent_loop_no_concat.py）
#
# ============================================================
# 实验目的
# ============================================================
# 在 VAGEN-Lite 框架下重现老 v9 的 per-5-action 设计，
# 与 v11（per-1-action）做控制变量对比，量化 credit assignment 改进效果：
#
#   v11：per-1-action，每 turn 1个 dense reward 信号，max_turns=50（共50 action）
#   v15：per-5-action，每 turn 5个 action 后1个 reward 信号，max_turns=10（共50 action）
#
#   两者总动作数相同（50），唯一变量：reward 信号频率（50次 vs 10次）
#   → 若 v11 显著好于 v15，说明 per-action dense reward 确实有效
#   → 若两者相近，说明 credit assignment 不是主要瓶颈
#
# ============================================================
# 与老 v9 的关键差异（VAGEN-Lite 框架升级）
# ============================================================
# [升级 1] no_concat_gae（替代老 masked_gae）
#   VAGEN-Lite 的 no_concat 模式：每个 turn 独立前向，无 multi-turn 拼接
#   → 避免老 VAGEN 的长序列 attention OOM 问题
#
# [升级 2] critic value_mask bug 已修复
#   老 v9 无此 bug（不同实现），VAGEN-Lite 已修复 ray_trainer.py + dp_critic.py
#
# [升级 3] 256×256 图像（老 v9 使用 512×512 + ANYRES）
#   10 turns × ~342 patches = 3,420 patches，远低于 OOM 阈值
#
# [升级 4] premature_done_penalty=-0.3（老 v9 无此防护）
#   per-5-action 下 done-spam 风险低（format_reward=0.05 <  Δscore=0.15-0.4）
#   但保留作为安全防护
#
# ============================================================
# 奖励尺度设计（per-5-action 专属）
# ============================================================
#   使用 env_config_5action_256.yaml：
#     max_actions_per_step: 5
#     format_reward: 0.05        （老 v9 一致；< per-5-action Δscore=0.15-0.4）
#     collision_penalty: -0.15   （老 v9 一致）
#     step_penalty: -0.01/env_step（5 action/turn → 每轮 -0.05）
#     premature_done_penalty: -0.3
#
# ============================================================
# 轨迹参数设计
# ============================================================
#   MAX_TURNS=10：10 turns × 5 actions = 50 总 action（与 v11 的 50 对齐）
#   MAX_RESPONSE_LENGTH=512：per-5-action 输出格式较长，需要 512
#     格式：<think>...</think>\n<action>a1|a2|a3|a4|a5|</action>
#     估算：think ~200 tok + 5 actions ~50 tok ≈ 250 tok，512 足够
#   MAX_TRAJECTORY_LENGTH=16000：
#     10 turns × (256px image ~342tok + text ~200tok + response ~512tok) ≈ 10,540
#     + system + initial obs ≈ 1500 → total ~12,000；设 16000 留余量
#   MAX_PROMPT_LENGTH=2048：
#     window_size=1 时单轮 prompt ~700 tok，2048 足够
#
# ============================================================
# v15 参数总结（相对 v11 的变化用 ★ 标注）
# ============================================================
#   ENV_CONFIG:     ★ env_config_5action_256.yaml（per-5-action，256px）
#   MAX_TURNS:      ★ 50 → 10（10×5=50 总动作，与 v11 等效）
#   MAX_TRAJECTORY_LENGTH: ★ 28000 → 16000（10 turns，更短）
#   其余与 v11 完全相同（N=1, TRAIN_BATCH_SIZE=24, gamma=0.95 等）
# ============================================================
#
# 启动命令：
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v16_update15.sh \
    > v16.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v16"  # _v3_test: 修复 episode 提前终止 bug（gym_agent_loop_no_concat.py）
                                     # 改动：ENTROPY_COEFF 0.008→0.001（per-5-action 信号稀疏需更小 entropy bonus）；
                                     #       KL_LOSS_COEF 0.015→0.04（KL 与 pg_loss 同量级以约束策略漂移）；
                                     #       TEMPERATURE 0.9→0.7（top_p=0.92 已够探索性，降温稳熵）；
                                     #       GRAD_CLIP 0.5→0.3（v2 grad_norm 8-10 clip 0.5 几乎无效）；
                                     #       CRITIC_WARMUP 10→30（reward 信号弱时让 critic 充分预热）。
ENV_CONFIG="env_config_5action_suc50.yaml"   # ★ per-5-action，奖励尺度匹配老 v9
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"  # ★ v3 从头训练，不 resume v2 的 corrupted checkpoint

# === ★ v3 调整：抑制 entropy 爆炸 + 加强 KL 防护 ===
ENTROPY_COEFF="0.001"   # ★ 0.008 → 0.001：v2 entropy 12× 爆炸的元凶
USE_KL_LOSS="True"
KL_LOSS_COEF="0.04"     # ★ 0.015 → 0.04：KL 防护提到与 pg_loss 同量级
TEMPERATURE="0.7"       # ★ 0.9 → 0.7：配合 top_p=0.92 已够探索
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=30         # ★ 10 → 30：per-5-action reward 信号弱，多预热
CLIPRANGE_VALUE="0.8"

# === 梯度 ===
GRAD_CLIP="0.3"          # ★ 0.5 → 0.3：v2 grad_norm 常 8-10，加严
ACTOR_LR="1e-6"

# === ★ 轨迹参数（per-5-action 适配）===
MAX_TURNS=15               # ★ 50 → 10：10 turns × 5 action/turn = 50 总动作
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=24000  # ★ 28000 → 16000：10 turns × ~1200tok/turn ≈ 12000
MAX_RESPONSE_LENGTH=768    # per-5-action 输出较长，保留 512
MAX_PROMPT_LENGTH=2048

# === 数据参数（继承 v11）===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch（继承 v11）===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === 训练参数 ===
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数（继承 v11）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
