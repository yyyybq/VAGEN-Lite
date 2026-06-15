# v10: Per-1-Action Dense Reward，基于 v9 改进
#
# ============================================================
# v9 问题分析
# ============================================================
# [问题1] done-spam（步骤 41–63）：actor warmup 后立刻发现
#   "直接输出 done" 是局部最优策略：
#     format_reward(0.05) > 探索风险（collision_penalty=-0.15 是 per-action Δscore 的 3x）
#   模型在 23 步内自我修正，但浪费了大量 compute
#   根因：v9 的 format_reward/collision_penalty 尺度是为 per-5-action 设计的
#
# [问题2] per-5-action 的 credit assignment 问题（本次核心修复）
#   每个 LLM turn 执行 5 个 action 才拿到一次 reward 信号
#   → 模型无法区分 5 个 action 中哪个有贡献
#   → 本质上削弱了 dense reward 的设计意图
#
# [问题3] critic vf_explained_var 持续偏低（0.001–0.03）
#   advantage 估计噪声大，但 v9 后期（step 73+）有改善趋势（0.1–0.3）
#   → v9 critic_warmup=40 的设计是正确的，保留
#
# [v9 亮点] step 89 出现 score=0.735 / success=25%，说明方向正确
#   entropy 稳定（0.65–0.87）、不崩溃、无 action 偏向
#   KL loss 起了保护作用
#
# ============================================================
# v10 核心设计
# ============================================================
# [核心变化] max_actions_per_step=1（per-action dense reward）
#   每个 LLM turn 只输出 1 个 action → 1 个 reward 信号
#   模型每步都能看到精确的因果反馈：
#     "我 turn_left 了，score 从 0.3 提升到 0.33"
#
# [补偿 max_actions=1 导致的每回合 action 数减少]
#   max_turns: 12 → 20（每 turn 1 action，20 turns = 20 actions）
#   vs v9: 12 turns × avg 4.5 actions ≈ 54 actions
#   轨迹覆盖范围与 v9 相当，但 reward 信号密度是 v9 的 5x
#
# [防 done-spam：奖励尺度重新匹配 per-action]
#   Per-action 典型 Δscore ≈ 0.03–0.08（v9 per-5-action ≈ 0.15–0.4）
#   format_reward: 0.05 → 0.01   （< Δscore，探索有利可图）
#   collision_penalty: -0.15 → -0.05  （v9 的 collision_penalty 是 Δscore 的 3x，过强）
#   这两个参数改在 env_config_single_action.yaml 中
#
# [上下文窗口适配]
#   max_trajectory_length: 26000 → 30000
#     20 turns × (image ~800tok + text ~300tok + response ~200tok) ≈ 26000
#     + system + initial obs ≈ 1500 → total ~27500，设 30000 留余量
#   max_response_length: 512 → 256
#     1-action response 格式极短：
#       "<think>Current score: 0.3... I will move forward.</think>\n<action>move_forward|</action>"
#       实际约 80–150 tokens，256 足够
#
# [继承 v9 稳定性设计（全部保留）]
#   entropy_coeff=0.008：温和多样性
#   kl_loss_coef=0.015：锚定干净 ref，防单调化
#   critic_warmup=40：先让 critic 对齐真实 return 再启动 actor
#   cliprange_value=0.8：新 critic 快速修正
#   temperature=0.9、grad_clip=0.5：稳定
#
# ============================================================
# v10 参数总结
# ============================================================
#   env_config:     env_config_single_action.yaml（max_actions_per_step=1）
#   max_turns:      20   (was 12; compensate for fewer actions/turn)
#   max_response_length: 256  (was 512; 1-action responses are short)
#   max_trajectory_length: 30000  (was 26000; 20 turns need more space)
#   RESUME:         disable（从头训练）
#   entropy_coeff:  0.008   (keep from v9)
#   kl_loss_coef:   0.015   (keep from v9)
#   critic_warmup:  40      (keep from v9)
#   cliprange_value: 0.8    (keep from v9)
#   critic_lr:      2e-5    (keep from v9)
#   grad_clip:      0.5     (keep from v9)
# ============================================================

EXPERIMENT_NAME="v10_per_action_dense"
ENV_CONFIG="env_config_single_action.yaml"
NUM_TRAIN_GPUS=4   # 与 v9 一致：GPU 0-3 训练
                   # 不用 6 训练卡：6 不能被 Qwen 注意力头整除 → TP 只能=2
                   #   TP=2 时每 GPU vLLM KV cache 翻倍 → 反复 OOM
RENDERING_GPU=4    # GPU 4 渲染（GPU 5,6 闲置）

# === 从头训练：env 格式变了（1 action/turn），不能复用 v9 checkpoint ===
RESUME_MODE="auto"

# === 继承 v9 的稳定性参数（全部保留） ===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4  # 4 训练卡 ÷ TP=4 → 1 个 vLLM 实例（与 v9 一致）
           # KV cache 在 4 GPU 上分摊，每 GPU vLLM 占用最低
GPU_MEM_UTIL=0.40  # 降低以避免与 FSDP worker 内存竞争（H200 139.8GB，FSDP 占 ~76GB，剩余 ~63.4GB，0.5*139.8=69.9GB > 63.4GB OOM）

# === Critic：继承 v9 ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=40
CLIPRANGE_VALUE="0.8"

# === 梯度 ===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 适配 per-1-action 的轨迹参数 ===
MAX_TURNS=12               # 与 v9 一致；per-action 设计下 12 turns = 12 个 dense reward 信号（已超 v9 的 ~10 个）
                           # 原 20 是过度补偿：导致 critic 单 GPU 累计 ~96K tok > 80K (OOM threshold)
WINDOW_SIZE=5              # 保持不变
MAX_TRAJECTORY_LENGTH=16000  # 12 turns × ~1200tok/turn ≈ 14400；与 v9 的 26000 路线一致安全
MAX_RESPONSE_LENGTH=512    # 冷启动阶段模型仍输出多-action think，512 防截断导致 </action> 缺失 → parse 失败
MAX_PROMPT_LENGTH=2048     # 不变

# === 数据参数 ===
TRAIN_BATCH_SIZE=24  # 4-GPU: 24/4=6/GPU，6/chunk=2=3 整除 ✓
                     # 与 PPO_MINI_BATCH_SIZE=12 配合：24/12=2 mini-batches
VAL_BATCH_SIZE=8     # 4-GPU: 整除 4

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12  # 4-GPU: 12/4=3 整除 ✓
MINI_BATCH_SIZE=8       # rollout_manager.mini_batch_size（4-GPU 整除 4）
N_TRAJECTORY=1          # 与 v9 一致；N=2 在 max_turns=20+window=5 下会让 critic backward 的拼接序列长度
                        # 翻倍至 ~48K tokens，attention O(N²) 达到 2.3B 元素，单层 ~9 GiB → OOM
                        # 等 critic 训练稳定后可以考虑用 critic.ppo_max_token_len_per_gpu 限流后再开 N=2

# === 训练参数 ===
SAVE_FREQ=20
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"             # Dense reward 场景：lam 不应过低
                       # γ=1.0, lam=0.95 → 有效 horizon=20 steps，正好覆盖整个 episode
                       # lam=0.9 → horizon=10 steps，会截断后半段的真实 dense reward 信号
                       # v9 vf_explained_var 低的根因是 done-spam 双峰分布，非 lam 问题
                       # v10 的 format_reward=0.01 已消除 done-spam，无需降 lam 来减方差
