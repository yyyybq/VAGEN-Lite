# v11: MAX_TURNS 15 → 50，修复 Critic Bug，256×256 图像
#
# 启动命令（下面的 : <<'RUN' ... RUN 块是 bash 的多行注释，可直接复制块内命令到终端）:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v11_max_turns50.sh \
    > v11_max_turns50_v2.log 2>&1 &
echo "PID: $!"
RUN
#
# ============================================================
# v10 失败根因分析（VAGEN-Lite 实验，step 1–200）
# ============================================================
# [根本问题 1] max_turns=15 → 模型永远无法成功
#   导航任务通常需要 20-50 步才能到达目标
#   15 步内几乎不可能完成 → success_reward=1.0 从未出现
#   → 模型无从学习"向目标靠近"的行为
#   → val success rate 始终 0%，val reward ≈ -0.015
#   历史教训：老版 VAGEN v12 已经踩过此坑（max_turns=12→50）
#
# [根本问题 2] Critic 训练目标被 -100 污染（代码 BUG）
#   no_concat_gae 的 returns tensor 中：
#     - 每个 turn 只在首个有效 token 位置写入真实 return
#     - 其余所有位置填充 ignore_value=-100.0
#   正确做法：用 value_mask 屏蔽 -100 位置，只在有意义位置训 critic
#   实际情况：
#     ray_trainer.py 的 value_mask 条件判断只覆盖 "no_concat_gae_last"/"_first"，
#     漏掉了当前使用的 "no_concat_gae" → value_mask 根本没有被计算
#     dp_critic.py 的 select_keys 不包含 value_mask → 即使设了也传不到 critic
#   后果：critic 的 ~98% 训练位置 target=-100 → critic 收敛到输出 ≈ -100
#   证据：critic/returns/mean ≈ -99（贯穿整个 v10 训练），vf_explained_var 无意义
#   已修复：
#     ray_trainer.py: 添加 "no_concat_gae" 到 value_mask 条件
#     dp_critic.py:   条件性添加 value_mask 到 select_keys
#
# [次要问题 3] Entropy 爆炸 step 193-196（已自愈）
#   premature_done_penalty=-0.3 封堵 done-spam 后，模型探索随机长输出
#   entropy 1.6→5.0，step 200 已回到 3.92，KL loss 拉回中
#   不影响重新开始新实验
#
# ============================================================
# v11 核心修复
# ============================================================
# [修复 1] MAX_TURNS: 15 → 50（最核心）
#   50 steps × 1 action/step = 50 总动作，模型现在有机会到达目标
#   max_episode_steps=50（env_config_single_action_256.yaml）与 max_turns=50 精确对齐
#
# [修复 2] 256×256 图像（防 ViT OOM）
#   50 turns × ~342 patches/image = 17,100 patches → attention ≈ 9 GB
#   vs 512×512: 50 turns × ~1369 patches = 68,450 patches → attention ≈ 150 GB (OOM)
#   使用 env_config_single_action_256.yaml
#
# [修复 3] Critic bug 已在代码层面修复（见上方说明）
#   - ray_trainer.py: "no_concat_gae" 现在正确计算 value_mask
#   - dp_critic.py: value_mask 正确传入 critic 训练
#   预期效果：critic/returns/mean 从 -99 回归到真实 return 范围（约 -0.1 ~ +0.5）
#
# [修复 4] MAX_TRAJECTORY_LENGTH: 16000 → 28000
#   50 turns × (256×256 image ~342tok + text ~200tok + response ~200tok) ≈ 37,100
#   但 Qwen2.5-VL 使用动态分辨率，256×256 实际约 ~260 tok/image
#   50 turns × ~660tok/turn = 33,000；加 system+initial ≈ 35,000
#   设 28000（保守估计，防 OOM；actual token budget will clip if needed）
#   注：window_size=1 减少 KV-cache 压力（只看上一轮观测）
#
# [继承 v10 稳定性设计]
#   premature_done_penalty=-0.3：防 done-spam（env_config_single_action_256.yaml 已添加）
#   entropy_coeff=0.008, kl_loss_coef=0.015：防崩溃
#   critic_warmup=40：新训练让 critic 先对齐真实 return
#   RESUME_MODE=disable：从头训练（v10 的 critic 完全失效，不值得续训）
#
# ============================================================
# v11 参数总结
# ============================================================
#   env_config:     env_config_single_action_256.yaml（256px，premature_done_penalty=-0.3）
#   MAX_TURNS:      ★ 50（核心修复，was 15）
#   MAX_TRAJECTORY_LENGTH: ★ 28000（was 16000）
#   WINDOW_SIZE:    1（防 OOM，只看上一轮观测）
#   MAX_RESPONSE_LENGTH: 512（保留，防截断）
#   RESUME:         disable（从头训练）
#   entropy_coeff:  0.008
#   kl_loss_coef:   0.015
#   critic_warmup:  40
#   cliprange_value: 0.8
#   critic_lr:      2e-5
#   grad_clip:      0.5
#   actor_lr:       1e-6
#   N_TRAJECTORY:   1（OOM 防护）
#   TRAIN_BATCH_SIZE: 24
# ============================================================

EXPERIMENT_NAME="v11_max_turns50_v2"  # _v2 suffix: fresh run after reward-shaping rebalance (potential 1->5, vis 0.3->0.05, success 1->5) + critic_warmup 40->10. Old wandb run preserved.
ENV_CONFIG="env_config_single_action_256.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

# === 从头训练：v10 critic 完全失效，不值得续训 ===
RESUME_MODE="disable"

# === 稳定性参数（继承 v10/v9）===
ENTROPY_COEFF="0.008"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.015"
TEMPERATURE="0.9"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.5

# === Critic（继承 v10）===
CRITIC_LR="2e-5"
CRITIC_WARMUP=10  # reward rebalance (potential 1->5, vis 0.3->0.05, success 1->5): warmup shortened to start actor updates sooner
CLIPRANGE_VALUE="0.8"

# === 梯度 ===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === ★ 核心修复：MAX_TURNS 15 → 50，扩大轨迹长度 ===
MAX_TURNS=50               # ★ 15 → 50：给模型足够的步数到达目标
WINDOW_SIZE=1              # 防 OOM：只看上一轮观测（50 turns × full context 会 OOM）
MAX_TRAJECTORY_LENGTH=28000  # ★ 16000 → 28000：支持 50 轮对话的 token 预算
MAX_RESPONSE_LENGTH=512    # 保留 512 防截断 </action>
MAX_PROMPT_LENGTH=2048     # 不变

# === 数据参数（继承 v10）===
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8
N_TRAJECTORY=1

# === PPO mini-batch（继承 v10）===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === 训练参数 ===
SAVE_FREQ=80
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 算法参数（继承 v10）===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"
