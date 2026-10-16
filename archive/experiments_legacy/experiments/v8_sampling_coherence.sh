# v8: 消除采样歧义 + 降低截断率
#
# ============================================================
# v7 现状诊断（step 162 中断）
# ============================================================
# [采样配置歧义]
#   - GenerationConfig 初始化时只传 (temperature, top_p, top_k)
#   - do_sample 作为独立参数传递，导致 Transformers 警告：
#     "do_sample=False but temperature/top_p/top_k are set"
#   - 实际生成正确用的是 do_sample=True，但配置不一致易引入混淆
#   - 修复：GenerationConfig 显式传入 do_sample=do_sample，确保一致性
#
# [截断率过高]
#   v4 之后 max_trajectory_length 降至 22000，导致截断频繁：
#   - step 1 就有 8 次 Left truncation 警告
#   - 每次截断都需要清理不完整的 image token block，成本高
#   修复：放宽 max_trajectory_length 到 26000，并降低 ANYRES_MAX_SUBIMAGES 
#        从 9 到 7，减少每条轨迹的 vision token 膨胀
#
# ============================================================
# v8 改变总结（vs v7）
# ============================================================
# [代码修复]
#   - GenerationConfig 显式传入 do_sample=True（消除警告与配置歧义）
#
# [轨迹与视觉token管理]
#   max_trajectory_length: 22000 → 26000  (+18%, 提高容纳空间)
#   ANYRES_MAX_SUBIMAGES:      9  → 7     (-22%, 降低token膨胀率)
#   预期效果：截断率 ↓50%+ (from 8 per step to ~3-4)
#
# [保持 v7 的成果]
#   entropy_coeff:   0.05  (action 多样性)
#   kl_loss_coef:    0.005 (策略自由度)
#   temperature:     1.05  (采样随机性)
#   top_p:           0.90  
#   critic_lr:       3e-5  (value 准确度)
#   critic_warmup:   40
#   cliprange_value: 0.8
#
# ============================================================

EXPERIMENT_NAME="v8_sampling_coherence"
ENV_CONFIG="env_config_balanced.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

# === 续训：从 v7 step 162 checkpoint（或最新的v7 ckpt）===
RESUME_MODE="auto"

# === 代码修复：消除采样歧义 ===
# (GenerationConfig 现在显式传入 do_sample=True)
TEMPERATURE="1.05"         # 保持：1.05（辅助探索）
TOP_P="0.90"               # 保持：0.90
# do_sample=True 现在在 GenerationConfig 中显式设置，与参数一致

# === 减少截断率 ===
MAX_TRAJECTORY_LENGTH=26000    # 22000 → 26000  (+18%, 提高轨迹容纳空间)
ANYRES_MAX_SUBIMAGES=7         # 9 → 7  (-22%, 降低每条轨迹的视觉token膨胀)
# 其他 rollout 参数保持
MAX_TURNS=12
WINDOW_SIZE=5
MINI_BATCH_SIZE=6
RM_MAX_PROMPT_LENGTH=8192
SI_TOKEN_LEN=729

# === 保持 v7 的 action 多样性修复 ===
ENTROPY_COEFF="0.05"       # 保持：0.05
USE_KL_LOSS="True"
KL_LOSS_COEF="0.005"       # 保持：0.005

# === 保持 v7 的 critic 修复 ===
CRITIC_LR="3e-5"           # 保持：3e-5
CRITIC_WARMUP=40           # 保持：40
CLIPRANGE_VALUE="0.8"      # 保持：0.8

# === 梯度稳定 ===
GRAD_CLIP="0.5"            # 保持
ACTOR_LR="1e-6"            # 保持

# === 训练参数 ===
SAVE_FREQ=20
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === 其他基线参数（from v7/v6 baseline）===
# Actor
ACTOR_OPTIMIZER_OFFLOAD="False"

# Critic
CRITIC_GRAD_CLIP="1.0"     # 保持
CRITIC_OPTIMIZER_OFFLOAD="False"

# Data
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512

# Trainer
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"

# Algorithm
KL_LOSS_TYPE="mse"
