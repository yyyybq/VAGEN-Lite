# v7: 解决 action 单调化 + critic 过估 两大核心问题
#
# ============================================================
# v6 核心诊断（step 141-161）
# ============================================================
# [好消息]
#   - entropy 维持在 0.024~0.046（KL+entropy 双保险生效，未再坍塌）
#   - valid rate: 100%（格式能力完整保持）
#   - success rate: 18~62%，均值约 43%；score 峰值 1.447（step 147）
#   - response length 从 673→451（thinking 变简洁，正常现象）
#
# [严重问题 1] Action 单调化（最核心）
#   v6 action 分布：turn_right 76.3%，move_backward 13.1%，turn_left 10.4%
#   → 模型在"原地转圈"，探索能力几乎丧失
#   → 偶尔转到正确方向才算 success，本质是随机碰运气
#   根因：KL loss 把策略钉在 ref（ref 也是 turn_right 主导的 v5 产物），
#         entropy_coeff=0.01 不足以打破这个强 attractor
#   修复：大幅提高 entropy_coeff (0.01→0.05) + 大幅降低 kl_loss_coef (0.02→0.005)
#        让策略有足够自由度 escape turn_right attractor
#
# [严重问题 2] Critic 系统性过估（10~20倍）
#   critic/values/mean: 0.47~1.33  vs  returns/mean: 0.04~0.10
#   → critic 预测的 value 比实际 return 高 10~20 倍
#   → vf_explained_var 始终为 -1.2 ~ -9.4（比随机还差）
#   根因：critic score head 随机初始化偏高，实际 return 因 GAE discount 极小
#   修复：
#     (a) critic_warmup=40：让 critic 先纯学 value，再更新 actor
#     (b) critic_lr=3e-5：aggressive 追赶实际 return 规模（短期需要快速收敛）
#     (c) cliprange_value=0.5→1.0：放宽 value clip，允许 critic 做大步修正
#
# [次要问题] 运行在 step 162 处中断（job timeout）
#   → 建议 RESUME_MODE=auto 从 step 160 续训
#
# ============================================================
# v7 改变总结（vs v6）
# ============================================================
#   entropy_coeff:   0.01  → 0.05  (+400%，强制 action 多样性)
#   kl_loss_coef:    0.02  → 0.005 (-75%，解锁策略逃离 turn_right attractor)
#   temperature:     0.9   → 1.05  (稍提高采样随机性辅助探索)
#   top_p:           0.95  → 0.9   (略收窄防止过度随机)
#   critic_lr:       1e-5  → 3e-5  (加速 critic 追赶实际 return 规模)
#   critic_warmup:   0     → 40    (续训也给 critic 重对齐窗口)
#   cliprange_value: 0.2   → 0.8   (允许 critic 大步修正过估)
#   actor_lr:        1e-6  (保持不变，actor 学习率够用)
#   grad_clip:       0.5   (保持不变)
#   use_kl_loss:     True  (保持，但大幅降低系数)
# ============================================================

EXPERIMENT_NAME="v7_action_diversity"
ENV_CONFIG="env_config_balanced.yaml"

# === 续训：从 v6 step 160 checkpoint ===
RESUME_MODE="auto"

# === 核心修复 1：打破 action 单调化 ===
ENTROPY_COEFF="0.05"       # 0.01→0.05  大幅提高，强制 action 多样性
USE_KL_LOSS="True"         # 保持 KL，但大幅降低系数
KL_LOSS_COEF="0.005"       # 0.02→0.005  解锁策略逃离 turn_right attractor

# === 采样：更随机 ===
TEMPERATURE="1.05"         # 0.9→1.05  辅助探索
TOP_P="0.90"               # 0.95→0.90  略收窄防过度随机

# === 核心修复 2：修复 critic 过估 ===
CRITIC_LR="3e-5"           # 1e-5→3e-5  加速追赶实际 return 规模
CRITIC_WARMUP=40           # 0→40  续训也给 critic 重新对齐窗口
CLIPRANGE_VALUE="0.8"      # 0.2→0.8  允许 critic 大步修正

# === 梯度稳定 ===
GRAD_CLIP="0.5"            # 保持

# === 训练参数 ===
SAVE_FREQ=20
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"
