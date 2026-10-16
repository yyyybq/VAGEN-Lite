# v6: 在 v5 成功基础上解决 entropy 坍塌 + critic 失效
# v5 成果: score -0.2→+1.06, success 3%→44%, valid 70%→100%, entropy 0.54→0.023
# v5 问题:
#   1) entropy 从 step 60 开始 <0.1, step 97 仅 0.023 → 探索能力几乎丧失
#   2) critic vf_explained_var 始终为负(-0.3 ~ -6) → critic 预测 value 差
#   3) 只跑了 97/2000 步就中断
# 策略:
#   1) 从 v5 step 80 checkpoint 恢复（保留已学到的 valid action 能力）
#   2) 开启 KL loss (use_kl_loss=True) 约束策略不偏离 ref 太远，间接维持 entropy
#      - KL penalty 比固定 entropy_coeff 更自适应：策略远离 ref 时自动加大惩罚
#      - kl_loss_coef=0.02 适中（太小无效，太大阻碍学习）
#   3) entropy_coeff 适度提高到 0.01：v5 的 0.004 不足以抵抗后期坍塌
#      结合 KL loss 可以双管齐下：KL 防止策略漂移，entropy 维持输出多样性
#   4) critic 关键改进：
#      - param_offload=False + optimizer_offload=False（已在 v5 设置）确保 critic 不掉显存
#      - critic_lr 降回 1e-5：v5 的 2e-5 让 critic grad_norm 波动剧烈(11~110)
#      - cliprange_value 从 0.5 降到 0.2：约束 value 预测的更新幅度，避免 value 跳变
#      - critic warmup 继续保持 25 步
#   5) 温度 0.9（比 v5 的 0.8 略高），给采样更多随机性辅助探索

EXPERIMENT_NAME="v6_antientropy_collapse"
ENV_CONFIG="env_config_balanced.yaml"

# === 从 v6 自身 checkpoint 恢复（原始从 v5 step 80 开始，已训练到 step 140） ===
# RESUME_MODE="checkpoints/vagen_active_spatial/v5_entropy_balanced_0410/global_step_80"  # 原始设定
RESUME_MODE="auto"  # 自动从 latest checkpoint (step 140) 恢复

# === 防 entropy 坍塌：双保险 ===
ENTROPY_COEFF="0.01"       # 0.004→0.01  提高 entropy bonus 抵抗坍塌
USE_KL_LOSS="True"         # 开启 KL loss，约束策略不偏离 ref
KL_LOSS_COEF="0.02"        # KL loss 系数（适中）

# === 采样：略提高随机性 ===
TEMPERATURE="0.9"          # 0.8→0.9  给探索更多空间
TOP_P="0.95"               # 保持 0.95

# === Critic 改进 ===
CRITIC_LR="1e-5"           # 2e-5→1e-5  降低 lr 减少 grad_norm 波动
CLIPRANGE_VALUE="0.2"      # 0.5→0.2  约束 value 更新幅度，减少 value 跳变

# === 梯度稳定 ===
GRAD_CLIP="0.5"            # 保持 0.5

# === Critic warmup ===
CRITIC_WARMUP=0             # 恢复训练时不需要额外 warmup（已经 warmup 过）

# === 训练参数 ===
SAVE_FREQ=20
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"   # 恢复训练无需再 val（已在之前验证）
