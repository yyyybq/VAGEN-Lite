# v5: 修复 v2 entropy 爆炸 + critic 失效
# v2 问题诊断:
#   - entropy 0.9→4.16 单调上升，entropy_coeff=0.01 矫枉过正
#   - 思维退化为 gibberish（温度+entropy 正则过强）
#   - vf_explained_var 始终大负（-1.9 到 -10.6），critic 完全失效
#   - reward -0.2→-3.4 持续下降，success 无上升趋势
#   - 动作退化为纯 turn，丢失 move_forward
# 策略:
#   1) entropy_coeff 折中到 0.004（0.001 坍塌, 0.01 爆炸）
#   2) temperature 降回 0.8，top_p 降回 0.95，减少采样随机性
#   3) critic_lr 2x + warmup 25 步，给 critic 更多学习机会
#   4) 保留 grad_clip=0.5 抑制梯度 spike

EXPERIMENT_NAME="v5_entropy_balanced_0410"
ENV_CONFIG="env_config_balanced.yaml"

# === Entropy: 在 0.001(坍塌) 和 0.01(爆炸) 之间折中 ===
ENTROPY_COEFF="0.004"      # 0.01→0.004  抑制 entropy 爆炸，但仍高于 baseline 防坍塌

# === 采样: 降低随机性，防止 gibberish ===
TEMPERATURE="0.8"          # 0.9→0.8   减少输出随机性
TOP_P="0.95"               # 0.98→0.95 回到 baseline，收紧采样

# === Critic: 加强学习能力 ===
CRITIC_LR="2e-5"           # 1e-5→2e-5  加速 critic 收敛
CRITIC_WARMUP=25           # 10→25      给 critic 更充分的预训练

# === 梯度稳定 (继承 v2) ===
GRAD_CLIP="0.5"            # 保持 0.5，抑制梯度 spike

# === 监控频率 ===
SAVE_FREQ=20               # 频繁保存，便于早期发现问题
TEST_FREQ=20               # 同步测试频率
