# v9: 从头训练，同时预防 v6（action单调化）和 v7（生成崩溃）两大问题
#
# ============================================================
# v6 / v7 / v8 失败根因分析
# ============================================================
# [v6 问题] action 单调化（turn_right 76.3%）
#   根因：KL loss 把策略锚定在 v5 遗留的 ref（已深度偏向 turn_right）
#         entropy_coeff=0.01 不足以克服 turn_right attractor
#   → 从头训练可天然绕开此问题：ref = 干净的 Qwen2.5-VL-3B，无任何 action 偏向
#     KL 此时是"保护性"的，阻止模型发展出新的偏向，而不是"锁定旧偏向"
#
# [v7 问题] 生成能力全面崩溃（0/10 指令跟随，空输出/重复循环）
#   根因：entropy_coeff=0.05（baseline 的 50 倍）+ kl_loss_coef=0.005（极低）
#         → 40 步内策略飞离原始分布，生成能力崩溃
#   → 从头训练不能重蹈覆辙：entropy_coeff 保持温和（0.008），KL 系数适中
#
# [v8 问题] 继承了 v7 的所有错误参数，从崩溃状态续训无法恢复
#
# ============================================================
# v9 从头训练设计原则
# ============================================================
# [防 v6：从根源消除 action 偏向]
#   - ref model = 原始 Qwen2.5-VL-3B（无 turn_right 偏向）
#   - KL loss 保持适中（kl_loss_coef=0.015）：锚定干净 ref，天然防止偏向发展
#   - entropy_coeff=0.008：适度高于 baseline（0.001），温和维持多样性
#     不需要激进设置，因为 ref 本身就是无偏的
#
# [防 v7：稳定的双重约束保证不崩溃]
#   - entropy_coeff=0.008 << v7 的 0.05：不会破坏生成能力
#   - kl_loss_coef=0.015  >> v7 的 0.005：KL 足够强，策略不会飞离分布
#   - 经验法则：entropy_coeff * 10 < kl_loss_coef 时策略趋于稳定
#     此处 0.008 * 10 = 0.08 < 0.15（满足约束，安全边际 2x）
#
# [从头训练特有的 critic 处理]
#   - critic 从随机初始化出发 → 必然初始过估
#   - critic_warmup=40：前 40 步只训 critic，让其先对齐真实 return 规模
#   - cliprange_value=0.8：允许 critic 做大步修正（新初始化时必要）
#   - critic_lr=2e-5：比 v6 的 1e-5 稍高，加速收敛新初始化的 critic
#
# ============================================================
# v9 参数总结
# ============================================================
#   RESUME:           disable（从头训练）
#   entropy_coeff:    0.008   (温和多样性，防 v6；远低于 0.05，防 v7)
#   kl_loss_coef:     0.015   (适中 KL，锚定干净 ref，防单调化且不崩溃)
#   temperature:      0.9     (稳定采样)
#   top_p:            0.92
#   critic_lr:        2e-5    (加速对齐新初始化 critic)
#   critic_warmup:    40      (从头训练必须：先对齐 critic 再更新 actor)
#   cliprange_value:  0.8     (允许 critic 大步修正)
#   max_traj_length:  26000   (v8 的修复，降低截断率)
#   grad_clip:        0.5     (稳定)
#   actor_lr:         1e-6    (不变)
# ============================================================

EXPERIMENT_NAME="v9_stable_diversity"
ENV_CONFIG="env_config_balanced.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

# === 从头训练：不加载任何历史 checkpoint ===
RESUME_MODE="disable"

# === 防 v6：温和 entropy + KL 锚定干净 ref，天然无偏向 ===
ENTROPY_COEFF="0.008"      # 温和，远低于 v7 的 0.05；高于 baseline 维持多样性
USE_KL_LOSS="True"         # 保持 KL：ref 是干净模型，KL 是保护性约束
KL_LOSS_COEF="0.015"       # 适中：防止 action 偏向形成；远高于 v7 的 0.005

# === 防 v7：保持稳定生成，不飞离分布 ===
TEMPERATURE="0.9"          # 稳定采样
TOP_P="0.92"               # 轻微收窄

# === Critic 从头初始化处理 ===
CRITIC_LR="2e-5"           # 稍高于 v6，加速对齐随机初始化 critic
CRITIC_WARMUP=40           # 必须：前 40 步纯 critic 学习，再启动 actor 更新
CLIPRANGE_VALUE="0.8"      # 允许 critic 做大步修正（新初始化时必要）

# === 轨迹容量（v8 的修复）===
MAX_TRAJECTORY_LENGTH=26000    # 降低截断率

# === 梯度稳定 ===
GRAD_CLIP="0.5"
ACTOR_LR="1e-6"

# === 训练参数 ===
SAVE_FREQ=20
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Rollout 参数 ===
MAX_TURNS=12
WINDOW_SIZE=5
MINI_BATCH_SIZE=6

# === 数据参数 ===
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=512

# === 算法参数 ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
