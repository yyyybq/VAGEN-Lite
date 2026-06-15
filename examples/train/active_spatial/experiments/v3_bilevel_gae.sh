# v3: 在 v2 基础上尝试 bi_level_gae (专为多轮交互设计)
# 假设: masked_gae 对跨 turn 的 credit assignment 不够好
EXPERIMENT_NAME="v3_bilevel_gae"
ENV_CONFIG="env_config_balanced.yaml"

# 继承 v2 的改进
ENTROPY_COEFF="0.01"
CRITIC_WARMUP=10
GRAD_CLIP="0.5"
TEMPERATURE="0.9"
TOP_P="0.98"

# v3 独有: 换 advantage estimator
ADV_ESTIMATOR="bi_level_gae"
HIGH_LEVEL_GAMMA="0.99"    # 跨 turn 折扣更保守
