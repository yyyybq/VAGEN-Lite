# v4: 在 v2 基础上增大 Critic LR + success reward
# 假设: Critic 学习太慢 (vf_exp_var<0), success reward 不够强
EXPERIMENT_NAME="v4_stronger_critic"
ENV_CONFIG="env_config_balanced.yaml"

# 继承 v2 的改进
ENTROPY_COEFF="0.01"
CRITIC_WARMUP=10
GRAD_CLIP="0.5"
TEMPERATURE="0.9"
TOP_P="0.98"

# v4 独有
CRITIC_LR="2e-5"           # 1e-5→2e-5  加速 Critic 收敛
