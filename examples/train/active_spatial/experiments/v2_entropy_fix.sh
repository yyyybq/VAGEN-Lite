# v2: 防 entropy 坍塌 + Critic 预训练 + 梯度稳定
# 改动: entropy 10x, critic_warmup=10, grad_clip 0.5, temp 0.9
# 基于 yr5xf955 分析：entropy 0.41→0.02 坍塌, grad spike 137.9, vf_exp_var 始终负
EXPERIMENT_NAME="v2_entropy_fix_0331"
ENV_CONFIG="env_config_balanced.yaml"

ENTROPY_COEFF="0.01"       # 0.001→0.01  防止 entropy 坍塌
CRITIC_WARMUP=10           # 0→10        先训练 Critic 10 步
GRAD_CLIP="0.5"            # 1.0→0.5     抑制梯度 spike
TEMPERATURE="0.9"          # 0.7→0.9     增强采样多样性
TOP_P="0.98"               # 0.95→0.98
SAVE_FREQ=20               # 50→20       更频繁保存 checkpoint
# GPU_MEM_UTIL 使用默认 0.5（配合 enforce_eager=False 不会 OOM）
