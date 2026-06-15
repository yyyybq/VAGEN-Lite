# =============================================================================
# v21_groupadv — 增加组内 rollout 数 (N_TRAJECTORY 1→4) + GRPO 可行性分析
# =============================================================================
# 动机:
#   评测里 best_of_4 vs tsucc_b4 (mean over 4) 一直有 ~13 pt 的 gap:
#     v20 step40: best@4=44.95  vs  mean@4=31.71  -> 4 条里有好坏分化
#     v19_no_farm step40: 47.45 vs 33.65          -> 同样 ~14pt
#   这意味着 "同一 prompt 的 4 次 rollout 自然存在显著回报差异",
#   但训练时 N_TRAJECTORY=1, 即 每个 prompt 只采一条, 优势完全由 GAE+critic 估,
#   无法利用 "同 prompt 的兄弟轨迹相对回报" 这一强信号.
#
#   做法 (本实验默认 = 安全路线):
#     N_TRAJECTORY: 1 -> 4
#     保持 ADV_ESTIMATOR=masked_gae (no_concat_gae) + critic
#     critic 仍然学 value baseline;同 prompt 4 条 rollout 之间的回报方差自动
#     进入 batch, GAE 估计更稳, 同时 policy 看到 best/worst 对比 -> 期望提升
#     峰值 best@4.
#
#   为控显存与 step 时间, 同步调:
#     TRAIN_BATCH_SIZE: 24 -> 12   (12*4=48 条轨迹/step, 比原来 24 多一倍)
#     PPO_MINI_BATCH_SIZE: 12 -> 8
#     其它一切等同 v20_winner (含全部稳定性补丁)
#
# -----------------------------------------------------------------------------
# GRPO 切换分析 ("能不能把 PPO 改成 GRPO?"):
#
#   verl 内置 GRPO: compute_grpo_outcome_advantage(token_level_rewards,
#     response_mask, index=non_tensor_batch["uid"], norm_adv_by_std_in_grpo=...)
#   核心机制: 对每个 uid 分组, 用 token-level reward 求和得到 outcome reward,
#     在组内 (z-score) 标准化得到 advantage, 不需要 critic.
#
#   在 VAGEN-Lite (multi-turn no-concat) 下要谨慎的点:
#     (1) uid 语义:
#         no-concat 模式下, 同一条轨迹的不同 turn 是不同 sample.
#         如果 uid 是 per-turn 的, GRPO 会把同一轨迹的不同 turn 当作组员,
#         outcome reward 在 turn 间被错误平均 -> 信号失真.
#         需要确认 vagen rollout 里 uid 是按 "trajectory" 写入而非 per-turn,
#         或显式把 uid 改成 group_idx (即同 prompt 的 N 条轨迹共享一个 uid,
#         同一轨迹的不同 turn 也共享).
#     (2) critic 关闭:
#         verl.need_critic() 在 adv_estimator != GAE 时默认关闭 critic,
#         但本仓库 adv_estimator='no_concat_gae' 时实际靠 critic.enable 默认.
#         切 GRPO 时需显式: critic.enable=False (或不传 critic 模型/优化器).
#         运行脚本里 CRITIC_LR/CRITIC_WARMUP/CLIPRANGE_VALUE 都会变成无效.
#     (3) ray_trainer.py:1524 附近的 compute_value_mask(batch) 只在
#         adv_estimator 以 'no_concat_gae' 开头时触发. 切 GRPO 时会绕过,
#         需要确认 GRPO 这条分支也走我们期望的 response_mask 计算.
#     (4) outcome reward 来源:
#         GRPO 直接对 token_level_rewards.sum(-1) 做组内归一. 我们当前 reward
#         整形 (no_farm / dual ori) 是 per-step 给的, 求和后能正确反映轨迹回报,
#         看起来兼容, 但要看是否有 KL/penalty 已经掺进 token_level_rewards.
#
#   结论:
#     "原地切 GRPO" 需要先做 3 个 unit-test:
#        a. 打印一个 batch 的 non_tensor_batch["uid"], 确认 per-trajectory.
#        b. 跑 1 step 看 critic.enable=False 是否走得通 (verl 容许 None critic).
#        c. 比较 GRPO advantage 与人工 groupwise z-score 一致.
#     在没验证前不作为默认. 因此本 sh 默认仍是 PPO + masked_gae + N=4,
#     已经能拿到 "groupwise variance" 大部分收益.
#     下方 EXTRA_OVERRIDES 块给出 GRPO 切换的实验性配置 (注释掉),
#     等 a/b/c 通过后可解开作为 v21_groupadv_grpo 跑.
# =============================================================================
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v21_groupadv.sh \
    > v21_groupadv.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v21_groupadv"
ENV_CONFIG="env_config_v19_no_farm_dual.yaml"   # 与 v20_winner 一致
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (与 v20_winner 完全一致) ===
ENTROPY_COEFF="0.002"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"
TEMPERATURE="0.8"
TOP_P="0.92"
TP_SIZE=4
GPU_MEM_UTIL=0.4

# === Validation sampling ===
VAL_TEMPERATURE="0.8"
VAL_TOP_P="0.95"
VAL_DO_SAMPLE="True"
VAL_N="4"

# === Critic ===
CRITIC_LR="2e-5"
CRITIC_WARMUP=60
CLIPRANGE_VALUE="0.5"

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data (★ 核心改动) ===
N_TRAJECTORY=4                  # ★ 1 -> 4 (同 prompt 采 4 条 rollout)
TRAIN_BATCH_SIZE=12             # ★ 24 -> 12 (12*4=48 轨迹/step, 比原 24 多一倍)
VAL_BATCH_SIZE=8

# === PPO mini-batch (★ 同步缩) ===
PPO_MINI_BATCH_SIZE=8           # ★ 12 -> 8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm (默认安全:PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# -----------------------------------------------------------------------------
# 实验性 GRPO 切换 (默认注释; 需先做 uid 单元测试再启用)
# -----------------------------------------------------------------------------
# export EXTRA_OVERRIDES="
#   algorithm.adv_estimator=grpo
#   algorithm.norm_adv_by_std_in_grpo=True
#   algorithm.use_kl_in_reward=False
#   critic.enable=False
# "
# 注: 启用 GRPO 时还需要把上面的 ADV_ESTIMATOR 改成不会被 run_experiment.sh
#     映射成 no_concat_gae 的值 (例如直接删 ADV_ESTIMATOR= 这行, 让脚本
#     使用 hydra 默认; 或修改 run_experiment.sh 让 EXTRA_OVERRIDES 覆盖优先).
