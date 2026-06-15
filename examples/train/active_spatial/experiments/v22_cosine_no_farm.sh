# =============================================================================
# v22_cosine_no_farm — cosine LR × no_farm yaml (组合两个 v21 的局部成功)
# =============================================================================
# 动机:
#   v21 两路实验各自揭示了一个方向:
#     v21_lrdecay        (dual yaml + cosine LR):  峰值 tsucc_b4=0.9474 ✓ 但 step60 已跌 0.8421
#     v21_no_farm_only_stab (no_farm + v20补丁):   峰值 tsucc_b4=0.8947 ✓ step60 保持稳定,
#                                                    mean@4 step60=0.605 (持续上升)
#
#   cosine LR 单独无法阻止崩塌 — step60 时 cosine 才运行了 2%, LR 仍≈1e-6,
#   所以 lrdecay 的 step60 下跌不是 LR 原因, 而是 dual yaml 本身的不稳定性.
#
#   → 核心假设: dual yaml 是崩塌根源; cosine LR 只是提升峰值. 把两者解耦:
#     no_farm yaml (稳定奖励) + cosine LR (更高峰值) = 期望: 峰值 ≥0.947 + 稳定性 ≥ v21_no_farm_stab
#
# yaml: env_config_v19_no_farm.yaml  (= v21_no_farm_only_stab)
# sh:   v21_no_farm_only_stab 的所有参数 + cosine LR EXTRA_OVERRIDES
#
# 与 v21_no_farm_only_stab 的唯一差异:
#   ★ lr_scheduler_type:  constant -> cosine  (EXTRA_OVERRIDES 注入)
#   ★ lr_warmup_steps_ratio:  0 -> 0.10
#   ★ min_lr_ratio:  - -> 0.1  (终端 LR = 1e-7)
#   ★ TOTAL_STEPS:   2000 -> 500  (cosine 周期对齐预期的 plateau; 500步内完整衰减)
#
# 与 v21_lrdecay 的差异:
#   ★ ENV_CONFIG: no_farm_dual -> no_farm  (去掉 dual reward)
#
# 期望:
#   - step 0–50:   LR warmup → 1e-6
#   - step 50–500: cosine decay → 1e-7 (step40 时 LR≈1e-6 ≈ 峰值, step80 时 LR≈7e-7)
#   - 峰值 tsucc_b4 ≥ 0.947 (cosine 提升效果)
#   - step60 tsucc_b4 ≥ 0.895 (no_farm 稳定效果, 不再崩跌)
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v22_cosine_no_farm.sh \
    > v22_cosine_no_farm.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v22_cosine_no_farm"
ENV_CONFIG="env_config_v19_no_farm.yaml"   # ★ 与 v21_lrdecay 不同: no dual
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.002"          # v20 stability patch
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"            # v20 stability patch
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
CRITIC_WARMUP=60               # v20 stability patch
CLIPRANGE_VALUE="0.5"          # v20 stability patch

# === Optim ===
GRAD_CLIP="0.3"
ACTOR_LR="1e-6"                # peak LR; cosine 终端 = 1e-6 * 0.1 = 1e-7

# === 轨迹 ===
MAX_TURNS=12
WINDOW_SIZE=1
MAX_TRAJECTORY_LENGTH=18000
MAX_RESPONSE_LENGTH=384
MAX_PROMPT_LENGTH=2048

# === Data ===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=20
TOTAL_STEPS=500          # ★ 与 no_farm_stab 不同: 500 (cosine 一个完整周期)
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === LR schedule (通过 run_experiment.sh 的 EXTRA_OVERRIDES 注入) ===
export EXTRA_OVERRIDES="\
actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.10 \
actor_rollout_ref.actor.optim.min_lr_ratio=0.1"

# === Plan A: OOD validation (周期性外推泛化曲线) ===
# 添加一个第二验证集, 19 个任务来自 6 个其他场景, task_type 分布与 in-domain 一致.
# 验证 jsonl 中前 76 条 = in-domain (19 prompts × 4 rollouts), 后 76 条 = OOD.
# 这样可以在每次 val 时同时看到:
#   - in-domain  tsucc_b4 (是否过拟合到 0267 这一个场景)
#   - OOD        tsucc_b4 (泛化到不同房间布局的真实能力)
# 不增加训练时间, 只多花一份 val 时间 (~+30s/val cycle).
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
