# =============================================================================
# v22_3scenes — 3-scene training (Plan B 多场景外推验证)
# =============================================================================
# 动机:
#   单场景 (0267_840790) 上的实验结论可能不外推. 在 v22 系列定下 winner 之前,
#   先用 3 个场景的小规模 curriculum 检验关键结论是否成立:
#     - no_farm 比 dual 稳定?
#     - cosine LR 提升峰值?
#     - 训练曲线还是 step40 达峰 step120 崩塌? 还是因为数据量 3x 而推迟?
#
#   3 个训练场景:
#     - 0267_840790 (原 in-domain, 259 train tasks)
#     - 0299_840574 (258 tasks)
#     - 0342_840398 (235 tasks)
#   总训练任务 752 (vs 单场景 259, ≈3x).
#   验证: 仍用原 0267 的 19 个 val 任务 (in-domain, 锚定历史曲线对比)
#         + Plan A 的 19 个 OOD 任务 (来自其他 6 个场景, 与训练场景不重叠)
#
# 用 v21_no_farm_only_stab 的同款参数作为 baseline (因为它当前最稳定).
#
# 与 v21_no_farm_only_stab 的差异:
#   ★ ENV_CONFIG: env_config_v19_no_farm.yaml -> env_config_v22_3scenes.yaml
#                  (差异: jsonl_path 改成 3 场景合并版, train_size 259->752)
#   ★ TOTAL_STEPS: 2000 -> 1000  (数据量 3x, 但每个 epoch 也 3x 步, 期望相似总 epoch 数)
#   ★ TEST_FREQ: 20 -> 30  (val 现在 38 prompts × 4 rollouts = 152 records, 慢一些)
#   + OOD val (Plan A)
#
# 期望:
#   - 学习曲线整体延后 (峰值可能落在 step ~80-120 而非 step 40)
#   - in-domain tsucc_b4 vs 单场景 v21_no_farm_stab 应该接近 (相同 0267 val)
#     如果显著低于 0.895 → 多场景训练干扰单场景表现
#     如果与之相当 → 多场景训练不削弱单场景能力, 同时获得泛化
#   - OOD tsucc_b4 显著高于 v22_cosine_no_farm 单场景的 OOD 曲线
#     (因为多场景训练直接见过多种几何)
#
# 启动 (注意需要充足 GPU 内存, 因为 set_scene 切换 3 个 PLY):
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v22_3scenes.sh \
    > v22_3scenes.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v22_3scenes"
ENV_CONFIG="env_config_v22_3scenes.yaml"   # ★ 3 场景合并 jsonl
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor (= v21_no_farm_only_stab) ===
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

# === Data ===
N_TRAJECTORY=1
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=12
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=150
TEST_FREQ=30                # ★ 20 -> 30 (val 现在更慢: 38 prompts vs 19)
TOTAL_STEPS=1000            # ★ 2000 -> 1000 (短期实验, 看到 ~3 个 epoch 即可)
VAL_BEFORE_TRAIN="False"

# === Algorithm ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
