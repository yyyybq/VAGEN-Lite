# =============================================================================
# v24_groupadv_100scenes — GroupAdv (N=4) + 94 个训练场景 + OOD val
# =============================================================================
# 动机:
#   截至 v21-v23 系列分析（2026-05-24）:
#
#   核心发现:
#     - v23_groupadv_3scenes (N=4 + 752 envs, 3 场景) 是迄今唯一同时满足
#       "OOD 稳定 ≥0.71 + critic 持续上升（2.1→3.9）+ 无崩塌"的实验.
#     - 对照: v21_groupadv (N=4 + 259 envs, 1 场景) 在 step 180 仍崩至 0.242.
#     - 结论: GroupAdv 有效的前提是足够的场景多样性 (envs 数量是稳定性关键).
#
#   本实验假设:
#     3 场景 (752 envs) → 94 场景 (28662 envs) 能进一步:
#       1. 提升 OOD 泛化 (训练分布更广)
#       2. 彻底消除 PPO 崩塌 (大量不同场景使 advantage 估计更稳健)
#       3. 在 OOD val 上实现 b4 > 0.9
#
# 与 v23_groupadv_3scenes 的差异:
#   ★ ENV_CONFIG: env_config_v22_3scenes.yaml -> env_config_v24_100scenes.yaml
#   ★ train envs: 752 (3 场景) -> 28662 (94 场景)
#   ★ TEST_FREQ: 30 -> 50  (每步采样数更多, 单步更慢, 适当降低 val 频率)
#   ★ TOTAL_STEPS: 2000    (维持不变)
#
# 注意:
#   - OOD val 场景 (6 个) 已从训练集中剔除, val_ood_v1.jsonl 仍然有效.
#   - 数据文件: output_100scenes/train_100scenes_no_ood.jsonl (28662 tasks)
#     生成命令:
#       python3 -c "
#         import json
#         ood={'0276_840780','0367_840260','0229_840306','0266_840789',
#              '0351_840366','0240_840881'}
#         rows=[json.loads(l) for l in open('output_100scenes/train.jsonl')]
#         rows=[r for r in rows if r['scene_id'] not in ood]
#         [open('output_100scenes/train_100scenes_no_ood.jsonl','w')
#          .write(json.dumps(r)+'\n') for r in rows]
#       "
#
# 启动:
: <<'RUN'
cd /scratch/by2593/project/Active_Spatial/VAGEN-Lite && \
  nohup bash examples/train/active_spatial/run_experiment.sh \
    examples/train/active_spatial/experiments/v24_groupadv_100scenes.sh \
    > v24_groupadv_100scenes.log 2>&1 &
echo "PID: $!"
RUN

EXPERIMENT_NAME="v24_groupadv_100scenes"
ENV_CONFIG="env_config_v24_100scenes.yaml"     # ★ 94 scenes, 28662 tasks
NUM_TRAIN_GPUS=4
RENDERING_GPU=4

RESUME_MODE="disable"

# === Actor ===
ENTROPY_COEFF="0.002"
USE_KL_LOSS="True"
KL_LOSS_COEF="0.10"                # 维持 v23 的 0.10（已验证底线）
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
CRITIC_WARMUP=60                   # 维持 v23 的 60
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

# === Data (GroupAdv: 同 prompt 4 rollout) ===
N_TRAJECTORY=4                     # GroupAdv
TRAIN_BATCH_SIZE=12                # 12 * 4 = 48 轨迹/step
VAL_BATCH_SIZE=8

# === PPO mini-batch ===
PPO_MINI_BATCH_SIZE=8
MINI_BATCH_SIZE=8

# === Trainer ===
SAVE_FREQ=80                       # ★ 改为 80
TEST_FREQ=50                       # ★ 3scenes 是 30, 100scenes 每步更慢, 改 50
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# === Algorithm (PPO + no_concat_gae) ===
ADV_ESTIMATOR="masked_gae"
HIGH_LEVEL_GAMMA="0.95"
KL_COEF="0.001"
LAM="0.95"

# === Plan A: OOD validation ===
export OOD_VAL_JSONL="/scratch/by2593/project/Active_Spatial/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v1.jsonl"
export OOD_VAL_N_ENVS=19
