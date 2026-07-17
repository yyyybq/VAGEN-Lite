#!/bin/bash
set -x

# =============================================================================
# VAGEN-Lite 通用 PPO 训练入口 - 通过实验配置文件驱动
# （对应旧版 VAGEN/scripts/examples/vagen_base/active_spatial/run_experiment.sh）
#
# 用法 (从 VAGEN-Lite 根目录运行):
#   nohup bash examples/train/active_spatial/run_experiment.sh \
#       experiments/v10_per_action_dense.sh > v10_per_action_dense.log 2>&1 &
#
# 与旧版 VAGEN run_experiment.sh 的核心差异:
#   ┌──────────────────────────────┬──────────────────────────────────────┐
#   │ 旧 VAGEN                     │ VAGEN-Lite                           │
#   ├──────────────────────────────┼──────────────────────────────────────┤
#   │ vagen.trainer.main_ppo       │ vagen.main_ppo (Hydra)               │
#   │ parquet 数据集               │ YAML 环境配置                        │
#   │ rollout: vllm                │ rollout: sglang (async, no-concat)   │
#   │ masked_gae                   │ no_concat_gae                        │
#   │ rollout_manager.*            │ env YAML + agent YAML                │
#   │ data.max_trajectory_length   │ rollout.max_num_batched_tokens       │
#   └──────────────────────────────┴──────────────────────────────────────┘
#
# 实验配置文件 (experiments/v*.sh) 与旧版完全相同，本脚本负责参数转换。
# =============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASEDIR="$( cd "$SCRIPT_DIR/../../.." && pwd )"   # VAGEN-Lite 根目录
PYTHON=/scratch/by2593/miniconda3/envs/vagen-lite/bin/python
# 在 run_experiment.sh 顶部加:
export TMPDIR=/tmp
export PYTHONMULTIPROCESSINGTEMPDIR=/tmp   # 双保险
# ========================= BASELINE DEFAULTS =========================
# 这些是 VAGEN-Lite 版的基线默认值，与旧 VAGEN run_experiment.sh 对齐
# 实验配置文件通过重新赋值来 override

EXPERIMENT_NAME="unnamed_$(date +%m%d_%H%M)"
ENV_CONFIG="env_config_balanced.yaml"
NUM_TRAIN_GPUS=4
RENDERING_GPU=4
USE_GPU_HOLDER=true

RENDER_MODE="local"   # "local" = 本机渲染GPU; "remote" = 远程渲染服务
RENDER_HOST=""
RENDER_PORT="8777"

# 模型（与旧 VAGEN 默认一致：3B；7B 可在实验配置文件中 override）
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# Actor
ACTOR_LR="1e-6"
ENTROPY_COEFF="0.001"
GRAD_CLIP="1.0"
USE_KL_LOSS="False"
KL_LOSS_COEF="0.001"

# Critic
CRITIC_LR="1e-5"
CRITIC_WARMUP=0
CLIPRANGE_VALUE="0.5"

# Rollout（sglang，不同于旧版 vllm）
TEMPERATURE="0.7"
TOP_P="0.95"
# ── Validation sampling (verl 默认是 greedy: do_sample=False / temp=0 / n=1)
#    Greedy + 固定 val jsonl seeds 会让 val 轨迹完全确定 → val_reward 一直是常数
#    (我们就遇到了 v17/v18 val_reward 卡在 63.5 / 26.0 的现象)。
#    这里改成有温度的采样 + 多轨迹平均，让 val 真正反映 policy distribution。
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.8}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
VAL_N="${VAL_N:-4}"
GPU_MEM_UTIL="0.5"
# 注：TP_SIZE 在 VAGEN-Lite sglang 中默认 1（每 GPU 独立实例）
# 旧版 vllm 的 TP=4 在 sglang 中不需要，因为 sglang 在 no-concat 模式下更高效
TP_SIZE=1

# Data
TRAIN_BATCH_SIZE=24
VAL_BATCH_SIZE=8
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=512

# 轨迹参数（MAX_TRAJECTORY_LENGTH → rollout.max_num_batched_tokens）
MAX_TRAJECTORY_LENGTH=16000   # 用于设置 max_num_batched_tokens
MAX_TURNS=12
WINDOW_SIZE=5    # 旧版 masked_gae window_size；VAGEN-Lite no_concat_gae 中不使用，保留兼容

# PPO mini-batch
MINI_BATCH_SIZE=8        # 旧版 rollout_manager.mini_batch_size；VAGEN-Lite 中不使用
PPO_MINI_BATCH_SIZE=12   # actor_rollout_ref.actor.ppo_mini_batch_size
N_TRAJECTORY=1

# Trainer
SAVE_FREQ=20
TEST_FREQ=20
TOTAL_STEPS=2000
VAL_BEFORE_TRAIN="False"

# Algorithm
ADV_ESTIMATOR="masked_gae"   # 自动映射为 no_concat_gae
HIGH_LEVEL_GAMMA="0.95"      # 映射为 algorithm.gamma
KL_COEF="0.001"
LAM="0.95"

# Resume
RESUME_MODE="auto"
RESUME_CKPT_PATH=""

# ========================= LOAD EXPERIMENT CONFIG =========================
if [ -z "$1" ]; then
    echo "ERROR: 请指定实验配置文件"
    echo "用法: bash run_experiment.sh experiments/v10_per_action_dense.sh"
    echo ""
    echo "可用实验:"
    ls "$SCRIPT_DIR/experiments/"*.sh 2>/dev/null | while read f; do
        name=$(basename "$f")
        desc=$(head -3 "$f" | grep "^# " | head -1 | sed 's/^# //')
        printf "  %-35s %s\n" "$name" "$desc"
    done
    exit 1
fi

EXPERIMENT_CONFIG="$1"
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    EXPERIMENT_CONFIG="$SCRIPT_DIR/experiments/$1"
fi
if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: 找不到实验配置: $1"
    exit 1
fi

echo "Loading experiment config: $EXPERIMENT_CONFIG"
source "$EXPERIMENT_CONFIG"

# ========================= PARAMETER MAPPING =========================
# masked_gae → no_concat_gae（VAGEN-Lite 的等效实现）
if [ "$ADV_ESTIMATOR" = "masked_gae" ]; then
    ADV_ESTIMATOR_LITE="no_concat_gae"
else
    ADV_ESTIMATOR_LITE="$ADV_ESTIMATOR"
fi

# ========================= ENVIRONMENT SETUP =========================
GPU_LIST=$(seq -s, 0 $((NUM_TRAIN_GPUS - 1)))

if [ "$RENDER_MODE" = "remote" ]; then
    if [ -z "$RENDER_HOST" ]; then
        echo "ERROR: RENDER_MODE=remote 时必须设置 RENDER_HOST"
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
else
    export CUDA_VISIBLE_DEVICES="${GPU_LIST},${RENDERING_GPU}"
    export RENDERING_GPU_ID=${RENDERING_GPU}
fi

export PYTHONUNBUFFERED=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager
export RAY_DEDUP_LOGS=0
export RAY_enable_metrics_collection=false
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export GS_RENDERER_VERBOSE=0
export ACTIVE_SPATIAL_ENV_VERBOSE=0
export PATH="/scratch/by2593/miniconda3/envs/vagen-lite/bin:$PATH"

# ---------------------------------------------------------------------------
# Redirect wandb / HuggingFace / tmp caches off $HOME to avoid "Disk quota
# exceeded" errors (home has a small per-user quota; /scratch has TBs).
# This includes wandb artifact *staging* directory which is controlled by
# WANDB_DATA_DIR (see wandb.env.get_data_dir -> get_staging_dir).
# ---------------------------------------------------------------------------
export WANDB_CACHE_ROOT="/scratch/by2593/.cache/wandb"
mkdir -p "${WANDB_CACHE_ROOT}/data" "${WANDB_CACHE_ROOT}/cache" "${WANDB_CACHE_ROOT}/artifacts" "${WANDB_CACHE_ROOT}/config"
export WANDB_DATA_DIR="${WANDB_CACHE_ROOT}/data"           # staging dir for artifact uploads
export WANDB_CACHE_DIR="${WANDB_CACHE_ROOT}/cache"
export WANDB_ARTIFACT_DIR="${WANDB_CACHE_ROOT}/artifacts"
export WANDB_CONFIG_DIR="${WANDB_CACHE_ROOT}/config"
export TMPDIR="/scratch/by2593/tmp"
mkdir -p "${TMPDIR}"

# Redirect framework caches off $HOME (home has a tight quota). Keep heavyweight
# HF/pip caches on /scratch, but put JIT build caches on node-local /tmp: flashinfer
# uses file locks during vLLM startup, and shared scratch/NFS can produce stale
# handles when multiple experiments compile kernels at the same time.
export XDG_CACHE_HOME="/scratch/by2593/.cache"
mkdir -p "${XDG_CACHE_HOME}"
JIT_CACHE_ROOT="/tmp/${USER}/vagen_jit/${EXPERIMENT_NAME}_$$"
export FLASHINFER_WORKSPACE_BASE="${JIT_CACHE_ROOT}/flashinfer"
export TRITON_CACHE_DIR="${JIT_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JIT_CACHE_ROOT}/torchinductor"
export TORCH_EXTENSIONS_DIR="${JIT_CACHE_ROOT}/torch_extensions"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TORCH_EXTENSIONS_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}"

# 注意：不设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 它与 sglang/vLLM CuMemAllocator 不兼容

EXPERIMENT_DIR="${BASEDIR}/exps/vagen_active_spatial/${EXPERIMENT_NAME}"
mkdir -p "$EXPERIMENT_DIR"

# ========================= PRINT CONFIG =========================
echo "=============================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Config:     $(basename $EXPERIMENT_CONFIG)"
echo "Model:      $MODEL_PATH"
echo "Env:        $ENV_CONFIG"
echo "Train GPUs: $GPU_LIST ($NUM_TRAIN_GPUS GPUs)"
if [ "$RENDER_MODE" = "remote" ]; then
    echo "Render:     REMOTE  ${RENDER_HOST}:${RENDER_PORT}"
else
    echo "Render GPU: $RENDERING_GPU (local)"
fi
echo "----------------------------------------------"
echo "Actor LR=$ACTOR_LR  Critic LR=$CRITIC_LR"
echo "Entropy=$ENTROPY_COEFF  Grad Clip=$GRAD_CLIP"
echo "Temp=$TEMPERATURE  Top-p=$TOP_P"
echo "Critic Warmup=$CRITIC_WARMUP"
echo "Save freq=$SAVE_FREQ  Total steps=$TOTAL_STEPS"
echo "ADV=$ADV_ESTIMATOR → $ADV_ESTIMATOR_LITE  gamma=$HIGH_LEVEL_GAMMA  lam=$LAM"
echo "KL_loss=$USE_KL_LOSS  KL_coef=$KL_LOSS_COEF  Cliprange_V=$CLIPRANGE_VALUE"
echo "MAX_TURNS=$MAX_TURNS  MAX_RESPONSE_LENGTH=$MAX_RESPONSE_LENGTH"
echo "Resume=$RESUME_MODE"
echo "=============================================="

# ========================= GENERATE ENV YAMLs =========================
# 将旧格式 env_config_*.yaml 转换为 VAGEN-Lite 的 envs[] YAML 格式
ENV_CONFIG_PATH="$SCRIPT_DIR/$ENV_CONFIG"
TRAIN_YAML="${EXPERIMENT_DIR}/train.yaml"
VAL_YAML="${EXPERIMENT_DIR}/val.yaml"

if [ "$RENDER_MODE" = "remote" ]; then
    RENDER_BACKEND="client"
    CLIENT_URL="ws://${RENDER_HOST}:${RENDER_PORT}/render/interiorgs"
else
    RENDER_BACKEND="local"
    CLIENT_URL=""
fi

$PYTHON - <<PYEOF
import json, os, yaml, sys

with open("${ENV_CONFIG_PATH}") as f:
    cfg = yaml.safe_load(f)

# 找到第一个 env 条目（旧格式：env1: {env_config: {...}, train_size: N, test_size: M}）
env_key = list(cfg.keys())[0]
env_entry = cfg[env_key]
env_config = dict(env_entry.get("env_config", {}))
train_size = env_entry.get("train_size", 259)
test_size = env_entry.get("test_size", 19)

# Optional in-domain val override:
# 1) ID_VAL_JSONL / ID_VAL_N_ENVS: use a custom ID val jsonl directly.
# 2) ID_VAL_DELTA_BOOST_N: build an ID val jsonl with at least N delta_control tasks.
id_val_jsonl = "${ID_VAL_JSONL:-}"
id_val_n = int("${ID_VAL_N_ENVS:-0}")
id_val_delta_boost_n = int("${ID_VAL_DELTA_BOOST_N:-0}")
id_val_include_types = [t.strip() for t in "${ID_VAL_INCLUDE_TASK_TYPES:-}".split(",") if t.strip()]
id_val_exclude_types = {t.strip() for t in "${ID_VAL_EXCLUDE_TASK_TYPES:-}".split(",") if t.strip()}

def _task_type(entry: dict) -> str:
    t = entry.get("task_type", "")
    if t:
        return str(t)
    desc = str(entry.get("task_description", "")).lower()
    if "closer" in desc or "farther" in desc:
        return "delta_control"
    return "unknown"

def _allowed_task_type(entry: dict) -> bool:
    t = _task_type(entry)
    if id_val_include_types and t not in id_val_include_types:
        return False
    if id_val_exclude_types and t in id_val_exclude_types:
        return False
    return True

if (id_val_delta_boost_n > 0 or id_val_include_types or id_val_exclude_types) and not id_val_jsonl:
    src_jsonl = env_config.get("jsonl_path", "")
    if src_jsonl and os.path.isfile(src_jsonl):
        with open(src_jsonl) as f:
            all_items = [json.loads(line) for line in f if line.strip()]

        base_start = int(train_size)
        base_end = min(len(all_items), int(train_size + test_size))
        base_items = all_items[base_start:base_end]

        if id_val_include_types or id_val_exclude_types:
            kept_items = [e for e in base_items if _allowed_task_type(e)]
            target_total = len(base_items)
            need_fill = max(0, target_total - len(kept_items))
            seen = {json.dumps(e, sort_keys=True, ensure_ascii=False) for e in kept_items}
            extras = []
            for e in all_items:
                if not _allowed_task_type(e):
                    continue
                k = json.dumps(e, sort_keys=True, ensure_ascii=False)
                if k in seen:
                    continue
                extras.append(e)
                seen.add(k)
                if len(extras) >= need_fill:
                    break
            base_items = kept_items + extras[:need_fill]
            print(
                f"[INFO] ID val task filter: include={id_val_include_types or 'ALL'} "
                f"exclude={sorted(id_val_exclude_types) or 'NONE'} -> total={len(base_items)}"
            )

        delta_items = [e for e in base_items if _task_type(e) == "delta_control"]
        non_delta_items = [e for e in base_items if _task_type(e) != "delta_control"]

        need_extra = max(0, id_val_delta_boost_n - len(delta_items))
        extra_delta = []
        if need_extra > 0:
            seen = {json.dumps(e, sort_keys=True, ensure_ascii=False) for e in base_items}
            for e in all_items:
                if _task_type(e) != "delta_control":
                    continue
                k = json.dumps(e, sort_keys=True, ensure_ascii=False)
                if k in seen:
                    continue
                extra_delta.append(e)
                seen.add(k)
                if len(extra_delta) >= need_extra:
                    break

        # Keep total ID val size unchanged by replacing tail non-delta items.
        final_delta = delta_items + extra_delta[:need_extra]
        keep_non_delta_n = max(0, len(base_items) - len(final_delta))
        boosted_items = non_delta_items[:keep_non_delta_n] + final_delta

        id_val_jsonl = "${EXPERIMENT_DIR}/val_id_delta_boost.jsonl"
        with open(id_val_jsonl, "w") as f:
            for e in boosted_items:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        id_val_n = len(boosted_items)

        final_delta_n = sum(1 for e in boosted_items if _task_type(e) == "delta_control")
        if id_val_delta_boost_n > 0:
            print(
                f"[INFO] ID delta boost enabled: target={id_val_delta_boost_n}, "
                f"actual={final_delta_n}, total_id_n={id_val_n}, file={id_val_jsonl}"
            )
        else:
            print(f"[INFO] ID val override built: total_id_n={id_val_n}, file={id_val_jsonl}")
    else:
        print(f"[WARN] ID delta boost skipped: source jsonl not found: {src_jsonl}")

if id_val_jsonl and id_val_n <= 0 and os.path.isfile(id_val_jsonl):
    id_val_n = sum(1 for _ in open(id_val_jsonl) if _.strip())

# Optional training data filter: TRAIN_EXCLUDE_TASK_TYPES
# If set, load the training jsonl, strip out the excluded task types, write a
# filtered copy into EXPERIMENT_DIR, and update env_config + train_size.
train_exclude_types = {t.strip() for t in "${TRAIN_EXCLUDE_TASK_TYPES:-}".split(",") if t.strip()}
if train_exclude_types:
    src_train_jsonl = env_config.get("jsonl_path", "")
    if src_train_jsonl and os.path.isfile(src_train_jsonl):
        with open(src_train_jsonl) as f:
            all_train = [json.loads(l) for l in f if l.strip()]
        filtered_train = [e for e in all_train if _task_type(e) not in train_exclude_types]
        filtered_path = "${EXPERIMENT_DIR}/train_filtered.jsonl"
        with open(filtered_path, "w") as f:
            for e in filtered_train:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        removed = len(all_train) - len(filtered_train)
        print(
            f"[INFO] TRAIN_EXCLUDE_TASK_TYPES={sorted(train_exclude_types)}: "
            f"removed {removed} entries ({removed/len(all_train)*100:.1f}%), "
            f"kept {len(filtered_train)}, writing to {filtered_path}"
        )
        env_config["jsonl_path"] = filtered_path
        train_size = len(filtered_train)
    else:
        print(f"[WARN] TRAIN_EXCLUDE_TASK_TYPES set but source jsonl not found: {src_train_jsonl}")

# 覆盖渲染配置
env_config["gpu_device"] = ${RENDERING_GPU}
if "${RENDER_BACKEND}" == "client":
    env_config["render_backend"] = "client"
    env_config["client_url"] = "${CLIENT_URL}"
else:
    env_config["render_backend"] = "local"

train_yaml = {
    "envs": [{
        "name": "ActiveSpatial",
        "n_envs": train_size,
        "data_source": "active_spatial",
        "seed": [0, train_size],
        "max_turns": ${MAX_TURNS},
        "response_length_per_turn": ${MAX_RESPONSE_LENGTH},
        "config": env_config,
    }]
}

id_env_config = dict(env_config)
id_env_n = test_size
id_env_seed = [train_size, train_size + test_size]
id_env_seed_list = None
id_env_source = "active_spatial"
if id_val_jsonl:
    id_env_config["jsonl_path"] = id_val_jsonl
    id_env_n = id_val_n if id_val_n > 0 else test_size
    id_env_seed = [0, id_env_n]
    id_env_seed_list = list(range(id_env_n))
    id_env_source = "active_spatial_id_override"
    print(f"[INFO] ID val override: {id_val_jsonl} (n_envs={id_env_n})")

id_val_spec = {
    "name": "ActiveSpatial",
    "n_envs": id_env_n,
    "data_source": id_env_source,
    "seed": id_env_seed,
    "max_turns": ${MAX_TURNS},
    "response_length_per_turn": ${MAX_RESPONSE_LENGTH},
    "config": id_env_config,
}
if id_env_seed_list is not None:
    id_val_spec["seed_list"] = id_env_seed_list

val_envs = [id_val_spec]

# Optional OOD val env (Plan A): if OOD_VAL_JSONL is set, add a second val env
# that points to a different jsonl_path (different scenes).
# Records in validation/{step}.jsonl will be ordered: in-domain first (test_size*VAL_N),
# then OOD (test_size_ood*VAL_N).
ood_val_jsonl = "${OOD_VAL_JSONL:-}"
ood_val_n = ${OOD_VAL_N_ENVS:-0}
if ood_val_jsonl and ood_val_n > 0:
    ood_env_config = dict(env_config)
    ood_env_config["jsonl_path"] = ood_val_jsonl
    val_envs.append({
        "name": "ActiveSpatial",
        "n_envs": ood_val_n,
        "data_source": "active_spatial_ood",
        "seed": [0, ood_val_n],  # index 0..ood_val_n-1 in the OOD jsonl
        "max_turns": ${MAX_TURNS},
        "response_length_per_turn": ${MAX_RESPONSE_LENGTH},
        "config": ood_env_config,
    })
    print(f"[INFO] OOD val enabled: {ood_val_jsonl} (n_envs={ood_val_n})")

# Multi-split OOD val: if OOD_SPLITS_DIR is set, add each ood_*.jsonl as a
# separate val env with its own data_source tag (e.g. "active_spatial_ood_scene").
import os as _os, glob as _glob
ood_splits_dir = "${OOD_SPLITS_DIR:-}"
ood_splits_n = int("${OOD_SPLITS_N_ENVS:-20}")
if ood_splits_dir and _os.path.isdir(ood_splits_dir):
    for ood_file in sorted(_glob.glob(_os.path.join(ood_splits_dir, "ood_*.jsonl"))):
        split_name = _os.path.basename(ood_file).replace(".jsonl", "")  # e.g. "ood_scene"
        n_items = sum(1 for _ in open(ood_file))
        n_envs_this = min(n_items, ood_splits_n)
        ood_cfg = dict(env_config)
        ood_cfg["jsonl_path"] = ood_file
        val_envs.append({
            "name": "ActiveSpatial",
            "n_envs": n_envs_this,
            "data_source": f"active_spatial_{split_name}",
            "seed": [0, n_envs_this],
            "max_turns": ${MAX_TURNS},
            "response_length_per_turn": ${MAX_RESPONSE_LENGTH},
            "config": ood_cfg,
        })
        print(f"[INFO] OOD split: {split_name} ({n_envs_this}/{n_items} envs) <- {ood_file}")

val_yaml = {"envs": val_envs}

with open("${TRAIN_YAML}", "w") as f:
    yaml.dump(train_yaml, f, default_flow_style=False)
with open("${VAL_YAML}", "w") as f:
    yaml.dump(val_yaml, f, default_flow_style=False)

print(f"[INFO] Generated train YAML: ${TRAIN_YAML}  (n_envs={train_size})")
print(f"[INFO] Generated val YAML:   ${VAL_YAML}  (n_envs_total={sum(e['n_envs'] for e in val_envs)})")
print(f"[INFO] env_config keys: {list(env_config.keys())}")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: 生成 env YAML 失败"
    exit 1
fi

# ========================= AGENT LOOP CONFIG =========================
# 生成动态 agent loop config YAML（控制 max_turns 和 response_length_per_turn）
AGENT_YAML="${BASEDIR}/vagen/configs/agent_no_concat_active_spatial.yaml"
# 使用已有的 agent_no_concat_active_spatial.yaml，max_turns 由 env YAML 覆盖

# ========================= CLEANUP =========================
HOLDER_PIDS=()
cleanup() {
    echo "Cleaning up..."
    for pid in "${HOLDER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -P $$ -f "gpu_holder.py" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

# ========================= GPU HOLDER (optional) =========================
HOLDER_LOG_DIR="$EXPERIMENT_DIR/logs/gpu_holders"
mkdir -p "$HOLDER_LOG_DIR"

if [ "$USE_GPU_HOLDER" = true ]; then
    # 渲染卡 holder（仅本地渲染模式）
    # 注：渲染卡不分配内存（HOLDER_MEM_FRAC=0.0），仅维持 SM 利用率
    # 原因：多个 AgentLoopWorker 同时在 GPU 上加载 GS 场景（每个 ~10-12 GiB），
    #       若预分配 0.35 × 140 GiB ≈ 49 GiB 则 GPU OOM。
    if [ "$RENDER_MODE" != "remote" ]; then
        HOLDER_GPU=$RENDERING_GPU HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
            $PYTHON "$SCRIPT_DIR/gpu_holder.py" \
            > "$HOLDER_LOG_DIR/holder_gpu${RENDERING_GPU}.log" 2>&1 &
        HOLDER_PIDS+=($!)
        echo "GPU Holder started for GPU $RENDERING_GPU, PID=$!, log=$HOLDER_LOG_DIR/holder_gpu${RENDERING_GPU}.log"
    fi

    # 训练卡 holders（keep-warm）
    for GPU_ID in $(seq 0 $((NUM_TRAIN_GPUS - 1))); do
        HOLDER_GPU=$GPU_ID HOLDER_MEM_FRAC=0.0 HOLDER_TARGET=75 \
            $PYTHON "$SCRIPT_DIR/gpu_holder.py" \
            > "$HOLDER_LOG_DIR/holder_gpu${GPU_ID}.log" 2>&1 &
        HOLDER_PIDS+=($!)
        echo "GPU Holder started for GPU $GPU_ID, PID=$!, log=$HOLDER_LOG_DIR/holder_gpu${GPU_ID}.log"
    done

    sleep 5
    echo "--- GPU Holder Health Check ---"
    ALL_HEALTHY=true
    for pid in "${HOLDER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  PID $pid: ALIVE"
        else
            echo "  PID $pid: DEAD! Check log: $HOLDER_LOG_DIR/"
            ALL_HEALTHY=false
        fi
    done
    if [ "$ALL_HEALTHY" = false ]; then
        echo "WARNING: Some GPU holders failed to start."
    fi
    echo "-------------------------------"
fi

# ========================= PPO TRAINING =========================
# 用法: vagen.main_ppo --config-path --config-name=vagen_multiturn
# 与旧 vagen.trainer.main_ppo 的关键区别:
#   1. rollout: sglang async (not vllm)
#   2. adv_estimator: no_concat_gae (not masked_gae)
#   3. trainer.concat_multi_turn=False
#   4. 无 rollout_manager.* 参数（由 env YAML + agent YAML 替代）
$PYTHON -m vagen.main_ppo \
    --config-path="${BASEDIR}/vagen/configs" \
    --config-name="vagen_multiturn" \
    data.train_files="${TRAIN_YAML}" \
    data.val_files="${VAL_YAML}" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    \
    algorithm.adv_estimator=$ADV_ESTIMATOR_LITE \
    algorithm.gamma=$HIGH_LEVEL_GAMMA \
    algorithm.lam=$LAM \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=$N_TRAJECTORY \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.temperature=$VAL_TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.top_p=$VAL_TOP_P \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$VAL_DO_SAMPLE \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TRAJECTORY_LENGTH \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_YAML}" \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.optim.lr=$CRITIC_LR \
    critic.model.path=$MODEL_PATH \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.cliprange_value=$CLIPRANGE_VALUE \
    \
    trainer.concat_multi_turn=False \
    trainer.critic_warmup=$CRITIC_WARMUP \
    trainer.logger=['console','wandb'] \
    trainer.project_name='vagen_active_spatial' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_TRAIN_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_STEPS \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.resume_mode=$RESUME_MODE \
    trainer.default_local_dir="${EXPERIMENT_DIR}/checkpoints" \
    trainer.validation_data_dir="${EXPERIMENT_DIR}/validation" \
    trainer.rollout_data_dir="${EXPERIMENT_DIR}/rollout_data" \
    trainer.log_val_generations=8 \
    ${EXTRA_OVERRIDES:-} \
    2>&1 | tee "${EXPERIMENT_DIR}/train.log"
