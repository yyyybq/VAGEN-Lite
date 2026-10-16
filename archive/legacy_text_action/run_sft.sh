#!/usr/bin/env bash
# Fine-tune a VLM on RoboCasa LeRobot demos via LLaMA-Factory.
#
# Works for BOTH:
#   --model-path $QWEN_PATH                  # original Qwen2.5-VL-3B-Instruct
#   --model-path $ACTIVE_SPATIAL_HF_CKPT     # HF actor dir (config.json + safetensors + tokenizer)
#
# The vision tower stays frozen; LLM + projector are full-FT.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH=""
DATA_ROOT=""
OUTPUT_DIR="${REPO_ROOT}/checkpoints/robocasa_sft"
TASK=""
TASK_SET=""
SPLIT="target"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
STRIDE=4
HORIZON=8
MAX_SAMPLES=0
SKIP_CONVERT=0

usage() {
  cat <<'EOF'
Usage: run_sft.sh --model-path PATH --data-root PATH [options]

Required:
  --model-path PATH     HF model dir or hub id (base Qwen or Active Spatial actor)
  --data-root PATH      Root containing LeRobot v2 RoboCasa demos

Optional:
  --output-dir PATH     Training output (default: checkpoints/robocasa_sft)
  --task NAME           Task or comma list
  --task-set NAME       atomic_seen | composite_seen | seen
  --split NAME          pretrain | target (default: target)
  --gpus LIST           CUDA_VISIBLE_DEVICES (default: 0)
  --stride N            Frame stride (default: 4)
  --horizon N           Actions per SFT turn (default: 8)
  --max-samples N       Cap converted samples (0 = all)
  --skip-convert        Reuse existing parquet in output-dir/lf_data
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --task-set) TASK_SET="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --horizon) HORIZON="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --skip-convert) SKIP_CONVERT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[error] --model-path is required (base Qwen or Active Spatial HF actor dir)." >&2
  usage
  exit 2
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[warn] model-path does not exist on disk; treating as hub id: ${MODEL_PATH}" >&2
else
  if [[ -d "${MODEL_PATH}" ]]; then
    if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
      echo "[error] ${MODEL_PATH} is not an HF dir (missing config.json)." >&2
      exit 2
    fi
  fi
fi

LF_DATA_DIR="${OUTPUT_DIR}/lf_data"
IMAGE_DIR="${OUTPUT_DIR}/images"
LF_CONFIG="${SCRIPT_DIR}/lf_qwen25vl_sft.yaml"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# vagen-lite: datasets==5.0.0, trl==0.9.6. Use LLaMA-Factory v0.9.3 + skip version gate.
export DISABLE_VERSION_CHECK=1

if [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[error] no python/python3 on PATH" >&2
  exit 2
fi

mkdir -p "${LF_DATA_DIR}" "${IMAGE_DIR}" "${OUTPUT_DIR}"

if [[ "${SKIP_CONVERT}" -eq 0 ]]; then
  if [[ -z "${DATA_ROOT}" ]]; then
    echo "[error] --data-root is required unless --skip-convert is set." >&2
    exit 2
  fi
  CONVERT_ARGS=(
    --data-root "${DATA_ROOT}"
    --split "${SPLIT}"
    --stride "${STRIDE}"
    --horizon "${HORIZON}"
    --max-samples "${MAX_SAMPLES}"
    --output-dir "${LF_DATA_DIR}"
    --image-dir "${IMAGE_DIR}"
  )
  [[ -n "${TASK}" ]] && CONVERT_ARGS+=(--task "${TASK}")
  [[ -n "${TASK_SET}" ]] && CONVERT_ARGS+=(--task-set "${TASK_SET}")
  echo "[convert] LeRobot -> sharegpt"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_lerobot_to_sft.py" "${CONVERT_ARGS[@]}"
fi

cp "${SCRIPT_DIR}/lf_dataset_info.json" "${LF_DATA_DIR}/dataset_info.json"
if [[ ! -f "${LF_DATA_DIR}/robocasa_sft.parquet" && ! -f "${LF_DATA_DIR}/robocasa_sft.jsonl" ]]; then
  echo "[error] converted dataset missing under ${LF_DATA_DIR}" >&2
  exit 2
fi

NPROC=$(awk -F',' '{print NF}' <<<"${GPUS}")
ACCEL_CFG="${SCRIPT_DIR}/lf_accelerate_fsdp.yaml"
if [[ ! -f "${ACCEL_CFG}" ]]; then
cat > "${ACCEL_CFG}" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
  fsdp_transformer_layer_cls_to_wrap: Qwen2_5_VLDecoderLayer
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: ${NPROC}
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
fi

echo "[train] model=${MODEL_PATH}"
echo "[train] dataset_dir=${LF_DATA_DIR}"
echo "[train] output_dir=${OUTPUT_DIR}"
echo "[train] gpus=${GPUS} nproc=${NPROC}"

# v0.9.3 CLI pops the `train` subcommand, then reads yaml as argv[1].
# Multi-GPU: llamafactory-cli auto-launches torchrun when it sees >1 CUDA device.
if command -v llamafactory-cli >/dev/null 2>&1; then
  llamafactory-cli train "${LF_CONFIG}" \
    model_name_or_path="${MODEL_PATH}" \
    dataset_dir="${LF_DATA_DIR}" \
    output_dir="${OUTPUT_DIR}"
elif "${PYTHON_BIN}" -c "import llamafactory" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m llamafactory.cli train "${LF_CONFIG}" \
    model_name_or_path="${MODEL_PATH}" \
    dataset_dir="${LF_DATA_DIR}" \
    output_dir="${OUTPUT_DIR}"
else
  echo "[error] llamafactory is not importable in ${PYTHON_BIN}." >&2
  echo "        pip install -e ${REPO_ROOT}/third_party/LLaMA-Factory --no-deps" >&2
  exit 2
fi

echo "[done] HF checkpoint at ${OUTPUT_DIR}"
