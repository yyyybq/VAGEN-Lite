#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Qwen2.5-VL-3B-Instruct SFT on active_spatial 5k (LLaMA-Factory).
#
# Pipeline:
#   1. Convert SFT JSONL → parquet (messages + images columns).
#   2. Stage parquet + dataset_info.json into ${LF_DATA_DIR}.
#   3. (optional) Clone & install LLaMA-Factory into a dedicated conda env.
#   4. Launch full-parameter SFT (ViT frozen) with FSDP across 4 GPUs.
#
# Train GPUs: 0,1,2,3 (GPU 4 is reserved for RL rollout / data gen).
#
# Outputs land in ${OUTPUT_DIR}; the final HF checkpoint is directly usable as
# verl RL warmstart (set `model.partial_pretrain=${OUTPUT_DIR}` in the verl
# trainer YAML).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT="/scratch/by2593/project/Active_Spatial/VAGEN-Lite"
SFT_DIR="${PROJECT_ROOT}/data_gen/active_spatial_sft"
SFT_JSONL="${SFT_DIR}/output_100scenes_5k/sft_data.jsonl"
SFT_IMG_BASE="${SFT_DIR}/output_100scenes_5k"

LF_DATA_DIR="${SFT_DIR}/lf_data"           # Will hold dataset_info.json + parquets
LF_CONFIG="${SFT_DIR}/lf_qwen25vl_3b_sft.yaml"

OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/sft_qwen25vl_3b_5k"
LF_REPO_DIR="${PROJECT_ROOT}/third_party/LLaMA-Factory"

CONDA_BASE="/scratch/by2593/miniconda3"
LF_ENV_NAME="llamafactory"

# ── Conda env bootstrap ──────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${LF_ENV_NAME}"; then
    echo "[setup] Creating conda env ${LF_ENV_NAME} (python=3.10)…"
    conda create -y -n "${LF_ENV_NAME}" python=3.10
fi
conda activate "${LF_ENV_NAME}"

# ── LLaMA-Factory install (idempotent) ───────────────────────────────────────
if ! command -v llamafactory-cli >/dev/null 2>&1; then
    echo "[setup] Installing LLaMA-Factory…"
    mkdir -p "$(dirname "${LF_REPO_DIR}")"
    if [[ ! -d "${LF_REPO_DIR}" ]]; then
        git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LF_REPO_DIR}"
    fi
    pip install -e "${LF_REPO_DIR}[torch,metrics,qwen]"
    pip install qwen-vl-utils                   # required by Qwen2.5-VL template
    pip install pyarrow pandas pillow           # parquet + image IO
fi

# ── Step 1: JSONL → Parquet ──────────────────────────────────────────────────
mkdir -p "${LF_DATA_DIR}"

PARQUET_THINK="${LF_DATA_DIR}/active_spatial_sft.parquet"
PARQUET_NOTHINK="${LF_DATA_DIR}/active_spatial_sft_no_think.parquet"

if [[ ! -f "${SFT_JSONL}" ]]; then
    echo "[error] SFT JSONL not found: ${SFT_JSONL}"
    echo "        Run the 5k generation pipeline first."
    exit 1
fi

echo "[convert] ${SFT_JSONL} → ${PARQUET_THINK} (+ no-think variant)"
python "${SFT_DIR}/convert_to_qwen25vl_sft.py" \
    --input            "${SFT_JSONL}" \
    --output           "${PARQUET_THINK}" \
    --output_no_think  "${PARQUET_NOTHINK}" \
    --image_base_dir   "${SFT_IMG_BASE}" \
    --to_parquet

# ── Step 2: Stage dataset_info.json ──────────────────────────────────────────
cp "${SFT_DIR}/lf_dataset_info.json" "${LF_DATA_DIR}/dataset_info.json"
echo "[stage] dataset_info.json → ${LF_DATA_DIR}/dataset_info.json"

# ── Step 3: Launch SFT ───────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=0
export TRANSFORMERS_VERBOSITY=info
export HF_HUB_OFFLINE=1                # weights already cached locally

# FSDP via accelerate
ACCEL_CFG="${SFT_DIR}/lf_accelerate_fsdp.yaml"
if [[ ! -f "${ACCEL_CFG}" ]]; then
cat > "${ACCEL_CFG}" <<'YAML'
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
num_processes: 4
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
fi

echo "[train] Launching LLaMA-Factory SFT (4 GPUs, FSDP)…"
echo "        config:  ${LF_CONFIG}"
echo "        dataset: ${LF_DATA_DIR}"
echo "        output:  ${OUTPUT_DIR}"

# Inject dataset_dir + output_dir overrides via CLI extra args.
accelerate launch --config_file "${ACCEL_CFG}" \
    "$(python -c 'import llamafactory, os; print(os.path.join(os.path.dirname(llamafactory.__file__), "launcher.py"))')" \
    "${LF_CONFIG}" \
    dataset_dir="${LF_DATA_DIR}" \
    output_dir="${OUTPUT_DIR}"

echo "[done] SFT finished. HF checkpoint at: ${OUTPUT_DIR}"
echo "       Use it as verl RL warmstart with:  model.partial_pretrain=${OUTPUT_DIR}"
