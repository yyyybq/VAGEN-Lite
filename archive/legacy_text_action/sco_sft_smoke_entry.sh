#!/usr/bin/env bash
# Zoetrope 1-GPU smoke: convert a few RoboCasa demos, then 4-step VLM SFT.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
PY="${PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
CONDA_BIN="${CONDA_BIN:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin}"
DATA_ROOT="${DATA_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets}"
MODEL_PATH="${MODEL_PATH:-/mnt/umm/users/yinbaiqiao/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3}"
LF_REPO="${LF_REPO:-$ROOT/third_party/LLaMA-Factory}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/robocasa_sft_smoke}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${STAMP}"
LF_DATA_DIR="${RUN_DIR}/lf_data"
IMAGE_DIR="${RUN_DIR}/images"
CKPT_DIR="${RUN_DIR}/ckpt"
LOG="${RUN_DIR}/smoke.log"

mkdir -p "${LF_DATA_DIR}" "${IMAGE_DIR}" "${CKPT_DIR}"
exec > >(tee -a "${LOG}") 2>&1

echo "[smoke] host=$(hostname) date=$(date -u +%FT%TZ)"
echo "[smoke] python=${PY}"
echo "[smoke] cuda=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
cd "${ROOT}"

export PATH="${CONDA_BIN}:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
# vagen-lite has datasets==5.0.0; LLaMA-Factory currently caps at 4.0.0.
export DISABLE_VERSION_CHECK=1

[[ -x "${PY}" ]] || { echo "[fatal] python missing: ${PY}"; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "[fatal] model missing: ${MODEL_PATH}"; exit 2; }
[[ -d "${DATA_ROOT}" ]] || { echo "[fatal] data-root missing: ${DATA_ROOT}"; exit 2; }
[[ -d "${LF_REPO}/src/llamafactory" ]] || { echo "[fatal] LLaMA-Factory missing: ${LF_REPO}"; exit 2; }

echo "[1/3] convert LeRobot -> sharegpt (max 16 samples)"
"${PY}" "${ROOT}/data_gen/robocasa_sft/convert_lerobot_to_sft.py" \
  --data-root "${DATA_ROOT}" \
  --task PickPlaceCounterToCabinet \
  --split target \
  --stride 8 \
  --horizon 4 \
  --max-samples 16 \
  --output-dir "${LF_DATA_DIR}" \
  --image-dir "${IMAGE_DIR}"

cp "${ROOT}/data_gen/robocasa_sft/lf_dataset_info.json" "${LF_DATA_DIR}/dataset_info.json"
ls -lh "${LF_DATA_DIR}"
# Fail early if convert produced nothing.
if [[ ! -f "${LF_DATA_DIR}/robocasa_sft.parquet" && ! -f "${LF_DATA_DIR}/robocasa_sft.jsonl" ]]; then
  echo "[fatal] converter wrote no dataset under ${LF_DATA_DIR}" >&2
  exit 3
fi
N_JSONL=0
if [[ -f "${LF_DATA_DIR}/robocasa_sft.jsonl" ]]; then
  N_JSONL=$(wc -l < "${LF_DATA_DIR}/robocasa_sft.jsonl" | tr -d " ")
  echo "[convert] jsonl lines=${N_JSONL}"
  (( N_JSONL > 0 )) || { echo "[fatal] empty jsonl"; exit 3; }
fi

echo "[2/3] install LLaMA-Factory into this python if needed (--no-deps)"
if ! "${PY}" -c "import llamafactory" >/dev/null 2>&1; then
  "${PY}" -m pip install -e "${LF_REPO}" --no-deps --disable-pip-version-check
fi
"${PY}" -c "import llamafactory, os; print('llamafactory', os.path.dirname(llamafactory.__file__))"
echo "[3/3] 4-step SFT smoke"
SMOKE_YAML="${ROOT}/examples/train/robocasa/lf_sft_smoke.yaml"
# v0.9.3: `cli train yaml` pops the subcommand so argv[1] is the yaml.
# `launcher.py yaml` also works; do not pass `train` into launcher.py.
if command -v llamafactory-cli >/dev/null 2>&1; then
  LF_CMD=(llamafactory-cli train "${SMOKE_YAML}")
else
  LF_CMD=("${PY}" -m llamafactory.cli train "${SMOKE_YAML}")
fi
# OmegaConf treats bare `no` as boolean False; keep eval_strategy in the yaml.
"${LF_CMD[@]}" \
  model_name_or_path="${MODEL_PATH}" \
  dataset_dir="${LF_DATA_DIR}" \
  output_dir="${CKPT_DIR}"

echo "[done] smoke finished"
echo "[done] log=${LOG}"
echo "[done] ckpt=${CKPT_DIR}"
ls -lh "${CKPT_DIR}" | head
if [[ ! -f "${CKPT_DIR}/config.json" && ! -d "${CKPT_DIR}/checkpoint-4" ]]; then
  echo "[fatal] no HF checkpoint written under ${CKPT_DIR}" >&2
  exit 4
fi
echo SMOKE_OK
