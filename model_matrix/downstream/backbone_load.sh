#!/usr/bin/env bash
# The StarVLA backbone acceptance of the model line, part 1 (the user's 4.4 of 2026-10-24): one start loaded as
# QwenOFT's vision-language backbone on one card, a fake batch through the training path (L1 action loss) and
# predict_action, with the downstream line's own load test (read only, by path).  A 1-card sco job; writes only OUT.
#
#   bash backbone_load.sh OUT NAME START_PATH
set -uo pipefail
OUT=${1:?output folder}; NAME=${2:?start name}; START=${3:?start path}
WT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
DS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
mkdir -p "${OUT}"
OWNER=$(stat -c %u:%g "${OUT}")
trap 'chown -R "${OWNER}" "${OUT}" 2>/dev/null || true' EXIT
exec > >(tee -a "${OUT}/load_${NAME}.log") 2>&1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled NO_PROXY='*' no_proxy='*' CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
cd "${DS}"
source "${WT}/examples/train/robocasa/setup_starvla_env.sh"
set +e
export PYTHONPATH="${DS}:${STARVLA_ROOT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python "${DS}/downstream/load_test/load_in_starvla.py" --name "${NAME}" --path "${START}" \
  --config "${WT}/examples/train/robocasa/starvla_qwenoft_navigatekitchen_smoke.yaml" --out "${OUT}/load_${NAME}.json"
echo "[backbone_load] exit $? $(date -u +%FT%TZ)"
