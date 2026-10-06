#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
# shellcheck source=../../train/robocasa/setup_starvla_env.sh
source "${ROOT}/examples/train/robocasa/setup_starvla_env.sh"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
exec python "${ROOT}/examples/evaluate/robocasa/eval_starvla_nk_infer.py" "$@"
