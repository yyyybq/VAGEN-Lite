#!/usr/bin/env bash
# CONFIG comes from prepare_starvla_transfer.py. Never substitute a base model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:?Usage: run_starvla_transfer.sh /absolute/prepared/source.yaml}"
CONFIG="$(readlink -f "$CONFIG")"
source "$ROOT/examples/train/robocasa/setup_starvla_env.sh"
export PYTHONPATH="${ROOT}:${ROOT}/third_party/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled WANDB_DISABLED=true
python "$ROOT/scripts/check_starvla_transfer.py" --config "$CONFIG"
[[ "${CHECK_ONLY:-0}" == 1 ]] && exit 0
cd "$ROOT/third_party/starVLA"
exec accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS:-8}" starVLA/training/train_starvla.py --config_yaml "$CONFIG"
