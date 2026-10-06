#!/usr/bin/env bash
# Thin wrapper around data_gen/robocasa_sft/run_sft.sh
#
# Examples:
#   # base Qwen2.5-VL
#   bash examples/train/robocasa/run_sft.sh \
#       --model-path "$QWEN_PATH" \
#       --data-root /path/to/lerobot_robocasa
#
#   # Active Spatial HF actor
#   bash examples/train/robocasa/run_sft.sh \
#       --model-path "$ACTIVE_SPATIAL_HF_CKPT" \
#       --data-root /path/to/lerobot_robocasa \
#       --task-set atomic_seen
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
exec bash "${REPO_ROOT}/data_gen/robocasa_sft/run_sft.sh" "$@"
