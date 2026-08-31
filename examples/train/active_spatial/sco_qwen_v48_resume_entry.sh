#!/usr/bin/env bash
set -euo pipefail

export ANCHOR=qwen
export QWEN_CLEAN_EXPERIMENT_NAME="${QWEN_CLEAN_EXPERIMENT_NAME:-qwen_v48_clean_d0pass_20260823_full}"
export QWEN_V48_CLEAN_EXPERIMENT_NAME="${QWEN_CLEAN_EXPERIMENT_NAME}"
export QWEN_CLEAN_FULL_CONFIG="examples/train/active_spatial/experiments/qwen_v48_clean_v1_resume.sh"
export INTERIORGS_HTTP_RETRIES=120
export INTERIORGS_HTTP_BACKOFF=0.25
export INTERIORGS_HTTP_MAX_BACKOFF=1.0

exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_clean_anchor_entry.sh
