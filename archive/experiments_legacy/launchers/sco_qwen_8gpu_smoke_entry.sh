#!/usr/bin/env bash
set -euo pipefail

export ANCHOR=qwen
export D0_17_SMOKE_ONLY=1
export QWEN_CLEAN_EXPERIMENT_NAME="qwen_v46_clean_d0pass_20260819_8gpu_smoke"
exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_clean_anchor_entry.sh
