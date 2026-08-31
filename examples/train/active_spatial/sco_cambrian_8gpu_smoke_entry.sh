#!/usr/bin/env bash
set -euo pipefail

export ANCHOR=cambrian
export D0_17_SMOKE_ONLY=1
export CAMBRIAN_CLEAN_EXPERIMENT_NAME="cambrian_c8_clean_d0pass_20260819_8gpu_smoke"
exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_clean_anchor_entry.sh
