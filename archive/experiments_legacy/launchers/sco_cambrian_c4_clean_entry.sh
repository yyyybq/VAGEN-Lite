#!/usr/bin/env bash
set -euo pipefail
export ANCHOR=cambrian
export CAMBRIAN_CLEAN_EXPERIMENT_NAME="${CAMBRIAN_CLEAN_EXPERIMENT_NAME:-cambrian_c4_clean_d0pass_20260821_full}"
export CAMBRIAN_C4_CLEAN_EXPERIMENT_NAME="${CAMBRIAN_CLEAN_EXPERIMENT_NAME}"
export CAMBRIAN_CLEAN_FULL_CONFIG="examples/train/active_spatial/experiments/cambrian_c4_clean_v1.sh"
export INTERIORGS_HTTP_TIMEOUT=900
export INTERIORGS_HTTP_RETRIES=120
export INTERIORGS_HTTP_BACKOFF=0.25
export INTERIORGS_HTTP_MAX_BACKOFF=1.0
exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_clean_anchor_entry.sh
