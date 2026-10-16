#!/usr/bin/env bash
# Keep the SCO startup command in the proven `bash entry.sh` form.
set -euo pipefail

export ANCHOR=cambrian
export ACTIVE_SPATIAL_CLEAN_RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
export ACTIVE_SPATIAL_CLEAN_RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
export ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"
export INTERIORGS_HTTP_RETRIES="${INTERIORGS_HTTP_RETRIES:-120}"
export INTERIORGS_HTTP_BACKOFF="${INTERIORGS_HTTP_BACKOFF:-0.25}"
export INTERIORGS_HTTP_MAX_BACKOFF="${INTERIORGS_HTTP_MAX_BACKOFF:-1.0}"
export CAMBRIAN_CLEAN_EXPERIMENT_NAME="${CAMBRIAN_CLEAN_EXPERIMENT_NAME:-cambrian_c8_clean_d0pass_20260821_full_r4}"

exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_clean_anchor_entry.sh
