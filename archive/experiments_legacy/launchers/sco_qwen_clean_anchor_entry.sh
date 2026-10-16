#!/usr/bin/env bash
# Keep the SCO startup command in the proven `bash entry.sh` form.
set -euo pipefail

export ANCHOR=qwen
export ACTIVE_SPATIAL_CLEAN_RENDER_HOST="${ACTIVE_SPATIAL_CLEAN_RENDER_HOST:-10.119.30.223}"
export ACTIVE_SPATIAL_CLEAN_RENDER_PORT="${ACTIVE_SPATIAL_CLEAN_RENDER_PORT:-8768}"
export ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL="${ACTIVE_SPATIAL_CLEAN_RENDER_PROTOCOL:-http}"
# Keep requests present in the renderer's FIFO admission queue while Cambrian
# shares the single-worker service. Each HTTP attempt already waits five seconds.
export INTERIORGS_HTTP_RETRIES="${INTERIORGS_HTTP_RETRIES:-120}"
export INTERIORGS_HTTP_BACKOFF="${INTERIORGS_HTTP_BACKOFF:-0.25}"
export INTERIORGS_HTTP_MAX_BACKOFF="${INTERIORGS_HTTP_MAX_BACKOFF:-1.0}"
export QWEN_CLEAN_EXPERIMENT_NAME="${QWEN_CLEAN_EXPERIMENT_NAME:-qwen_v46_clean_d0pass_20260821_full_r6}"

exec bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite/examples/train/active_spatial/sco_clean_anchor_entry.sh
