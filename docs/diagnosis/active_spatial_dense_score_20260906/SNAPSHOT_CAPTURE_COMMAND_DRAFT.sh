#!/usr/bin/env bash
# NOT_RUN. Use only after the formal data/canonical/scorer gate is regenerated PASS.
# This deliberately stops before critic/actor optimizer steps after exactly one
# post-GAE/pre-update batch has been written. It will still perform the one
# future rollout needed to collect that batch; do not execute in this round.
set -euo pipefail

export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR="/ABS/PATH/to/new/pass-gated/snapshot"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_STOP_AFTER_WRITE=1
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_ACTOR_CHECKPOINT_PATH="/ABS/PATH/to/immutable/actor"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_ACTOR_CHECKPOINT_SHA256="<64-hex-content-sha256>"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_CRITIC_CHECKPOINT_PATH="/ABS/PATH/to/immutable/critic"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_CRITIC_CHECKPOINT_SHA256="<64-hex-content-sha256>"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_REFERENCE_CHECKPOINT_PATH="/ABS/PATH/to/immutable/reference"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_REFERENCE_CHECKPOINT_SHA256="<64-hex-content-sha256>"

# First obtain PASS with active_spatial_dense_score_launch_gate.py, materialize
# the PASS-gated S1 config, then replace this command's config/entry explicitly.
# Do not share or restart port 8877; do not submit SCO from this draft.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m vagen.main_ppo \
  --config-path /ABS/PATH/to/PASS-gated/config-dir \
  --config-name S1_exact_snapshot_no_update
