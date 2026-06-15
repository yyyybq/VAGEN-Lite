#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# v8: dense feedback (max_actions_per_turn=1) + partial trajectories + goal-reached examples
#
# Changes vs v7:
#   - max_actions_per_turn=1  : model sees a new image after EVERY single action
#   - max_total_actions=200   : more search budget
#   - _plan_to_position sign bug fixed (move_backward toward target)
#   - --allow_failed + partial_success_min_score=0.4 : include near-success
#     trajectories that didn't fully reach the goal, without a final 'done'
#   - --include_goal_reached_examples : add 1-turn "already at goal → done"
#     examples to teach the model the stopping boundary
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JSONL_PATH="data_gen/active_spatial_pipeline/output_v2/train_data_0267_840790.jsonl"
GS_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_DIR="data_gen/active_spatial_sft/output_0267_v8_dense"
GPU_DEVICE=0

cd "$VAGEN_ROOT"

mkdir -p "$OUTPUT_DIR"

$PYTHON data_gen/active_spatial_sft/run_sft_generation.py \
    --jsonl_path   "$JSONL_PATH" \
    --gs_root      "$GS_ROOT" \
    --output_dir   "$OUTPUT_DIR" \
    --output_name  sft_data \
    --render_backend local \
    --gpu_device   $GPU_DEVICE \
    --image_width  512 \
    --image_height 512 \
    --max_actions_per_turn  1 \
    --max_total_actions    200 \
    --beam_width           3 \
    --success_threshold    0.95 \
    --min_improvement      0.005 \
    --plateau_tolerance    5 \
    --position_weight      0.7 \
    --orientation_weight   0.3 \
    --allow_failed \
    --partial_success_min_score 0.4 \
    --include_goal_reached_examples \
    --verbose \
    2>&1 | tee "$OUTPUT_DIR/generation.log"

echo ""
echo "=== v8 dense generation complete ==="
echo "Output: $OUTPUT_DIR"
