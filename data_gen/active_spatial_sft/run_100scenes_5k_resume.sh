#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Resume SFT generation on 100-scenes 5k after OOM kill.
#
# Status at restart:
#   - Previous run processed source items [0, 2185) and was killed by OOM.
#   - Existing artefacts (PRESERVED, do NOT touch):
#       output_100scenes_5k/sft_data.jsonl         (1362 records, ids ≤ sft_002079)
#       output_100scenes_5k/sft_data_stats.json
#       output_100scenes_5k/generation.log
#       output_100scenes_5k/images/                (existing renders)
#
# This script writes a SEPARATE output:
#       output_100scenes_5k/sft_data_part2.jsonl   (new records, ids ≥ sft_002185)
#       output_100scenes_5k/sft_data_part2_stats.json
#       output_100scenes_5k/generation_part2.log
#       output_100scenes_5k/images/                (appended; ids non-collide)
#
# After completion, merge with:
#   cat output_100scenes_5k/sft_data.jsonl output_100scenes_5k/sft_data_part2.jsonl \
#       > output_100scenes_5k/sft_data_merged.jsonl
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GS_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_DIR="data_gen/active_spatial_sft/output_100scenes_5k"
SAMPLED_JSONL="$OUTPUT_DIR/sampled_5k.jsonl"
GPU_DEVICE=4
START_IDX=2185   # resume from source item 2185 (last processed: 2184)

cd "$VAGEN_ROOT"

if [[ ! -f "$SAMPLED_JSONL" ]]; then
  echo "ERROR: $SAMPLED_JSONL not found. Run run_100scenes_5k.sh first." >&2
  exit 1
fi
echo "subset: $(wc -l < "$SAMPLED_JSONL") items, resuming from idx=$START_IDX"

$PYTHON data_gen/active_spatial_sft/run_sft_generation.py \
    --jsonl_path        "$SAMPLED_JSONL" \
    --gs_root           "$GS_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --output_name       sft_data_part2 \
    --start_idx         $START_IDX \
    --render_backend    local \
    --gpu_device        $GPU_DEVICE \
    --image_width       256 \
    --image_height      256 \
    --image_format      jpg \
    --image_quality     85 \
    --step_translation  0.3 \
    --step_rotation_deg 20.0 \
    --max_actions_per_turn 5 \
    --max_total_actions    60 \
    --beam_width           3 \
    --success_threshold    0.95 \
    --min_improvement      0.005 \
    --plateau_tolerance    5 \
    --position_weight      0.7 \
    --orientation_weight   0.3 \
    --allow_failed \
    --partial_success_min_score 0.5 \
    --include_goal_reached_examples \
    --prompt_format        free_think \
    --verbose \
    2>&1 | tee "$OUTPUT_DIR/generation_part2.log"

echo ""
echo "=== SFT resume generation complete ==="
echo "Output dir   : $OUTPUT_DIR"
echo "Part2 records: $(wc -l < "$OUTPUT_DIR/sft_data_part2.jsonl" 2>/dev/null || echo '?')"
echo "Part1 records: $(wc -l < "$OUTPUT_DIR/sft_data.jsonl" 2>/dev/null || echo '?')"
echo ""
echo "Merge with:"
echo "  cat $OUTPUT_DIR/sft_data.jsonl $OUTPUT_DIR/sft_data_part2.jsonl > $OUTPUT_DIR/sft_data_merged.jsonl"
