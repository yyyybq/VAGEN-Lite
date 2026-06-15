#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SFT generation on 100-scenes data, 5k sampled items, 256x256 (matches RL env)
#
# Output: data_gen/active_spatial_sft/output_100scenes_5k/
#   ├── sampled_5k.jsonl             (random subset, task_type-stratified)
#   ├── sft_data.jsonl
#   ├── sft_data_stats.json
#   ├── generation.log
#   └── images/sft_NNNNNN_stepKK.jpg
#
# Strategy vs v8_dense:
#   ★ image_width/height: 512 -> 256  (match RL training resolution)
#   ★ max_actions_per_turn: 1 -> 5    (match RL env max_actions_per_step=5,
#                                        avoids policy distribution mismatch)
#   ★ step_rotation_deg: 30 -> 20      (match RL env: step_rotation_deg=20)
#   ★ allow_failed + partial_success_min_score=0.5
#                                       (relax: 100scenes 包含远距离 occupancy
#                                        任务, success_threshold=0.95 难达成)
#   ★ include_goal_reached_examples
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

INPUT_JSONL="data_gen/active_spatial_pipeline/output_100scenes/train.jsonl"
GS_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_DIR="data_gen/active_spatial_sft/output_100scenes_5k"
SAMPLED_JSONL="$OUTPUT_DIR/sampled_5k.jsonl"
GPU_DEVICE=4   # 与 RL rendering GPU 一致, 训练空闲时段跑

cd "$VAGEN_ROOT"
mkdir -p "$OUTPUT_DIR"

# ── Step 1: sample 5k stratified by task_type ───────────────────────────────
if [[ ! -f "$SAMPLED_JSONL" ]]; then
  echo "=== sampling 5k subset ==="
  $PYTHON data_gen/active_spatial_sft/sample_subset.py \
      --input  "$INPUT_JSONL" \
      --output "$SAMPLED_JSONL" \
      --n 5000 --seed 42 --stratify_by task_type
fi
echo "subset: $(wc -l < "$SAMPLED_JSONL") items"

# ── Step 2: generate SFT trajectories + render ──────────────────────────────
$PYTHON data_gen/active_spatial_sft/run_sft_generation.py \
    --jsonl_path        "$SAMPLED_JSONL" \
    --gs_root           "$GS_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --output_name       sft_data \
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
    2>&1 | tee "$OUTPUT_DIR/generation.log"

echo ""
echo "=== SFT generation complete ==="
echo "Output dir : $OUTPUT_DIR"
echo "Records    : $(wc -l < "$OUTPUT_DIR/sft_data.jsonl" 2>/dev/null || echo '?')"
echo "Stats      : $OUTPUT_DIR/sft_data_stats.json"
