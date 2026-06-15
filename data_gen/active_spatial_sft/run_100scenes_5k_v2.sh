#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SFT generation v2 — 5-bug fix版本
#
# 修复了以下问题后重新生成：
#   Bug1: path_finder._extract_target_pose 忽略 sample_forward
#   Bug2: _score_delta_control FoV使用 target_point 造成score上界~0.87
#   Bug3: _score_delta_control 朝向方向推断错误（16%的delta任务）
#   Bug4: _score_fov_inclusion 朝向公式惩罚合理位置
#   Bug5: _score_size_distance_invariance 使用3D距离但pipeline用2D
#
# Oracle pose验证结果（修复后）：
#   总计 4996/5000 (99.9%) 达到 ≥ 0.95
#
# 输入：和v1相同的 sampled_5k.jsonl（不重新采样，保持可对比性）
# 输出：output_100scenes_5k_v2/
# ---------------------------------------------------------------------------
set -euo pipefail

PYTHON=/scratch/by2593/miniconda3/envs/vagen/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAGEN_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 使用和v1相同的 sampled_5k.jsonl（固定seed=42，任务均衡采样）
SAMPLED_JSONL="data_gen/active_spatial_sft/output_100scenes_5k/sampled_5k.jsonl"
GS_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_DIR="data_gen/active_spatial_sft/output_100scenes_5k_v2"
GPU_DEVICE=4

cd "$VAGEN_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "=== SFT generation v2 (5-bug-fix) ==="
echo "Input  : $SAMPLED_JSONL ($(wc -l < "$SAMPLED_JSONL") items)"
echo "Output : $OUTPUT_DIR"
echo "GPU    : $GPU_DEVICE"
echo ""

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
echo "=== 完成 ==="
echo "输出目录 : $OUTPUT_DIR"
echo "总条数   : $(wc -l < "$OUTPUT_DIR/sft_data.jsonl" 2>/dev/null || echo '?')"
echo "统计文件 : $OUTPUT_DIR/sft_data_stats.json"
