#!/bin/bash
# Complete test script for path generation and rendering
#
# This script:
# 1. Generates navigation paths for a specific scene
# 2. Renders images along each path
#
# Usage:
#   ./run_full_test.sh <scene_id> [samples_per_task] [render_every_n]
#
# Example:
#   ./run_full_test.sh 0267_840790 2 5

set -e

# Parse arguments
SCENE_ID=${1:-0267_840790}
SAMPLES_PER_TASK=${2:-2}
RENDER_EVERY_N=${3:-5}

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/scratch/by2593/project/Active_Spatial"
VAGEN_ROOT="${PROJECT_ROOT}/VAGEN"
DATA_DIR="${VAGEN_ROOT}/data/active_spatial"
GS_ROOT="${PROJECT_ROOT}/InteriorGS"

# Input data file (scene-specific)
TRAIN_DATA="${DATA_DIR}/train_data_${SCENE_ID}.jsonl"

# Output paths
PATHS_OUTPUT="${DATA_DIR}/paths_${SCENE_ID}.jsonl"
RENDER_OUTPUT="${DATA_DIR}/rendered/${SCENE_ID}"

echo "=============================================="
echo "Active Spatial Path Generation & Rendering"
echo "=============================================="
echo "Scene ID:          ${SCENE_ID}"
echo "Samples per task:  ${SAMPLES_PER_TASK}"
echo "Render every N:    ${RENDER_EVERY_N}"
echo "Input data:        ${TRAIN_DATA}"
echo "Output paths:      ${PATHS_OUTPUT}"
echo "Output images:     ${RENDER_OUTPUT}"
echo "=============================================="
echo ""

# Check if input file exists
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "[ERROR] Training data file not found: ${TRAIN_DATA}"
    echo "[INFO] Please run run_pipeline.py first to generate training data."
    exit 1
fi

# Activate environment
source /scratch/by2593/miniconda3/bin/activate vagen

# Step 1: Generate diverse paths (all 9 task types)
echo "[Step 1/2] Generating paths for all task types..."
python "${SCRIPT_DIR}/generate_diverse_paths.py" \
    --jsonl_path "${TRAIN_DATA}" \
    --scene_id "${SCENE_ID}" \
    --output_path "${PATHS_OUTPUT}" \
    --samples_per_task ${SAMPLES_PER_TASK} \
    --max_steps 200 \
    --min_steps 5 \
    --min_distance 0.5 \
    --min_yaw_offset 25

echo ""

# Step 2: Render images along paths
echo "[Step 2/2] Rendering images..."
python "${SCRIPT_DIR}/render_path_images.py" \
    --paths_jsonl "${PATHS_OUTPUT}" \
    --output_dir "${RENDER_OUTPUT}" \
    --gs_root "${GS_ROOT}" \
    --render_backend "local" \
    --image_width 640 \
    --image_height 480 \
    --render_every_n ${RENDER_EVERY_N}

echo ""
echo "=============================================="
echo "Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  Paths file:    ${PATHS_OUTPUT}"
echo "  Rendered dir:  ${RENDER_OUTPUT}"
echo ""
echo "Quick summary:"
cat "${RENDER_OUTPUT}/render_summary.json" | python -m json.tool
echo ""
echo "View individual paths:"
echo "  ls ${RENDER_OUTPUT}/${SCENE_ID}/"
