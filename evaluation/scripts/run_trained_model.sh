#!/bin/bash
# =============================================================
# Evaluate a trained model checkpoint
# Usage: bash evaluation/scripts/run_trained_model.sh /path/to/checkpoint
# =============================================================
set -e

CHECKPOINT_PATH=${1:-""}

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: bash evaluation/scripts/run_trained_model.sh /path/to/checkpoint"
    echo ""
    echo "Examples:"
    echo "  bash evaluation/scripts/run_trained_model.sh outputs/2026-03-18/12-06-59/checkpoints/step_200"
    echo "  bash evaluation/scripts/run_trained_model.sh /scratch/by2593/project/Active_Spatial/VAGEN/data/active_spatial_ppo_4gpu_warmer/checkpoint_step200"
    exit 1
fi

cd /scratch/by2593/project/Active_Spatial/VAGEN

# Ensure ninja is on PATH (needed for gsplat CUDA JIT compilation)
export PATH="$(python -c 'import sys; print(sys.prefix)')/bin:$PATH"

STEP_NAME=$(basename "$CHECKPOINT_PATH")
EVAL_NAME="eval_trained_${STEP_NAME}"
OUTPUT_DIR="evaluation/outputs/${EVAL_NAME}"

echo "============================================"
echo "  Trained Model Evaluation"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

python evaluation/run_eval.py \
    --agent model \
    --checkpoint "$CHECKPOINT_PATH" \
    --model-name Qwen/Qwen2.5-VL-3B-Instruct \
    --output-dir "$OUTPUT_DIR" \
    --eval-name "$EVAL_NAME" \
    --max-turns 20 \
    --temperature 0.1 \
    --save-trajectories

echo "Done! Results in ${OUTPUT_DIR}/"
