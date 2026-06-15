#!/bin/bash
# =============================================================
# Full evaluation pipeline: baselines + frozen + trained + compare
# Usage: bash evaluation/scripts/run_full_eval.sh [checkpoint_path]
# =============================================================
set -e

CHECKPOINT_PATH=${1:-""}

cd /scratch/by2593/project/Active_Spatial/VAGEN

# Ensure ninja is on PATH (needed for gsplat CUDA JIT compilation)
export PATH="$(python -c 'import sys; print(sys.prefix)')/bin:$PATH"

echo "============================================================"
echo "  Active Spatial — Full Evaluation Pipeline"
echo "============================================================"

# ---- Step 1: Baselines (no model needed) ----
echo ""
echo ">>> Step 1: Random Baseline"
python evaluation/run_eval.py --config evaluation/configs/eval_random.yaml

echo ""
echo ">>> Step 2: Heuristic Oracle Baseline"
python evaluation/run_eval.py --config evaluation/configs/eval_heuristic.yaml

# ---- Step 2: Frozen VLM ----
echo ""
echo ">>> Step 3: Frozen VLM (Zero-Shot)"
python evaluation/run_eval.py --config evaluation/configs/eval_frozen_vlm.yaml

# ---- Step 3: Trained Model (if checkpoint provided) ----
COMPARE_FILES=(
    "evaluation/outputs/eval_random/results_random.json"
    "evaluation/outputs/eval_heuristic/results_heuristic.json"
    "evaluation/outputs/eval_frozen/results_model.json"
)

if [ -n "$CHECKPOINT_PATH" ]; then
    echo ""
    echo ">>> Step 4: Trained Model"
    STEP_NAME=$(basename "$CHECKPOINT_PATH")
    EVAL_NAME="eval_trained_${STEP_NAME}"
    OUTPUT_DIR="evaluation/outputs/${EVAL_NAME}"
    
    python evaluation/run_eval.py \
        --agent model \
        --checkpoint "$CHECKPOINT_PATH" \
        --model-name Qwen/Qwen2.5-VL-3B-Instruct \
        --output-dir "$OUTPUT_DIR" \
        --eval-name "$EVAL_NAME" \
        --max-turns 20 \
        --temperature 0.1 \
        --save-trajectories
    
    COMPARE_FILES+=("${OUTPUT_DIR}/results_model.json")
else
    echo ""
    echo ">>> Step 4: Skipped (no checkpoint provided)"
    echo "    To include trained model: bash run_full_eval.sh /path/to/checkpoint"
fi

# ---- Step 4: Compare All ----
echo ""
echo "============================================================"
echo "  Comparison"
echo "============================================================"

# Only compare files that exist
EXISTING_FILES=()
for f in "${COMPARE_FILES[@]}"; do
    if [ -f "$f" ]; then
        EXISTING_FILES+=("$f")
    fi
done

if [ ${#EXISTING_FILES[@]} -ge 2 ]; then
    python evaluation/compare.py "${EXISTING_FILES[@]}" \
        --output evaluation/outputs/comparison.json
else
    echo "Not enough result files for comparison."
fi

echo ""
echo "============================================================"
echo "  All Done! Results in evaluation/outputs/"
echo "============================================================"
