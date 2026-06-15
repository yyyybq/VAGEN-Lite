#!/bin/bash
# =============================================================
# Run ALL baseline evaluations (random + heuristic)
# These don't require a GPU model, just the rendering GPU.
# =============================================================
set -e

cd /scratch/by2593/project/Active_Spatial/VAGEN

echo "============================================"
echo "  Active Spatial Evaluation Pipeline"
echo "============================================"

# ---- 1. Random Baseline ----
echo ""
echo "[1/2] Running Random Baseline..."
python evaluation/run_eval.py \
    --config evaluation/configs/eval_random.yaml

# ---- 2. Heuristic Baseline ----
echo ""
echo "[2/2] Running Heuristic (Oracle) Baseline..."
python evaluation/run_eval.py \
    --config evaluation/configs/eval_heuristic.yaml

# ---- Compare ----
echo ""
echo "============================================"
echo "  Comparing Results"
echo "============================================"
python evaluation/compare.py \
    evaluation/outputs/eval_random/results_random.json \
    evaluation/outputs/eval_heuristic/results_heuristic.json

echo ""
echo "Done! Results saved in evaluation/outputs/"
