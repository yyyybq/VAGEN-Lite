#!/bin/bash
# =============================================================
# Quick test: run 5 episodes with random agent to verify 
# the evaluation pipeline works end-to-end
# =============================================================
set -e

cd /scratch/by2593/project/Active_Spatial/VAGEN

echo "============================================"
echo "  Quick Smoke Test (5 episodes)"
echo "============================================"

python evaluation/run_eval.py \
    --agent random \
    --max-episodes 5 \
    --max-turns 5 \
    --output-dir evaluation/outputs/smoke_test \
    --eval-name smoke_test \
    --verbose \
    --save-trajectories

echo ""
echo "Smoke test complete! Check evaluation/outputs/smoke_test/"
