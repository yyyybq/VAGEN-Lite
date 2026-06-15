#!/bin/bash
# =============================================================
# Evaluate a frozen (untrained) VLM as zero-shot baseline
# Requires vLLM + GPU
# =============================================================
set -e

cd /scratch/by2593/project/Active_Spatial/VAGEN

# Ensure ninja is on PATH (needed for gsplat CUDA JIT compilation)
export PATH="$(python -c 'import sys; print(sys.prefix)')/bin:$PATH"

echo "============================================"
echo "  Frozen VLM Zero-Shot Evaluation"
echo "============================================"

python evaluation/run_eval.py \
    --config evaluation/configs/eval_frozen_vlm.yaml

echo "Done! Results in evaluation/outputs/eval_frozen/"
