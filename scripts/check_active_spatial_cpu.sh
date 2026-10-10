#!/usr/bin/env bash
# Run the explicit Active Spatial CPU regression set; never source a launcher.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: PYTHON=/path/to/vagen-lite/bin/python bash scripts/check_active_spatial_cpu.sh [pytest options]

Runs dataset/split, environment/reward, PPO replay, R1 and QA contract tests.
Uses synthetic fixtures and existing local artifacts; no model loading,
renderer requests, downloads, Ray launch or optimizer step is requested.
Extra arguments are passed to pytest (for example --junitxml=/tmp/results.xml).
Any known failing contract test remains a failure; it is not skipped here.
EOF
    exit 0
fi

if [[ -z "${PYTHON:-}" ]]; then
    PROJECT_PYTHON="${REPO_ROOT%/*}/.conda/envs/vagen-lite/bin/python"
    if [[ -x "$PROJECT_PYTHON" ]]; then
        PYTHON="$PROJECT_PYTHON"
    else
        PYTHON=python3
    fi
fi

cd -- "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PYTHON" -c 'import pytest, numpy, yaml, torch' >/dev/null 2>&1; then
    echo 'Set PYTHON to the existing vagen-lite interpreter with pytest/numpy/PyYAML/torch installed.' >&2
    exit 2
fi

exec "$PYTHON" -m pytest -q -p no:cacheprovider \
    tests/active_spatial/test_pipeline_contracts.py \
    tests/active_spatial/test_r1_sft_pipeline.py \
    tests/active_spatial/test_reward_trace.py \
    tests/active_spatial/test_ppo_snapshot_replay.py \
    tests/test_r1_clean_projective_v0.py \
    tests/test_r1_projective_projection_frontier.py \
    data_gen/active_spatial_qa/test_qa_contract.py \
    data_gen/active_spatial_qa/test_all_no_diagnosis.py \
    "$@"
