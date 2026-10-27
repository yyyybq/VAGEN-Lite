#!/usr/bin/env bash
# X5 (the user's decision 7 of 2026-10-26, lists confirmed 10-27), one 8-card job per start: the E1 final checkpoint
# (DS-<start>-100-s1/run/final_model, read by path) on the 32 unseen tasks of downstream/x/x5_plan.json (16 atomic +
# 16 composite, 20 episodes each), 8 policy servers and 96 simulator processes as in the E1 evaluation, into
# X5/<start>/out; then the analysis of every start done so far (x5_analyze.py) into X5/results.  Resumable:
# closed_loop.py runs only the episodes still missing.
# Usage: run_x5.sh START [X5_DIR]
set -uo pipefail
S="${1:?Usage: run_x5.sh START [X5_DIR]}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${X5_RUNS:-/mnt/umm/users/yinbaiqiao/downstream_runs/experiments}"  # X5_* only change for the tests
X5="${2:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X5}"
EVAL="${X5_EVAL:-$WT/downstream/eval/run_eval_ckpt.sh}"
PY="${X5_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"  # has scipy (Wilcoxon)
[[ -x "$PY" ]] || PY=python3
CKPT="$RUNS/DS-$S-100-s1/run/final_model/pytorch_model.pt"
[[ -f "$CKPT" ]] || { echo "[x5] $S: no checkpoint $CKPT"; exit 6; }
echo "[x5] $S: $CKPT $(date -u +%FT%TZ)"
EVAL_PLAN="$WT/downstream/x/x5_plan.json" bash "$EVAL" "$CKPT" "$X5/$S/out" 8 96
rc=$?
echo "[x5] $S: evaluation rc=$rc $(date -u +%FT%TZ)"
"$PY" "$WT/downstream/x/x5_analyze.py" --x5 "$X5" || echo "[x5] analysis failed (the episodes are kept; run it again by hand)"
exit $rc
