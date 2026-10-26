#!/usr/bin/env bash
# X3 (the user's decision 5 of 2026-10-26), one 8-card job per start: the E1 final checkpoint (DS-<start>-100-s1/run/
# final_model, read by path; the checkpoint the E1 evaluation used) replays the 447 spatial episodes of the fixed
# evaluation with the read-only target recorder (downstream/x/x3_plan.json: closed_loop.py writes end_state) into
# X3/<start>/out, 8 policy servers and 96 simulator processes as in the E1 evaluation; then the analysis of every start
# done so far (x3_targets.py analyze: the per-episode identity check against the E1 evaluation, the target table) into
# X3/results.  Resumable: closed_loop.py runs only the episodes still missing.
# Usage: run_x3.sh START [X3_DIR]
set -uo pipefail
S="${1:?Usage: run_x3.sh START [X3_DIR]}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="${X3_RUNS:-/mnt/umm/users/yinbaiqiao/downstream_runs/experiments}"  # X3_* only change for the tests
X3="${2:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X3}"
EVAL="${X3_EVAL:-$WT/downstream/eval/run_eval_ckpt.sh}"
CKPT="$RUNS/DS-$S-100-s1/run/final_model/pytorch_model.pt"
[[ -f "$CKPT" ]] || { echo "[x3] $S: no checkpoint $CKPT"; exit 6; }
echo "[x3] $S: $CKPT $(date -u +%FT%TZ)"
EVAL_PLAN="$WT/downstream/x/x3_plan.json" bash "$EVAL" "$CKPT" "$X3/$S/out" 8 96
rc=$?
echo "[x3] $S: evaluation rc=$rc $(date -u +%FT%TZ)"
python3 "$WT/downstream/x/x3_targets.py" analyze --x3 "$X3" --experiments "$RUNS" --out "$X3/results" \
  || echo "[x3] analysis failed (the episodes are kept; run it again by hand)"
exit $rc
