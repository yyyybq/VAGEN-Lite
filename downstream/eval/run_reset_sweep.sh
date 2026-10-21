#!/usr/bin/env bash
# Seeded-reset reproducibility sweep (supplementary requirement 2) in one sco job, no policy: every (task, seed) of the
# given tasks reset REPEATS times, each time in a fresh process (downstream/eval/reset_repeat.py), WORKERS at a time;
# once with RoboCasa as it is (OUT/sweep_plain.json) and once with the evaluation's stable counter-region order
# (OUT/sweep_stable.json).
# Usage: run_reset_sweep.sh OUT WORKERS SEEDS REPEATS [TASK ...]   (no TASK: the 34 tasks of eval_d6.json)
# Writes only under OUT.
set -uo pipefail
OUT="${1:?Usage: run_reset_sweep.sh OUT WORKERS SEEDS REPEATS [TASK ...]}"; WORKERS="${2:?}"; SEEDS="${3:?}"
REPEATS="${4:?}"; shift 4
TASKS=("$@"); (( ${#TASKS[@]} )) || TASKS=(plan)
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
mkdir -p "$OUT"
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 14)
THREADS=$(( CPUS / WORKERS )); (( THREADS < 1 )) && THREADS=1
echo "[sweep] cpus=$CPUS workers=$WORKERS threads/process=$THREADS seeds=$SEEDS repeats=$REPEATS tasks=${TASKS[*]}"
rc=0
for MODE in plain stable; do
  FLAG=(); [[ $MODE == stable ]] && FLAG=(--stable-regions)
  echo "[sweep] >>> $MODE $(date -u +%T)"
  sim "$THREADS" "$SIM_PY" "$WT/downstream/eval/reset_repeat.py" --tasks "${TASKS[@]}" --seeds "$SEEDS" \
    --repeats "$REPEATS" --workers "$WORKERS" "${FLAG[@]}" --out "$OUT/sweep_$MODE.json" > "$OUT/sweep_$MODE.log" 2>&1
  r=$?; (( r )) && rc=$r
  echo "[sweep] <<< $MODE rc=$r $(date -u +%T) $(tail -1 "$OUT/sweep_$MODE.log" | cut -c1-600)"
done
exit $rc
