#!/usr/bin/env bash
# N1 instances (the user's decision 3 of 2026-10-31), one one-card sco job (the simulator on CPU):
# downstream/n/n_gen.py scene for the first N_SCENES of X0's scenes (n_gen.py scenes: the tasks in turn), WORKERS at a
# time, each scene in its own process with at most SCENE_TIMEOUT seconds.  A scene killed at its limit (or failing) is
# recorded in OUT/scenes/<scene>.timeout.json and skipped, also by later runs; the task types it finished stay in its
# scene file.  Then collect (OUT/instances_eval.jsonl, summary.json) and the review page (OUT/page.html).  Every step
# skips what is done.  Writes only under OUT.
# Usage: run_n_gen.sh OUT [WORKERS] [N_SCENES] [PER_FAMILY]
# Environment: SCENE_TIMEOUT (default 1500 s), SOFT_LIMIT (default 1000 s: no new candidate after it), N_X0,
#   SCENE_SLICE "I/K": only every K-th of the N_SCENES from the I-th (1-based), so K jobs can share OUT.
set -uo pipefail
OUT="${1:?Usage: run_n_gen.sh OUT [WORKERS] [N_SCENES] [PER_FAMILY]}"; WORKERS="${2:-12}"; N_SCENES="${3:-120}"
PER_FAMILY="${4:-50}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
X0="${N_X0:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out/questions.jsonl}"
SCENE_TIMEOUT="${SCENE_TIMEOUT:-1500}"; SOFT_LIMIT="${SOFT_LIMIT:-1000}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/scenes" "$OUT/logs"
log() { echo "[n-gen] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n_gen.log"; }
mapfile -t SCENES < <(sim 1 "$SIM_PY" downstream/n/n_gen.py scenes --x0 "$X0" --n "$N_SCENES")
if [[ -n ${SCENE_SLICE:-} ]]; then
  first="${SCENE_SLICE%/*}"; step="${SCENE_SLICE#*/}"; ALL=("${SCENES[@]}"); SCENES=()
  for ((j = first - 1; j < ${#ALL[@]}; j += step)); do SCENES+=("${ALL[$j]}"); done
fi
log "${#SCENES[@]} scenes${SCENE_SLICE:+ (slice $SCENE_SLICE)}, $WORKERS at a time, at most $SCENE_TIMEOUT s each (no new candidate after $SOFT_LIMIT s)"

worker() {  # worker I takes scenes I, I + WORKERS, ...
  local i=$1 j spec task seed sid t0 rc
  for ((j = i; j < ${#SCENES[@]}; j += WORKERS)); do
    spec="${SCENES[$j]}"; task="${spec%:*}"; seed="${spec##*:}"; sid="${task}_s${seed}"
    [[ -f $OUT/scenes/$sid.timeout.json ]] && continue
    grep -q '"complete": true\|"error": ' "$OUT/scenes/$sid.json" 2>/dev/null && continue
    t0=$(date +%s)
    sim 1 timeout -s KILL "$SCENE_TIMEOUT" "$SIM_PY" downstream/n/n_gen.py scene --task "$task" --seed "$seed" \
      --out "$OUT" --soft-limit "$SOFT_LIMIT" >> "$OUT/logs/$sid.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      printf '{"task": "%s", "seed": %s, "rc": %s, "seconds": %s}\n' "$task" "$seed" "$rc" "$(( $(date +%s) - t0 ))" \
        > "$OUT/scenes/$sid.timeout.json"
      echo "[n-gen] $sid skipped: rc=$rc after $(( $(date +%s) - t0 )) s $(date -u +%FT%TZ)" >> "$OUT/n_gen.log"
    fi
  done
}

for ((w = 0; w < WORKERS; w++)); do
  worker "$w" &
done
wait
log "scenes done: $(ls "$OUT"/scenes/*.json 2>/dev/null | grep -vc timeout) scene files, $(ls "$OUT"/scenes/*.timeout.json 2>/dev/null | wc -l) skipped"
sim 1 "$SIM_PY" downstream/n/n_gen.py collect --out "$OUT" --per-family "$PER_FAMILY" > "$OUT/collect.log" 2>&1
log "collect rc=$?"
sim 1 "$SIM_PY" downstream/n/n_gen.py page --out "$OUT" --page "$OUT/page.html" >> "$OUT/n_gen.log" 2>&1
log "page rc=$?"
