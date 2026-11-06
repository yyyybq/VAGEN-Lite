#!/usr/bin/env bash
# D3 (a) axis (the user's task book of 2026-11-06): the variants of N1's left / right, absolute-distance and
# screen-occupancy instances, each checked solvable from the same start (downstream/n/d3_gen.py), in one one-card sco
# job (the simulator on CPU).  Each scene in its own process with at most SCENE_TIMEOUT seconds, WORKERS at a time; a
# scene killed at its limit (or failing) is recorded in OUT/scenes/<scene>.timeout.json and not retried.  Then
# collect (OUT/variants.jsonl, OUT/summary.json).  Writes only under OUT; every step skips what is done.
# The reachable-pose search runs first in its own process (OUT/states/<scene>.pkl); the scene process that renders
# only loads it (11-06: renders after the search in the same process were often colour noise).  ALSO="DIR ..." adds
# earlier output folders to collect (this run's complete scenes replace theirs).
# Usage: run_d3_gen.sh OUT [WORKERS] INSTANCES.jsonl [INSTANCES.jsonl ...]
# Environment: SCENE_TIMEOUT (default 900 s), SOFT_LIMIT (default 600 s: no new candidate after it), SCENE_SLICE
#   "I/K": only every K-th scene from the I-th (1-based), so K jobs can share OUT.
set -uo pipefail
OUT="${1:?Usage: run_d3_gen.sh OUT [WORKERS] INSTANCES.jsonl ...}"; WORKERS="${2:-10}"; shift 2
INSTANCES=("$@")
[[ ${#INSTANCES[@]} -gt 0 ]] || { echo "no instance files" >&2; exit 2; }
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE_TIMEOUT="${SCENE_TIMEOUT:-900}"; SOFT_LIMIT="${SOFT_LIMIT:-600}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/scenes" "$OUT/logs"
log() { echo "[d3-gen] $* $(date -u +%FT%TZ)" | tee -a "$OUT/d3_gen.log"; }
mapfile -t ALL < <(sim 1 "$SIM_PY" downstream/n/d3_gen.py scenes --instances "${INSTANCES[@]}")
SCENES=("${ALL[@]}")
if [[ -n ${SCENE_SLICE:-} ]]; then
  first="${SCENE_SLICE%/*}"; step="${SCENE_SLICE#*/}"; SCENES=()
  for ((j = first - 1; j < ${#ALL[@]}; j += step)); do SCENES+=("${ALL[$j]}"); done
fi
log "${#SCENES[@]} scenes${SCENE_SLICE:+ (slice $SCENE_SLICE of ${#ALL[@]})}, $WORKERS at a time, at most $SCENE_TIMEOUT s each"

scene_done() {  # finished (complete or failed) or skipped for time
  [[ -f $OUT/scenes/$1.timeout.json ]] || grep -q '"complete": true\|"error": ' "$OUT/scenes/$1.json" 2>/dev/null
}

worker() {  # worker I takes scenes I, I + WORKERS, ...
  local i=$1 j spec task seed sid t0 rc
  for ((j = i; j < ${#SCENES[@]}; j += WORKERS)); do
    spec="${SCENES[$j]}"; task="${spec%:*}"; seed="${spec##*:}"; sid="${task}_s${seed}"
    scene_done "$sid" && continue
    t0=$(date +%s); tries=0
    if [[ ! -f $OUT/states/$sid.pkl ]]; then  # the pose search in its own process (d3_gen.reach_cmd)
      sim 1 timeout -s KILL "$SCENE_TIMEOUT" "$SIM_PY" downstream/n/d3_gen.py reach --task "$task" --seed "$seed" \
        --out "$OUT" >> "$OUT/logs/$sid.log" 2>&1
    fi
    while :; do  # rc 3: this process's renderer drew garbage (d3_gen.RenderBroken); a new process, RENDER_TRIES in all
      tries=$((tries + 1))
      sim 1 timeout -s KILL "$SCENE_TIMEOUT" "$SIM_PY" downstream/n/d3_gen.py scene --instances "${INSTANCES[@]}" \
        --task "$task" --seed "$seed" --out "$OUT" --soft-limit "$SOFT_LIMIT" >> "$OUT/logs/$sid.log" 2>&1
      rc=$?
      [[ $rc -eq 3 && $tries -lt ${RENDER_TRIES:-4} ]] || break
      echo "[d3-gen] $sid render broken (try $tries), again $(date -u +%FT%TZ)" >> "$OUT/d3_gen.log"
    done
    if [[ $rc -ne 0 ]]; then
      printf '{"task": "%s", "seed": %s, "rc": %s, "seconds": %s}\n' "$task" "$seed" "$rc" "$(( $(date +%s) - t0 ))" \
        > "$OUT/scenes/$sid.timeout.json"
      echo "[d3-gen] $sid skipped: rc=$rc after $(( $(date +%s) - t0 )) s $(date -u +%FT%TZ)" >> "$OUT/d3_gen.log"
    fi
  done
}

for ((w = 0; w < WORKERS; w++)); do
  worker "$w" &
done
wait
log "scenes done: $(ls "$OUT"/scenes/*.json 2>/dev/null | grep -vc timeout) scene files, $(ls "$OUT"/scenes/*.timeout.json 2>/dev/null | wc -l) skipped"
ALSO_DIRS=(${ALSO:-})
sim 1 "$SIM_PY" downstream/n/d3_gen.py collect --instances "${INSTANCES[@]}" --out "$OUT" \
  ${ALSO_DIRS[@]+--also "${ALSO_DIRS[@]}"} | tee -a "$OUT/d3_gen.log"
log "done"
