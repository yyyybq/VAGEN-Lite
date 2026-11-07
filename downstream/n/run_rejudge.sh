#!/usr/bin/env bash
# Re-judge N1 / N3 (e) instance records with the render fix (downstream/n/rejudge.py; RENDER_CAUSE_20261107 item 1,
# the user's 2026-11-07 item 5), in one one-card sco job (the simulator on CPU).  Each (set, scene) in its own process
# with at most SCENE_TIMEOUT seconds, WORKERS at a time; a scene killed at its limit is recorded in
# OUT/scenes/<set>/<scene>.timeout.json and not retried.  Then collect (OUT/rejudge.jsonl, OUT/summary.json).
# Writes only under OUT; a scene already complete is skipped.
# Usage: run_rejudge.sh OUT WORKERS NAME=INSTANCES.jsonl [NAME=INSTANCES.jsonl ...]
# Environment: SCENE_TIMEOUT (default 600 s); N_RENDER_FIX (default on: n_env.render_fix_on).
set -uo pipefail
OUT="${1:?Usage: run_rejudge.sh OUT WORKERS NAME=FILE ...}"; WORKERS="${2:?}"; shift 2
SETS=("$@")
[[ ${#SETS[@]} -gt 0 ]] || { echo "no sets" >&2; exit 2; }
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE_TIMEOUT="${SCENE_TIMEOUT:-600}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/logs"
log() { echo "[rejudge] $* $(date -u +%FT%TZ)" | tee -a "$OUT/rejudge.log"; }
mapfile -t SCENES < <(sim 1 "$SIM_PY" downstream/n/rejudge.py scenes --set "${SETS[@]}")
log "${#SCENES[@]} scenes (set:task:seed), $WORKERS at a time, at most $SCENE_TIMEOUT s each, render fix ${N_RENDER_FIX:-1}"

file_of() {  # NAME -> its instance file
  local s
  for s in "${SETS[@]}"; do [[ ${s%%=*} == "$1" ]] && { echo "${s#*=}"; return; }; done
}

worker() {
  local i=$1 j spec name rest task seed sid rc
  for ((j = i; j < ${#SCENES[@]}; j += WORKERS)); do
    spec="${SCENES[$j]}"; name="${spec%%:*}"; rest="${spec#*:}"; task="${rest%%:*}"; seed="${rest##*:}"
    sid="${task}_s${seed}"
    mkdir -p "$OUT/scenes/$name"
    grep -q '"complete": true' "$OUT/scenes/$name/$sid.json" 2>/dev/null && continue
    [[ -f $OUT/scenes/$name/$sid.timeout.json ]] && continue
    sim 1 timeout -s KILL "$SCENE_TIMEOUT" "$SIM_PY" downstream/n/rejudge.py scene --set "$name=$(file_of "$name")" \
      --task "$task" --seed "$seed" --out "$OUT" >> "$OUT/logs/${name}_$sid.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      printf '{"set": "%s", "task": "%s", "seed": %s, "rc": %s}\n' "$name" "$task" "$seed" "$rc" \
        > "$OUT/scenes/$name/$sid.timeout.json"
      log "$name/$sid: rc=$rc"
    fi
  done
}

for ((w = 0; w < WORKERS; w++)); do worker "$w" & done
wait
sim 1 "$SIM_PY" downstream/n/rejudge.py collect --set "${SETS[@]}" --out "$OUT" | tee -a "$OUT/rejudge.log"
log "done"
