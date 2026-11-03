#!/usr/bin/env bash
# N3 tier (c) (the user's decision 2 of 2026-11-03: the 600 instances re-checked for solvability under the new
# collision rule first), one one-card sco job (the simulator on CPU): downstream/n/n3_gen.py recheck-scene for every
# scene of INSTANCES under COLLISION (sphere: the upstream camera sphere; robot: N1's own rule, in fresh processes),
# WORKERS at a time, each scene in its own process with at most SCENE_TIMEOUT seconds (a scene past it is recorded in
# OUT/scenes/<scene>.timeout.json and skipped, also by later runs); then recheck-collect (OUT/instances_checked.jsonl,
# OUT/solvable.jsonl, OUT/summary.json).  Every step skips what is done.  The simulator runs without the Mesa shader
# cache (run_n2.sh says why).  Writes only under OUT.
# Usage: run_n3_recheck.sh OUT INSTANCES COLLISION [WORKERS]   (SCENE_TIMEOUT default 1500 s)
set -uo pipefail
OUT="${1:?Usage: run_n3_recheck.sh OUT INSTANCES COLLISION [WORKERS]}"; INST="${2:?}"; COLL="${3:?}"; WORKERS="${4:-12}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENE_TIMEOUT="${SCENE_TIMEOUT:-1500}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1 MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
mkdir -p "$OUT/scenes" "$OUT/logs"
log() { echo "[n3-recheck] $* $(date -u +%FT%TZ)" | tee -a "$OUT/recheck.log"; }
mapfile -t SCENES < <(python3 -c 'import json,sys
seen = []
for line in open(sys.argv[1]):
    if line.strip():
        r = json.loads(line)
        k = "%s:%s" % (r["task"], r["seed"])
        if k not in seen:
            seen.append(k)
print("\n".join(seen))' "$INST")
log "${#SCENES[@]} scenes of $INST under $COLL, $WORKERS at a time, at most $SCENE_TIMEOUT s each"

worker() {  # worker I takes scenes I, I + WORKERS, ...
  local i=$1 j spec task seed sid t0 rc
  for ((j = i; j < ${#SCENES[@]}; j += WORKERS)); do
    spec="${SCENES[$j]}"; task="${spec%:*}"; seed="${spec##*:}"; sid="${task}_s${seed}"
    [[ -f $OUT/scenes/$sid.timeout.json ]] && continue
    grep -q '"complete": true\|"error": ' "$OUT/scenes/$sid.json" 2>/dev/null && continue
    t0=$(date +%s)
    sim 1 timeout -s KILL "$SCENE_TIMEOUT" "$SIM_PY" downstream/n/n3_gen.py recheck-scene --task "$task" --seed "$seed" \
      --instances "$INST" --collision "$COLL" --out "$OUT" >> "$OUT/logs/$sid.log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
      printf '{"task": "%s", "seed": %s, "rc": %s, "seconds": %s}\n' "$task" "$seed" "$rc" "$(( $(date +%s) - t0 ))" \
        > "$OUT/scenes/$sid.timeout.json"
      echo "[n3-recheck] $sid skipped: rc=$rc $(date -u +%FT%TZ)" >> "$OUT/recheck.log"
    fi
  done
}

for ((w = 0; w < WORKERS; w++)); do
  worker "$w" &
done
wait
sim 1 "$SIM_PY" downstream/n/n3_gen.py recheck-collect --instances "$INST" --out "$OUT" > "$OUT/collect.log" 2>&1
log "collect rc=$? $(tail -c 400 "$OUT/collect.log")"
