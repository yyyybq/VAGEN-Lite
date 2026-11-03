#!/usr/bin/env bash
# N1 instances (the user's decision 3 of 2026-10-31), one one-card sco job (the simulator on CPU):
# downstream/n/n_gen.py scene for the first N_SCENES of X0's scenes (n_gen.py scenes: the tasks in turn), WORKERS at a
# time, each scene in its own process with at most SCENE_TIMEOUT seconds.  A scene killed at its limit (or failing) is
# recorded in OUT/scenes/<scene>.timeout.json and skipped, also by later runs; the task types it finished stay in its
# scene file.  Then collect (OUT/instances_eval.jsonl, summary.json) and the review page (OUT/page.html).  Every step
# skips what is done.  Writes only under OUT.
# Once every scene of the whole list (all slices) is done or skipped, the collection is also written to
# OUT/instances_final.jsonl, which the evaluation jobs wait for (a slice's own collection may miss the other slices').
# Usage: run_n_gen.sh OUT [WORKERS] [N_SCENES] [PER_FAMILY]   (N_SCENES 0 = the whole list)
# Environment: SCENE_TIMEOUT (default 1500 s), SOFT_LIMIT (default 1000 s: no new candidate after it), N_X0,
#   SCENE_SLICE "I/K": only every K-th of the N_SCENES from the I-th (1-based), so K jobs can share OUT.
#   N_SEEDS_FROM S with N_PER_TASK K (N1's second set, the user's decision 3 of 2026-11-02): X0's tasks with the new
#   resets S .. S + K - 1 instead of X0's scenes; N_RNG_SEED: the candidate and collection seed (default n_gen's).
#   N_PATH_FRAMES=1 (N2's demonstrations, the user's decision 1 of 2026-11-03): every scene also saves every frame of
#   its instances' shortest paths (n_gen.py scene --path-frames).
set -uo pipefail
OUT="${1:?Usage: run_n_gen.sh OUT [WORKERS] [N_SCENES] [PER_FAMILY]}"; WORKERS="${2:-12}"; N_SCENES="${3:-120}"
PER_FAMILY="${4:-50}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
X0="${N_X0:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out/questions.jsonl}"
SCENE_TIMEOUT="${SCENE_TIMEOUT:-1500}"; SOFT_LIMIT="${SOFT_LIMIT:-1000}"
NEW=()
[[ -n ${N_SEEDS_FROM:-} ]] && NEW=(--seeds-from "$N_SEEDS_FROM" --per-task "${N_PER_TASK:?N_PER_TASK with N_SEEDS_FROM}")
RNG=()
[[ -n ${N_RNG_SEED:-} ]] && RNG=(--rng-seed "$N_RNG_SEED")
PF=()
[[ ${N_PATH_FRAMES:-0} == 1 ]] && PF=(--path-frames)
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/scenes" "$OUT/logs"
log() { echo "[n-gen] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n_gen.log"; }
mapfile -t ALL < <(sim 1 "$SIM_PY" downstream/n/n_gen.py scenes --x0 "$X0" --n "$N_SCENES" ${NEW[@]+"${NEW[@]}"})
SCENES=("${ALL[@]}")
if [[ -n ${SCENE_SLICE:-} ]]; then
  first="${SCENE_SLICE%/*}"; step="${SCENE_SLICE#*/}"; SCENES=()
  for ((j = first - 1; j < ${#ALL[@]}; j += step)); do SCENES+=("${ALL[$j]}"); done
fi
log "${#SCENES[@]} scenes${SCENE_SLICE:+ (slice $SCENE_SLICE of ${#ALL[@]})}, $WORKERS at a time, at most $SCENE_TIMEOUT s each (no new candidate after $SOFT_LIMIT s)"

scene_done() {  # scene_done SID: finished (complete or failed) or skipped for time
  [[ -f $OUT/scenes/$1.timeout.json ]] || grep -q '"complete": true\|"error": ' "$OUT/scenes/$1.json" 2>/dev/null
}

all_done() {  # every scene of the whole list, all slices
  local spec
  for spec in "${ALL[@]}"; do
    scene_done "${spec%:*}_s${spec##*:}" || return 1
  done
}

worker() {  # worker I takes scenes I, I + WORKERS, ...
  local i=$1 j spec task seed sid t0 rc
  for ((j = i; j < ${#SCENES[@]}; j += WORKERS)); do
    spec="${SCENES[$j]}"; task="${spec%:*}"; seed="${spec##*:}"; sid="${task}_s${seed}"
    scene_done "$sid" && continue
    t0=$(date +%s)
    sim 1 timeout -s KILL "$SCENE_TIMEOUT" "$SIM_PY" downstream/n/n_gen.py scene --task "$task" --seed "$seed" \
      --out "$OUT" --soft-limit "$SOFT_LIMIT" ${RNG[@]+"${RNG[@]}"} ${PF[@]+"${PF[@]}"} >> "$OUT/logs/$sid.log" 2>&1
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
final=0
all_done && final=1  # checked before collecting, so the collection sees every scene file
sim 1 "$SIM_PY" downstream/n/n_gen.py collect --out "$OUT" --per-family "$PER_FAMILY" ${RNG[@]+"${RNG[@]}"} \
  > "$OUT/collect.log" 2>&1
rc=$?
log "collect rc=$rc"
if [[ $final -eq 1 && $rc -eq 0 ]]; then
  cp "$OUT/instances_eval.jsonl" "$OUT/instances_final.jsonl.tmp.$$" && mv "$OUT/instances_final.jsonl.tmp.$$" "$OUT/instances_final.jsonl"
  log "all ${#ALL[@]} scenes done: instances_final.jsonl ($(wc -l < "$OUT/instances_final.jsonl") instances)"
fi
sim 1 "$SIM_PY" downstream/n/n_gen.py page --out "$OUT" --page "$OUT/page.html" >> "$OUT/n_gen.log" 2>&1
log "page rc=$?"
