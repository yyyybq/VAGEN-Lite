#!/usr/bin/env bash
# N series "can it answer, can it do" (the user's task book of 2026-11-05, item 7), one one-card sco job (its CPUs),
# one process per scene, PARALLEL at a time, each with a TIMEOUT_S limit (a scene that hangs or fails is listed in
# OUT/failed_<MODE>.txt; for truth also OUT/failed.txt as before, and its instances get no question of that kind):
#   MODE truth (default): the one-step truth of every instance (n_qa.py truth: from the start pose each of the six
#     actions once in N1's environment) -> OUT/truth/<scene>.jsonl, then OUT/truth.jsonl;
#   MODE paths: the search from the start (n_qa.py paths: every pose reachable in 12 actions, as the instances were
#     generated) and per instance the fewest actions to a box success after each first action -> OUT/paths/, then
#     OUT/paths.jsonl.
# Resumable (a scene with a file is skipped).  The simulator runs without the Mesa shader cache.  Writes only under
# OUT.
# Usage: run_n_qa_truth.sh INSTANCES OUT [PARALLEL] [TIMEOUT_S] [MODE]
#   N_QA_SIM_PY: another Python than sim_env.sh's (tests).
set -uo pipefail
INST="${1:?Usage: run_n_qa_truth.sh INSTANCES OUT [PARALLEL] [TIMEOUT_S] [MODE]}"; OUT="${2:?}"; PAR="${3:-12}"
TO="${4:-900}"; MODE="${5:-truth}"
case $MODE in truth|paths) ;; *) echo "unknown mode $MODE"; exit 2 ;; esac
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
SIM_PY="${N_QA_SIM_PY:-$SIM_PY}"
export PYTHONDONTWRITEBYTECODE=1 MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
mkdir -p "$OUT/$MODE" "$OUT/logs"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
LOG="$OUT/$MODE.log"
FAILED="$OUT/failed.txt"
[[ $MODE == paths ]] && FAILED="$OUT/failed_paths.txt"
log() { echo "[n-qa-$MODE] $* $(date -u +%FT%TZ)" | tee -a "$LOG"; }
"$SIM_PY" -c 'import json, sys; print("\n".join(sorted({json.loads(l)["scene_id"] for l in open(sys.argv[1]) if l.strip()})))' \
  "$INST" > "$OUT/scenes.txt"
log "$(grep -c . "$OUT/scenes.txt") scenes, $PAR at a time"
while read -r sid; do
  [[ -s $OUT/$MODE/$sid.jsonl ]] && continue
  (
    sim 1 timeout "$TO" "$SIM_PY" downstream/n/n_qa.py "$MODE" --instances "$INST" --scene "$sid" --out "$OUT/$MODE" \
      < /dev/null > "$OUT/logs/${MODE}_$sid.log" 2>&1
    rc=$?
    (( rc )) && echo "$sid rc=$rc" >> "$FAILED"
  ) &
  while (( $(jobs -rp | wc -l) >= PAR )); do wait -n; done
done < "$OUT/scenes.txt"
wait
"$SIM_PY" downstream/n/n_qa.py collect --truth "$OUT/$MODE" --out "$OUT/$MODE.jsonl" >> "$LOG" 2>&1
log "done: $(cat "$OUT/$MODE.jsonl" 2>/dev/null | grep -c .) instances, $(cat "$FAILED" 2>/dev/null | grep -c .) scenes failed"
