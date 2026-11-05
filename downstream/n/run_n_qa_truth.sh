#!/usr/bin/env bash
# N series "can it answer, can it do" (the user's task book of 2026-11-05, item 7), one one-card sco job (its CPUs):
# the one-step truth of every instance (n_qa.py truth: from the start pose each of the six actions once in N1's
# environment), one process per scene, PARALLEL at a time, each with a TIMEOUT_S limit (a scene that hangs or fails
# is listed in OUT/failed.txt and its instances get no best_action question); then OUT/truth.jsonl (n_qa.py collect).
# Resumable (a scene with a truth file is skipped).  The simulator runs without the Mesa shader cache.  Writes only
# under OUT.
# Usage: run_n_qa_truth.sh INSTANCES OUT [PARALLEL] [TIMEOUT_S]
#   N_QA_SIM_PY: another Python than sim_env.sh's (tests).
set -uo pipefail
INST="${1:?Usage: run_n_qa_truth.sh INSTANCES OUT [PARALLEL] [TIMEOUT_S]}"; OUT="${2:?}"; PAR="${3:-12}"; TO="${4:-900}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
SIM_PY="${N_QA_SIM_PY:-$SIM_PY}"
export PYTHONDONTWRITEBYTECODE=1 MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
mkdir -p "$OUT/truth" "$OUT/logs"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
log() { echo "[n-qa-truth] $* $(date -u +%FT%TZ)" | tee -a "$OUT/truth.log"; }
"$SIM_PY" -c 'import json, sys; print("\n".join(sorted({json.loads(l)["scene_id"] for l in open(sys.argv[1]) if l.strip()})))' \
  "$INST" > "$OUT/scenes.txt"
log "$(grep -c . "$OUT/scenes.txt") scenes, $PAR at a time"
while read -r sid; do
  [[ -s $OUT/truth/$sid.jsonl ]] && continue
  (
    sim 1 timeout "$TO" "$SIM_PY" downstream/n/n_qa.py truth --instances "$INST" --scene "$sid" --out "$OUT/truth" \
      < /dev/null > "$OUT/logs/$sid.log" 2>&1
    rc=$?
    (( rc )) && echo "$sid rc=$rc" >> "$OUT/failed.txt"
  ) &
  while (( $(jobs -rp | wc -l) >= PAR )); do wait -n; done
done < "$OUT/scenes.txt"
wait
"$SIM_PY" downstream/n/n_qa.py collect --truth "$OUT/truth" --out "$OUT/truth.jsonl" >> "$OUT/truth.log" 2>&1
log "done: $(cat "$OUT/truth.jsonl" 2>/dev/null | grep -c .) instances, $(cat "$OUT/failed.txt" 2>/dev/null | grep -c .) scenes failed"
