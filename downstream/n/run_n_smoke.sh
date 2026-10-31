#!/usr/bin/env bash
# N series smoke test, part 1 (the user's decision of 2026-10-30), one one-card sco job (the simulator on CPU):
# PARTS processes of downstream/n/n_smoke.py generate over X0's scenes (each stops after PER_PART instances), then the
# review page of the first 20 instances (OUT/page.html).  Every step skips what is done.  Writes only under OUT.
# Usage: run_n_smoke.sh OUT [PARTS] [PER_PART]
# Environment (2026-10-31, after the first smoke):
#   HIDE_ROBOT=1     camera A's renders leave the robot out (generate --hide-robot)
#   PART_TIMEOUT     seconds a generate process may take before it is killed (default 1200; two of the first smoke's
#                    twelve stopped writing after an environment reset and the job waited on them for three hours)
#   VIEWS_SCENES     "TASK:SEED ...": first, those scenes' start views with and without the robot (OUT/views.html)
set -uo pipefail
OUT="${1:?Usage: run_n_smoke.sh OUT [PARTS] [PER_PART]}"; PARTS="${2:-12}"; PER_PART="${3:-3}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
X0="${N_X0:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out/questions.jsonl}"
PART_TIMEOUT="${PART_TIMEOUT:-1200}"
HIDE=()
[[ -n ${HIDE_ROBOT:-} ]] && HIDE=(--hide-robot)
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
log() { echo "[n-smoke] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n_smoke.log"; }
if [[ -n ${VIEWS_SCENES:-} && ! -s $OUT/views.html ]]; then
  # shellcheck disable=SC2086  # VIEWS_SCENES is a list
  sim 1 timeout -s KILL "$PART_TIMEOUT" "$SIM_PY" downstream/n/n_smoke.py views --scenes $VIEWS_SCENES \
    --out "$OUT/views.html" > "$OUT/views.log" 2>&1
  log "views rc=$?"
fi
log "generate: $PARTS processes, $PER_PART instances each${HIDE[0]:+, robot hidden}, at most $PART_TIMEOUT s each"
for i in $(seq 1 "$PARTS"); do
  mkdir -p "$OUT/gen/part$i"
  ( sim 1 timeout -s KILL "$PART_TIMEOUT" "$SIM_PY" downstream/n/n_smoke.py generate --x0 "$X0" --part "$i/$PARTS" \
      --per-part "$PER_PART" "${HIDE[@]}" --out "$OUT/gen/part$i" > "$OUT/gen/part$i/generate.log" 2>&1
    echo "[n-smoke] part $i rc=$? $(date -u +%FT%TZ)" >> "$OUT/n_smoke.log" ) &
done
wait
log "generate: $(cat "$OUT"/gen/part*/instances.jsonl 2>/dev/null | grep -c '"instance": {') instances in $(cat "$OUT"/gen/part*/instances.jsonl 2>/dev/null | wc -l) scenes"
sim 1 "$SIM_PY" downstream/n/n_smoke.py page --gen "$OUT"/gen/part* --n 20 --out "$OUT/page.html" >> "$OUT/n_smoke.log" 2>&1
log "page rc=$?"
