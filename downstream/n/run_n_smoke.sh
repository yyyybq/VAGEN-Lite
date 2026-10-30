#!/usr/bin/env bash
# N series smoke test, part 1 (the user's decision of 2026-10-30), one one-card sco job (the simulator on CPU):
# PARTS processes of downstream/n/n_smoke.py generate over X0's scenes (each stops after PER_PART instances), then the
# review page of the first 20 instances (OUT/page.html).  Every step skips what is done.  Writes only under OUT.
# Usage: run_n_smoke.sh OUT [PARTS] [PER_PART]
set -uo pipefail
OUT="${1:?Usage: run_n_smoke.sh OUT [PARTS] [PER_PART]}"; PARTS="${2:-12}"; PER_PART="${3:-3}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
X0="${N_X0:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out/questions.jsonl}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT"
log() { echo "[n-smoke] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n_smoke.log"; }
log "generate: $PARTS processes, $PER_PART instances each"
for i in $(seq 1 "$PARTS"); do
  mkdir -p "$OUT/gen/part$i"
  sim 1 "$SIM_PY" downstream/n/n_smoke.py generate --x0 "$X0" --part "$i/$PARTS" --per-part "$PER_PART" \
    --out "$OUT/gen/part$i" > "$OUT/gen/part$i/generate.log" 2>&1 &
done
wait
log "generate: $(cat "$OUT"/gen/part*/instances.jsonl 2>/dev/null | grep -c '"instance": {') instances in $(cat "$OUT"/gen/part*/instances.jsonl 2>/dev/null | wc -l) scenes"
sim 1 "$SIM_PY" downstream/n/n_smoke.py page --gen "$OUT"/gen/part* --n 20 --out "$OUT/page.html" >> "$OUT/n_smoke.log" 2>&1
log "page rc=$?"
