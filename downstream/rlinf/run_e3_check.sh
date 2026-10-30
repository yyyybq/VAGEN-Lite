#!/usr/bin/env bash
# The E3 check (the user's request of 2026-10-30, at most 5 card-hours), one one-card job: a policy server for the
# start E3-B began from (DS-B-030-s1's final model, read by path), then downstream/rlinf/e3_check.py:
#   obs      (in the background, one process) the 20 first frames side by side, the robot's action layout
#   run      the 2 x 2 cells (action path gym / rlinf x split target / pretrain), 3 E3 tasks x SEEDS, WORKERS processes
#   rejudge  20 failed rlinf-path episodes judged again on their saved final states
#   report   OUT/report.json, report.md
# Each step skips what is done (run goes on from its episodes.jsonl).  Writes only under OUT.
# Usage: run_e3_check.sh OUT [CKPT] [SEEDS] [WORKERS] [CELLS]
#   CELLS (2026-10-30 evening, the second round): e.g. noise/target,noise/pretrain, added to the same OUT (obs and
#   rejudge are done already, so only these cells run; the report covers every cell)
# (PROBE_PORT and SERVER_SCRIPT only change for the tests.)
set -uo pipefail
OUT="${1:?Usage: run_e3_check.sh OUT [CKPT] [SEEDS] [WORKERS] [CELLS]}"
CKPT="${2:-/mnt/umm/users/yinbaiqiao/downstream_runs/experiments/DS-B-030-s1/run/final_model/pytorch_model.pt}"
SEEDS="${3:-1000-1009}"; WORKERS="${4:-11}"; CELLS="${5:-}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
SERVER_SCRIPT="${SERVER_SCRIPT:-$WT/examples/evaluate/robocasa/run_starvla_mobile.sh}"
PORT="${PROBE_PORT:-5700}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY='*' no_proxy='*' STARVLA_ROOT HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
mkdir -p "$OUT"
log() { echo "[e3_check] $* $(date -u +%FT%TZ)" | tee -a "$OUT/e3_check.log"; }
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 CKPT="$CKPT" PORT=$PORT bash "$SERVER_SCRIPT" server --idle_timeout -1 > "$OUT/server.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$PORT" && break
  sleep 5
done
log "server ready on $PORT ($CKPT)"
if [[ ! -s $OUT/obs/frames.json ]]; then
  sim 1 "$SIM_PY" downstream/rlinf/e3_check.py obs --ckpt "$CKPT" --out "$OUT" > "$OUT/obs.log" 2>&1 &
  OBS=$!
else
  OBS=""
fi
sim 1 "$SIM_PY" downstream/rlinf/e3_check.py run --ckpt "$CKPT" --servers "127.0.0.1:$PORT" --seeds "$SEEDS" \
  --workers "$WORKERS" --cells "$CELLS" --out "$OUT" >> "$OUT/run.log" 2>&1
log "run rc=$?"
[[ -n $OBS ]] && { wait "$OBS"; log "obs rc=$?"; }
[[ -s $OUT/rejudge.json ]] || { sim 1 "$SIM_PY" downstream/rlinf/e3_check.py rejudge --out "$OUT" > "$OUT/rejudge.log" 2>&1; log "rejudge rc=$?"; }
sim 1 "$SIM_PY" downstream/rlinf/e3_check.py report --out "$OUT" >> "$OUT/e3_check.log" 2>&1
log "report rc=$?"
