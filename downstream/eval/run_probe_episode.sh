#!/usr/bin/env bash
# Diagnostics run of an episode that hangs its simulator process (the user's request of 2026-10-30: why does
# OpenDrawer seed 1012 keep doing it).  One policy server on GPU 0, then for each phase (SEEDS WORKERS) closed_loop.py
# on TASK x SEEDS with WORKERS simulator processes, the probe on (closed_loop.Probe: every reset, policy call and
# simulator step timed with the contact count and largest joint speed; the Python stacks of anything stalled over
# STALL_DUMP seconds), the hang limits raised so a slow episode runs to its end (STALL_SECONDS, HANG_LIMIT) and no
# exclusion list (closed_loop applies it only with a plan file: the excluded episode is the one studied).
# Usage: run_probe_episode.sh CKPT OUT TASK SEEDS WORKERS [SEEDS WORKERS ...]   (phase i writes OUT/phase<i>/)
#   e.g. run_probe_episode.sh CKPT OUT OpenDrawer 1012 1 1004-1016 12: the episode alone, then among 12 processes
#   (about one CPU each, as in the 8-card evaluation: 96 processes on 112 vCPUs)
# (PROBE_PORT, STALL_DUMP, STALL_SECONDS, HANG_LIMIT and SERVER_SCRIPT only change for the tests.)
set -uo pipefail
CKPT="${1:?Usage: run_probe_episode.sh CKPT OUT TASK SEEDS WORKERS [SEEDS WORKERS ...]}"; OUT="${2:?}"; TASK="${3:?}"
shift 3
(( $# >= 2 && $# % 2 == 0 )) || { echo "phases come as pairs: SEEDS WORKERS"; exit 2; }
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/downstream/eval/sim_env.sh"
SERVER_SCRIPT="${SERVER_SCRIPT:-$WT/examples/evaluate/robocasa/run_starvla_mobile.sh}"
PORT="${PROBE_PORT:-5700}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY='*' no_proxy='*' STARVLA_ROOT HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export CLOSED_LOOP_STALL_DUMP="${STALL_DUMP:-120}" CLOSED_LOOP_STALL_SECONDS="${STALL_SECONDS:-5400}"
export CLOSED_LOOP_HANG_LIMIT_SECONDS="${HANG_LIMIT:-7200}"
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 CKPT="$CKPT" PORT=$PORT bash "$SERVER_SCRIPT" server --idle_timeout -1 > "$OUT/server.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$PORT" && break
  sleep 5
done
echo "[probe] server ready on $PORT $(date -u +%T)"
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 14)
rc=0 i=0
while (( $# )); do
  SEEDS=$1 W=$2; shift 2; i=$((i + 1))
  D="$OUT/phase$i"
  mkdir -p "$D"
  THREADS=$(( (CPUS - 2) / W )); (( THREADS < 1 )) && THREADS=1
  echo "[probe] phase $i: $TASK seeds $SEEDS, $W processes, $THREADS threads each $(date -u +%T)"
  CLOSED_LOOP_PROBE_DIR="$D/probe" sim "$THREADS" "$SIM_PY" "$WT/downstream/eval/closed_loop.py" --ckpt "$CKPT" \
    --servers "127.0.0.1:$PORT" --tasks "$TASK" --seeds "$SEEDS" --workers "$W" --out "$D/episodes" \
    > "$D/closed_loop.log" 2>&1
  r=$?; (( r )) && rc=$r
  echo "[probe] phase $i rc=$r $(date -u +%T)"
done
exit $rc
