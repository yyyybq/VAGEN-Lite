#!/usr/bin/env bash
# Diagnostics run under the full evaluation load (the user's request of 2026-10-30: why OpenDrawer seed 1012 keeps
# hanging its simulator process).  The one-card probe (run_probe_episode.sh, pt-jhwj97iu) ran that episode normally,
# alone and among 12 processes.  In the 8-card evaluations (96 simulator processes) workers are killed as hung in
# almost every run (0-9 per evaluation, concentrated on a few long episodes: PrepareCoffee 1003 in 25 of 55,
# SearingMeat 1035 in 20, ...; their retries finish), never in the 12-process ones.  This runs one plan the way an
# 8-card evaluation does (SERVERS policy servers on GPUs 0..SERVERS-1, WORKERS simulator processes) with the probe on
# (closed_loop.Probe: every reset, policy call and simulator step timed with the contact count; the Python stacks of
# anything stalled over STALL_DUMP seconds), the hang limits raised so a slow phase can finish and show how long it
# takes (STALL_SECONDS 1800, HANG_LIMIT 10800), no exclusion list (the excluded episode is the one studied), and the
# machine's load sampled every 30 s into OUT/load.csv (load average, available memory, the job's CPU time and
# memory from its cgroup).  Default plan and checkpoint: X3-RD's (downstream/x/x3_plan.json, E1-RD 200k), the run in
# which 1012 hung three times.  Writes only under OUT.
# Usage: run_probe_full.sh CKPT OUT PLAN [SERVERS WORKERS]
# (PROBE_PORT, STALL_DUMP, STALL_SECONDS, HANG_LIMIT, SERVER_SCRIPT and LOAD_EVERY only change for the tests.)
set -uo pipefail
CKPT="${1:?Usage: run_probe_full.sh CKPT OUT PLAN [SERVERS WORKERS]}"; OUT="${2:?}"; PLAN="${3:?}"
SERVERS="${4:-8}"; WORKERS="${5:-96}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/downstream/eval/sim_env.sh"
SERVER_SCRIPT="${SERVER_SCRIPT:-$WT/examples/evaluate/robocasa/run_starvla_mobile.sh}"
BASE_PORT="${PROBE_PORT:-5700}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY='*' no_proxy='*' STARVLA_ROOT HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export CLOSED_LOOP_STALL_DUMP="${STALL_DUMP:-120}" CLOSED_LOOP_STALL_SECONDS="${STALL_SECONDS:-1800}"
export CLOSED_LOOP_HANG_LIMIT_SECONDS="${HANG_LIMIT:-10800}" CLOSED_LOOP_PROBE_DIR="$OUT/probe"
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
# the machine's load: utc, 1/5/15-minute load average, MemAvailable (kB), the cgroup's CPU time (us) and memory (B)
(
  echo "utc_seconds,load1,load5,load15,mem_available_kb,cgroup_cpu_usec,cgroup_memory_bytes"
  while true; do
    read -r l1 l5 l15 _ < /proc/loadavg
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    cpu=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null)
    mem=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
    echo "$(date +%s),$l1,$l5,$l15,$avail,$cpu,$mem"
    sleep "${LOAD_EVERY:-30}"
  done
) > "$OUT/load.csv" &
PIDS+=($!)
ADDRS=""
for g in $(seq 0 $((SERVERS - 1))); do
  port=$((BASE_PORT + g))
  CUDA_VISIBLE_DEVICES=$g CKPT="$CKPT" PORT=$port bash "$SERVER_SCRIPT" server --idle_timeout -1 \
    > "$OUT/server_$g.log" 2>&1 &
  PIDS+=($!)
  ADDRS="${ADDRS:+$ADDRS,}127.0.0.1:$port"
done
for port in $(seq "$BASE_PORT" $((BASE_PORT + SERVERS - 1))); do
  for i in $(seq 1 180); do
    python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$port" && break
    sleep 5
  done
done
echo "[probe-full] servers ready: $ADDRS $(date -u +%T)"
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 14)
THREADS=$(( (CPUS - 2 * SERVERS) / WORKERS )); (( THREADS < 1 )) && THREADS=1
echo "[probe-full] plan $PLAN, $WORKERS processes, $THREADS threads each, no exclusion list $(date -u +%T)"
sim "$THREADS" "$SIM_PY" "$WT/downstream/eval/closed_loop.py" --ckpt "$CKPT" --servers "$ADDRS" --eval-plan "$PLAN" \
  --exclusions "" --workers "$WORKERS" --out "$OUT/episodes" > "$OUT/closed_loop.log" 2>&1
rc=$?
echo "[probe-full] rc=$rc $(date -u +%T) $(head -c 400 "$OUT/episodes/summary.json" 2>/dev/null | tr -d '\n')"
exit $rc
