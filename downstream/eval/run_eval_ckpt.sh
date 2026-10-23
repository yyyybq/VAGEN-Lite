#!/usr/bin/env bash
# Evaluate one StarVLA checkpoint with the fixed plan (downstream/eval/eval_d6.json) in one sco job.
#   SERVERS policy servers (run_starvla_mobile.sh server, one per GPU), WORKERS simulator processes spread over them,
#   OSMesa rendering (downstream/eval/sim_env.sh), GPU utilisation sampled every 5 s.
# Usage: run_eval_ckpt.sh CKPT OUT SERVERS WORKERS (SERVERS policy servers on GPUs 0..SERVERS-1; the job's card
# count only buys CPUs: 14 vCPU per H800).  WORKERS may be a list, e.g. 48,96: the whole plan then runs once per value,
# one after the other (parallelism test), into OUT/episodes_P<n>/ with OUT/gpu_util_P<n>.csv; a single value writes
# OUT/episodes/ and OUT/gpu_util.csv.  Optional 5th argument SHARD "I/K": run only shard I of K of the plan (the
# one-card evaluation jobs of 2026-10-22; each shard has its own OUT).  Writes only under OUT.
set -uo pipefail
CKPT="${1:?Usage: run_eval_ckpt.sh CKPT OUT SERVERS WORKERS [SHARD]}"
OUT="${2:?}"; SERVERS="${3:?}"; WORKERS="${4:?}"; SHARD="${5:-}"
SHARD_ARGS=(); [[ -n "$SHARD" ]] && SHARD_ARGS=(--shard "$SHARD")
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/downstream/eval/sim_env.sh"
PLAN="$WT/downstream/eval/eval_d6.json"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY='*' no_proxy='*' STARVLA_ROOT HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
ADDRS=""
# --idle_timeout -1 (StarVLA's websocket server: no idle watchdog): the server's default closes it after 30 minutes
# without a request, which broke reruns late in an evaluation (S 50k, LoadDishwasher 1002: ConnectionClosedOK).  The
# user's decision 4 of 2026-10-23: the servers stay up for the whole evaluation; cleanup() stops them at the end.
for g in $(seq 0 $((SERVERS - 1))); do
  port=$((5700 + g))
  CUDA_VISIBLE_DEVICES=$g CKPT="$CKPT" PORT=$port bash "$WT/examples/evaluate/robocasa/run_starvla_mobile.sh" server \
    --idle_timeout -1 > "$OUT/server_$g.log" 2>&1 &
  PIDS+=($!)
  ADDRS="${ADDRS:+$ADDRS,}127.0.0.1:$port"
done
for port in $(seq 5700 $((5700 + SERVERS - 1))); do
  for i in $(seq 1 180); do
    python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$port" && break
    sleep 5
  done
done
echo "[eval] servers ready: $ADDRS $(date -u +%T)"
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 14)
IFS=, read -r -a PLIST <<< "$WORKERS"
rc=0
for W in "${PLIST[@]}"; do
  if (( ${#PLIST[@]} == 1 )); then SUFFIX=""; else SUFFIX="_P$W"; fi
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -lms 5000 \
    > "$OUT/gpu_util$SUFFIX.csv" &
  GPID=$!
  THREADS=$(( (CPUS - 2 * SERVERS) / W )); (( THREADS < 1 )) && THREADS=1
  echo "[eval] >>> workers=$W threads/process=$THREADS $(date -u +%T)"
  sim "$THREADS" "$SIM_PY" "$WT/downstream/eval/closed_loop.py" --ckpt "$CKPT" --servers "$ADDRS" --eval-plan "$PLAN" \
    --workers "$W" "${SHARD_ARGS[@]}" --out "$OUT/episodes$SUFFIX" > "$OUT/closed_loop$SUFFIX.log" 2>&1
  r=$?; (( r )) && rc=$r
  kill "$GPID" 2>/dev/null
  echo "[eval] <<< workers=$W rc=$r $(date -u +%T) $(head -c 400 "$OUT/episodes$SUFFIX/summary.json" 2>/dev/null | tr -d '\n')"
done
exit $rc
