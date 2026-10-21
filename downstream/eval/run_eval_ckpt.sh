#!/usr/bin/env bash
# Evaluate one StarVLA checkpoint with the fixed plan (downstream/eval/eval_d6.json) in one sco job.
#   SERVERS policy servers (run_starvla_mobile.sh server, one per GPU), WORKERS simulator processes spread over them,
#   OSMesa rendering (downstream/eval/sim_env.sh), GPU utilisation sampled every 5 s.
# Usage: run_eval_ckpt.sh CKPT OUT SERVERS WORKERS (SERVERS policy servers on GPUs 0..SERVERS-1; the job's card
# count only buys CPUs: 14 vCPU per H800).  Writes only under OUT.
set -uo pipefail
CKPT="${1:?Usage: run_eval_ckpt.sh CKPT OUT SERVERS WORKERS}"
OUT="${2:?}"; SERVERS="${3:?}"; WORKERS="${4:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/downstream/eval/sim_env.sh"
PLAN="$WT/downstream/eval/eval_d6.json"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY='*' no_proxy='*' STARVLA_ROOT HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
TASKS=$(python3 -c 'import json,sys
p=json.load(open(sys.argv[1])); t=json.load(open(sys.argv[2]))["tasks"]
print(" ".join(x["task"] for x in t if x["group"] in p["groups"]))' "$PLAN" "$WT/downstream/demos/target50_tasks.json")
SEEDS=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seeds"])' "$PLAN")
ADDRS=""
for g in $(seq 0 $((SERVERS - 1))); do
  port=$((5700 + g))
  CUDA_VISIBLE_DEVICES=$g CKPT="$CKPT" PORT=$port bash "$WT/examples/evaluate/robocasa/run_starvla_mobile.sh" server \
    > "$OUT/server_$g.log" 2>&1 &
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
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -lms 5000 > "$OUT/gpu_util.csv" &
PIDS+=($!)
CPUS=$(awk '{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 14)
THREADS=$(( (CPUS - 2 * SERVERS) / WORKERS )); (( THREADS < 1 )) && THREADS=1
sim "$THREADS" "$SIM_PY" "$WT/downstream/eval/closed_loop.py" --ckpt "$CKPT" --servers "$ADDRS" --tasks $TASKS \
  --seeds "$SEEDS" --workers "$WORKERS" --out "$OUT/episodes" > "$OUT/closed_loop.log" 2>&1
rc=$?
echo "[eval] rc=$rc $(date -u +%T) threads/process=$THREADS $(head -c 400 "$OUT/episodes/summary.json" 2>/dev/null | tr -d '\n')"
exit $rc
