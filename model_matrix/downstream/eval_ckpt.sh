#!/usr/bin/env bash
# One StarVLA checkpoint of the model line through the downstream line's evaluation (the fixed plan
# downstream/eval/eval_d6.json, policy servers and simulator processes of
# VAGEN-Lite-downstream/downstream/eval/run_eval_ckpt.sh, read only).  Every start but Cambrian-S runs that script
# itself; for a Cambrian-S start this is the same script with one change: each policy server is started through the
# Cambrian backbone entry (cambrian_env.sh), as backbone_smoke.sh does, instead of run_starvla_mobile.sh server.
# Usage: eval_ckpt.sh START_PATH CKPT OUT SERVERS WORKERS [SHARD]   (writes only under OUT)
set -uo pipefail
START="${1:?Usage: eval_ckpt.sh START_PATH CKPT OUT SERVERS WORKERS [SHARD]}"
CKPT="${2:?}"; OUT="${3:?}"; SERVERS="${4:?}"; WORKERS="${5:?}"; SHARD="${6:-}"
DS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream
MWT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1] + "/config.json")).get("model_type") == "cambrian_qwen" else 1)' "$START" 2>/dev/null; then
  exec bash "$DS/downstream/eval/run_eval_ckpt.sh" "$CKPT" "$OUT" "$SERVERS" "$WORKERS" ${SHARD:+"$SHARD"}
fi
SHARD_ARGS=(); [[ -n "$SHARD" ]] && SHARD_ARGS=(--shard "$SHARD")
WT="$DS"  # sim_env.sh, the plan and the closed-loop client are the downstream line's
source "$DS/downstream/eval/sim_env.sh"
PLAN="$DS/downstream/eval/eval_d6.json"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NO_PROXY='*' no_proxy='*' STARVLA_ROOT HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
ADDRS=""
for g in $(seq 0 $((SERVERS - 1))); do
  port=$((5700 + g))
  (
    cd "$STARVLA_ROOT"
    source "$MWT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
    set +e
    export PYTHONPATH="$DS:$STARVLA_ROOT"
    source "$MWT/model_matrix/downstream/cambrian_env.sh"
    starvla_entry "$START"
    CUDA_VISIBLE_DEVICES=$g exec python "${ENTRY[0]}" deployment/model_server/server_policy.py --ckpt_path "$CKPT" \
      --port "$port" --use_bf16 --idle_timeout -1
  ) > "$OUT/server_$g.log" 2>&1 &
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
  sim "$THREADS" "$SIM_PY" "$DS/downstream/eval/closed_loop.py" --ckpt "$CKPT" --servers "$ADDRS" --eval-plan "$PLAN" \
    --workers "$W" "${SHARD_ARGS[@]}" --out "$OUT/episodes$SUFFIX" > "$OUT/closed_loop$SUFFIX.log" 2>&1
  r=$?; (( r )) && rc=$r
  kill "$GPID" 2>/dev/null
  echo "[eval] <<< workers=$W rc=$r $(date -u +%T) $(head -c 400 "$OUT/episodes$SUFFIX/summary.json" 2>/dev/null | tr -d '\n')"
done
exit $rc
