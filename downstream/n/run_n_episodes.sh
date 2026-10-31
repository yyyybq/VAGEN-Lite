#!/usr/bin/env bash
# N series smoke test, part 2 (and later N1): MODEL answers the instances of part 1 in RoboCasa, one one-card job:
# downstream/n/n_policy_server.py (the StarVLA environment's Python, GPU 0) and downstream/n/n_episode.py (the
# simulator's Python, WORKERS episodes at a time), then the review page with the episodes (OUT/page.html).
# Resumable (episodes already in OUT/episodes.jsonl are skipped).  Writes only under OUT.
# Usage: run_n_episodes.sh OUT GEN MODEL [WORKERS] [LIMIT]
#   GEN: the smoke's generation folder (GEN/part*/instances.jsonl; the review page is made too), or N1's instance file
#   (n_gen.py collect's instances_eval.jsonl; LIMIT 0 = all).  N_PORT: the model server's port (default 5800).
set -uo pipefail
OUT="${1:?Usage: run_n_episodes.sh OUT GEN MODEL [WORKERS] [LIMIT]}"; GEN="${2:?}"; MODEL="${3:?}"
WORKERS="${4:-4}"; LIMIT="${5:-20}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
PORT="${N_PORT:-5800}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_PROXY='*' no_proxy='*'
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
mkdir -p "$OUT"
log() { echo "[n-episodes] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n_episodes.log"; }
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; wait 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/n/n_policy_server.py --model "$MODEL" --port "$PORT" > "$OUT/server.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$PORT" && break
  sleep 5
done
log "server ready ($MODEL on $PORT)"
if [[ -f $GEN ]]; then INST=("$GEN"); else INST=("$GEN"/part*/instances.jsonl); fi
sim 1 "$SIM_PY" downstream/n/n_episode.py --instances "${INST[@]}" --port "$PORT" --out "$OUT" \
  --workers "$WORKERS" --limit "$LIMIT" > "$OUT/episodes.log" 2>&1
log "episodes rc=$? $(grep -c '"success": true' "$OUT/episodes.jsonl" 2>/dev/null) successes of $(wc -l < "$OUT/episodes.jsonl" 2>/dev/null)"
if [[ ! -f $GEN ]]; then
  sim 1 "$SIM_PY" downstream/n/n_smoke.py page --gen "$GEN"/part* --n "$LIMIT" --episodes "$OUT/episodes.jsonl" \
    --out "$OUT/page.html" >> "$OUT/n_episodes.log" 2>&1
  log "page rc=$?"
fi
