#!/usr/bin/env bash
# N3 (the transfer ladder; the user's decision 2 of 2026-11-03): run_n_episodes.sh with n3_episode.py, one tier of one
# model: downstream/n/n_policy_server.py (the StarVLA environment's Python, GPU 0) and n3_episode.py (the simulator's
# Python, WORKERS episodes at a time) with collision rule N3_COLLISION (robot or sphere; default robot) and the
# instances' own camera.  Resumable (episodes already in OUT/episodes.jsonl are skipped).  At the end every frame is
# checked for colour noise (n_frames.py scan: OUT/frames_scan.json).  Writes only under OUT.
# Usage: run_n3_episodes.sh OUT INSTANCES MODEL [WORKERS] [LIMIT]
#   N_PORT (default 5800); N_VAL_ENV: the validation config whose prompt the model sees (the Qwen3 models: their own
#   chain's, as in N1); N_HISTORY=1: every call carries the episode's earlier turns (n_episode.py --history, tier g).  The simulator runs without the Mesa shader cache (run_n2.sh says why).
#   Qwen3 models: run as the container's root (their base is readable by root only); everything under OUT is given back
#   to OWNER_UID:OWNER_GID (default 20325) when it ends.
set -uo pipefail
OUT="${1:?Usage: run_n3_episodes.sh OUT INSTANCES MODEL [WORKERS] [LIMIT]}"; INST="${2:?}"; MODEL="${3:?}"
WORKERS="${4:-12}"; LIMIT="${5:-0}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
PORT="${N_PORT:-5800}"
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false NO_PROXY='*' no_proxy='*'
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
mkdir -p "$OUT"
log() { echo "[n3-episodes] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n_episodes.log"; }
PIDS=()
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
  rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"
  if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$OUT"; fi
}
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/n/n_policy_server.py --model "$MODEL" --port "$PORT" > "$OUT/server.log" 2>&1 &
PIDS+=($!)
for i in $(seq 1 180); do
  python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(("127.0.0.1",int(sys.argv[1]))))' "$PORT" && break
  sleep 5
done
log "server ready ($MODEL on $PORT); collision ${N3_COLLISION:-robot}; history ${N_HISTORY:-0}"
VAL=()
[[ -n ${N_VAL_ENV:-} ]] && VAL=(--val-env "$N_VAL_ENV")
[[ ${N_HISTORY:-0} == 1 ]] && VAL+=(--history)
sim 1 "$SIM_PY" downstream/n/n3_episode.py --collision "${N3_COLLISION:-robot}" --instances "$INST" --port "$PORT" \
  --out "$OUT" --workers "$WORKERS" --limit "$LIMIT" ${VAL[@]+"${VAL[@]}"} > "$OUT/episodes.log" 2>&1
rc=$?
log "episodes rc=$rc $(grep -c '"success": true' "$OUT/episodes.jsonl" 2>/dev/null) successes of $(wc -l < "$OUT/episodes.jsonl" 2>/dev/null)"
mkdir -p "$OUT/.scan/$MODEL" && ln -sfn "$OUT/frames" "$OUT/.scan/$MODEL/frames"
"$VLM_PY" downstream/n/n_frames.py scan --eval "$OUT/.scan" --out "$OUT/frames_scan.json" >> "$OUT/n_episodes.log" 2>&1
rm -rf "$OUT/.scan"
exit $rc  # the episode runner's (11-03: a run that stopped before any episode ended with 0)
