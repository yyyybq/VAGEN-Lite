#!/usr/bin/env bash
# N2 (2026-11-03): the evaluation episodes with corrupted camera-A frames played again, one one-card sco job per model
# after its run_n2.sh job: for each tier folder RUN/eval/<n>, n_repair.py drop moves the corrupted episodes aside
# (OUT/episodes_corrupted.jsonl, OUT/frames_corrupted/; nothing deleted), then run_n_episodes.sh with the tier's
# adapter plays the missing ones again with WORKERS at once (default 4, against 12 in the job where they broke);
# finally every frame is scanned again (RUN/frames_scan_repaired.json).  Writes only under RUN.
# Usage: run_n2_repair.sh RUN MODEL INSTANCES [WORKERS]
#   The Q3 models get their own validation config (N_VAL_ENV, as in run_n2.sh) and run as the container's root;
#   run_n_episodes.sh gives each tier folder back to its owner, and this script the rest of RUN.
set -uo pipefail
RUN="${1:?Usage: run_n2_repair.sh RUN MODEL INSTANCES [WORKERS]}"; MODEL="${2:?}"; INSTANCES="${3:?}"; WORKERS="${4:-4}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
case $MODEL in
  Q3*) export N_VAL_ENV="${N_VAL_ENV_QWEN3:-/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_q3d/val_env.yaml}" ;;
esac
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
trap 'if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$RUN"; fi' EXIT
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
log() { echo "[n2-repair] $* $(date -u +%FT%TZ)" | tee -a "$RUN/n2.log"; }
cd "$WT"
for d in "$RUN"/eval/*/; do
  d="${d%/}"; n="$(basename "$d")"
  res="$("$VLM_PY" downstream/n/n_repair.py drop --out "$d")"
  log "tier $n: $res"
  [[ $res == *'"dropped": 0'* ]] && continue
  N_ADAPTER="$RUN/adapters/$n/adapter.pt" bash downstream/n/run_n_episodes.sh "$d" "$INSTANCES" "$MODEL" "$WORKERS" 0
  log "tier $n replayed: $(wc -l < "$d/episodes.jsonl") episodes"
done
"$VLM_PY" downstream/n/n_frames.py scan --eval "$RUN/eval" --out "$RUN/frames_scan_repaired.json" >> "$RUN/n2.log" 2>&1
log "scan: $(python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["evals"]; print([(v["corrupted_tiles"], v["episodes"]) for v in r.values()])' "$RUN/frames_scan_repaired.json")"
