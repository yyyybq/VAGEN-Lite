#!/usr/bin/env bash
# N1 (the user's task book of 2026-11-05, item 2): N1's broken episodes played again, one one-card sco job per model.
# For each of N1's two sets that has the model's folder (n1_eval_20261031 with n1_20261031/instances_eval.jsonl,
# n1b_eval_20261102 with n1b_20261102/instances_final.jsonl):
#   n_repair.py drop --broken moves aside (nothing deleted) the episodes whose camera-A frames are colour noise or
#   near-black, and those of the instances NAMED lists for this set and model (judged broken by eye);
#   run_n_episodes.sh plays the missing ones again with WORKERS at once (default 4, against 12 in N1's jobs), with the
#   prompt N1 used (the Qwen3 models their RL chain's, N_VAL_ENV; the Qwen2.5 ones run_n_episodes.sh's default);
#   the model's frames are scanned again with n_frames.py's rules (OUT/frames_scan_repaired.json, OUT/n1_repair.log).
# The Mesa shader cache is off, as in the N2, N3 and matrix jobs.  Writes only under the model's folders.
# Usage: run_n1_repair.sh MODEL [WORKERS] [NAMED]
#   NAMED: a file of lines "SET MODEL INSTANCE" (SET: the set's folder name, e.g. n1b_eval_20261102).
#   N1_ROOT: another root than downstream_runs/n (tests).  The Qwen3 models run as the container's root
#   (sco_job.prepare as_root); each model folder is given back to OWNER_UID:OWNER_GID (default 20325) at the end.
set -uo pipefail
MODEL="${1:?Usage: run_n1_repair.sh MODEL [WORKERS] [NAMED]}"; WORKERS="${2:-4}"; NAMED="${3:-}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${N1_ROOT:-/mnt/umm/users/yinbaiqiao/downstream_runs/n}"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
case $MODEL in
  Q3*) export N_VAL_ENV="${N_VAL_ENV_QWEN3:-/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_q3d/val_env.yaml}" ;;
esac
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false MESA_SHADER_CACHE_DISABLE=true MESA_GLSL_CACHE_DISABLE=true
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
cd "$WT"
rc=0
for pair in "n1_eval_20261031:n1_20261031/instances_eval.jsonl" "n1b_eval_20261102:n1b_20261102/instances_final.jsonl"; do
  SET="${pair%%:*}"; INST="$ROOT/${pair#*:}"; OUT="$ROOT/$SET/$MODEL"
  [[ -d $OUT ]] || continue
  log() { echo "[n1-repair] $* $(date -u +%FT%TZ)" | tee -a "$OUT/n1_repair.log"; }
  NAMES=()
  if [[ -n $NAMED && -f $NAMED ]]; then
    while read -r s m i; do [[ $s == "$SET" && $m == "$MODEL" && -n $i ]] && NAMES+=("$i"); done < "$NAMED"
  fi
  EXTRA=()
  (( ${#NAMES[@]} )) && EXTRA=(--instances "${NAMES[@]}")
  res="$("$VLM_PY" downstream/n/n_repair.py drop --out "$OUT" --broken ${EXTRA[@]+"${EXTRA[@]}"})"
  log "$SET $MODEL drop: $res (named: ${NAMES[*]:-none})"
  if [[ $res != *'"dropped": 0'* ]]; then
    bash downstream/n/run_n_episodes.sh "$OUT" "$INST" "$MODEL" "$WORKERS" 0
    r=$?; (( r )) && rc=$r
    log "$SET $MODEL replay rc=$r: $(grep -c . "$OUT/episodes.jsonl") episodes of $(grep -c . "$INST")"
  fi
  mkdir -p "$OUT/.scan/$MODEL" && ln -sfn "$OUT/frames" "$OUT/.scan/$MODEL/frames"
  "$VLM_PY" downstream/n/n_frames.py scan --eval "$OUT/.scan" --out "$OUT/frames_scan_repaired.json" >> "$OUT/n1_repair.log" 2>&1
  rm -rf "$OUT/.scan"
  log "$SET $MODEL scan: $("$VLM_PY" -c 'import json,sys; r=list(json.load(open(sys.argv[1]))["evals"].values())[0]; print("corrupted episodes", r["episodes"], "broken", sorted(r["broken"]), "hints", len(r["hints"]))' "$OUT/frames_scan_repaired.json" 2>&1)"
  if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$OUT"; fi
done
exit $rc
