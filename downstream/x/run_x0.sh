#!/usr/bin/env bash
# X0 (the user's decision 4 of 2026-10-26), one 8-card job:
#   1. smoke test: the 20 reviewed samples asked to B (stops the job within minutes if the model cannot be asked);
#   2. the questions: first frames of the 34 evaluation tasks x 15 resets (seeds 1000-1014), PARTS simulator
#      processes in parallel (downstream/x/x0_qa.py generate --part I/PARTS);
#   3. the set: about 170 per kind, questions with a movable object first (x0_qa.py select);
#   4. the four start models asked zero-shot on GPUs 0-3 in parallel (downstream/x/x0_ask.py);
#   5. the analysis: accuracy per kind with 95% intervals, unparsed replies, McNemar against B (x0_qa.py analyze).
# Each step skips what is already there, so a relaunch goes on.  Writes only under OUT.
# Usage: run_x0.sh OUT SAMPLES_JSONL
set -uo pipefail
OUT="${1:?Usage: run_x0.sh OUT SAMPLES_JSONL}"
SAMPLES="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARTS="${X0_PARTS:-48}"
VLM_PY="${X0_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
STARTS=(B S RS RD)
mkdir -p "$OUT"
log() { echo "[x0] $* $(date -u +%FT%TZ)" | tee -a "$OUT/x0.log"; }
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY='*' no_proxy='*'
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface

# 1. smoke test
if [[ ! -s "$OUT/smoke_B.jsonl" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/x/x0_ask.py --start B --questions "$SAMPLES" --out "$OUT/smoke_B.jsonl" \
    >> "$OUT/x0.log" 2>&1 || { log "smoke test failed: stop"; exit 3; }
  log "smoke test: $(grep -c '"letter": "[AB]"' "$OUT/smoke_B.jsonl") of $(wc -l < "$OUT/smoke_B.jsonl") replies parsed"
fi

# 2. questions
if [[ ! -f "$OUT/gen/DONE" ]]; then
  log "generate: $PARTS processes"
  for i in $(seq 1 "$PARTS"); do
    mkdir -p "$OUT/gen/part$i"
    sim 2 "$SIM_PY" downstream/x/x0_qa.py generate --tasks seen34 --seeds 1000-1014 --part "$i/$PARTS" \
      --out "$OUT/gen/part$i" > "$OUT/gen/part$i/generate.log" 2>&1 &
  done
  wait
  touch "$OUT/gen/DONE"
  log "generate: $(cat "$OUT"/gen/part*/scenes.jsonl | wc -l) scenes, $(cat "$OUT"/gen/part*/candidates.jsonl | wc -l) candidates"
fi

# 3. the set
if [[ ! -s "$OUT/questions.jsonl" ]]; then
  sim 1 "$SIM_PY" downstream/x/x0_qa.py select --qa "$OUT"/gen/part*/candidates.jsonl --per-kind 170 \
    --out "$OUT/questions.jsonl" >> "$OUT/x0.log" 2>&1 || { log "select failed: stop"; exit 4; }
  log "set: $(wc -l < "$OUT/questions.jsonl") questions; $(cat "$OUT/questions.stats.json" | tr -d '\n ')"
fi

# 4. ask the four starts, one GPU each
for g in 0 1 2 3; do
  s="${STARTS[$g]}"
  CUDA_VISIBLE_DEVICES=$g "$VLM_PY" downstream/x/x0_ask.py --start "$s" --questions "$OUT/questions.jsonl" \
    --out "$OUT/answers_$s.jsonl" > "$OUT/ask_$s.log" 2>&1 &
done
wait
for s in "${STARTS[@]}"; do log "asked $s: $(wc -l < "$OUT/answers_$s.jsonl") answers"; done

# 5. analysis
sim 1 "$SIM_PY" downstream/x/x0_qa.py analyze --questions "$OUT/questions.jsonl" \
  --answers B="$OUT/answers_B.jsonl" S="$OUT/answers_S.jsonl" RS="$OUT/answers_RS.jsonl" RD="$OUT/answers_RD.jsonl" \
  --out "$OUT/results.json" >> "$OUT/x0.log" 2>&1
rc=$?
log "analysis rc=$rc"
exit $rc
