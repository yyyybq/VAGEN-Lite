#!/usr/bin/env bash
# X4 follow-up (the user's decision of 2026-10-27), a one-card job per model: the loss of the main line's expert
# answers (downstream/x/x4_ppl.py score, the starVLA environment's transformers) into X4/ppl/<NAME>.jsonl, then the
# analysis of every model scored so far (x4_ppl.py analyze) into X4/ppl/results.  Resumable.
# Usage: run_x4_ppl.sh NAME MODEL_PATH [PPL_DIR]
set -uo pipefail
NAME="${1:?Usage: run_x4_ppl.sh NAME MODEL_PATH [PPL_DIR]}"
MODEL="${2:?}"
DIR="${3:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X4/ppl}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${X4_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"  # X4_* only change for the tests
PY="${X4_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"  # has scipy (Wilcoxon)
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$DIR"
echo "[x4-ppl] $NAME: $MODEL $(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$VLM_PY" "$WT/downstream/x/x4_ppl.py" score --model "$MODEL" \
  --out "$DIR/$NAME.jsonl" > "$DIR/$NAME.log" 2>&1
rc=$?
echo "[x4-ppl] $NAME: score rc=$rc $(tail -1 "$DIR/$NAME.log") $(date -u +%FT%TZ)"
"$PY" "$WT/downstream/x/x4_ppl.py" analyze --dir "$DIR" || echo "[x4-ppl] analysis failed (scores kept)"
exit $rc
