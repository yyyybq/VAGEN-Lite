#!/usr/bin/env bash
# X0 (510 questions) and G0 (300) asked to Qwen3-VL folders given by path (the main line's request of 2026-11-06,
# approved by the user on 2026-11-06, decision 4: "X0 / G0 问 12 份权重现在做"), the way the question-answering batch
# of 2026-10-30 asked them (q0_ask.py, greedy, X0's and G0's question files unchanged), then the analysis against
# the Qwen3 base's answers of that batch: per model the accuracy, the replies that cannot be read, and the exact
# McNemar test against Q3B on the same questions (X0: x0b_qa.py analyze --reference Q3B; G0 per kind: g0_qa.py
# analyze, every model named Q3* is set against Q3B).
# MODELS is a file of lines "NAME PATH" (NAME without spaces; the answers are answers/x0_Q3_NAME.jsonl and
# answers/g0_Q3_NAME.jsonl, so the analysis sets each against Q3B).  Folders are read by path, nothing is copied.
# One one-card job; run as the container's root (a folder without preprocessor_config.json takes the processor of
# the root-only Qwen3 base snapshot), and everything under OUT is given back to the owner when it ends.
# Every model skips the questions it has already answered, so a relaunch goes on.
# Usage: run_qa_paths.sh OUT MODELS [QA]   (QA: the 2026-10-30 batch folder with g0/questions.jsonl and Q3B's answers)
set -uo pipefail
OUT="${1:?Usage: run_qa_paths.sh OUT MODELS [QA]}"; MODELS="${2:?}"
QA="${3:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/QA_20261030}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
X0="${QA_X0:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out}"
VLM_PY="${QA_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
mkdir -p "$OUT"/{answers,results}
log() { echo "[qa-paths] $* $(date -u +%FT%TZ)" | tee -a "$OUT/qa_paths.log"; }
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
finish() { if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$OUT"; fi; }
trap finish EXIT
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$WT"
for f in "$X0/questions.jsonl" "$QA/g0/questions.jsonl" "$QA/answers/x0_Q3B.jsonl" "$QA/answers/g0_Q3B.jsonl"; do
  [[ -s $f ]] || { log "missing $f"; exit 2; }
done
rc_all=0
names=()
while read -r name path; do
  [[ -z ${name:-} || $name == \#* ]] && continue
  names+=("$name")
  if [[ ! -d $path ]]; then log "$name: no folder $path"; rc_all=1; continue; fi
  CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/x/q0_ask.py --model Q3B --model-path "$path" --smoke 2 \
    --ask "$X0/questions.jsonl=$OUT/answers/x0_Q3_$name.jsonl" \
    --ask "$QA/g0/questions.jsonl=$OUT/answers/g0_Q3_$name.jsonl" >> "$OUT/ask_$name.log" 2>&1
  rc=$?
  log "asked $name ($path): rc=$rc"
  (( rc == 0 )) || rc_all=1
done < "$MODELS"
x0=("Q3B=$QA/answers/x0_Q3B.jsonl") g0=("Q3B=$QA/answers/g0_Q3B.jsonl")
for name in "${names[@]}"; do
  [[ -s $OUT/answers/x0_Q3_$name.jsonl ]] && x0+=("Q3_$name=$OUT/answers/x0_Q3_$name.jsonl")
  [[ -s $OUT/answers/g0_Q3_$name.jsonl ]] && g0+=("Q3_$name=$OUT/answers/g0_Q3_$name.jsonl")
done
"$VLM_PY" downstream/x/x0b_qa.py analyze --questions "$X0/questions.jsonl" --answers "${x0[@]}" --reference Q3B \
  --out "$OUT/results/x0_vs_Q3B.json" >> "$OUT/qa_paths.log" 2>&1 || rc_all=1
"$VLM_PY" downstream/x/g0_qa.py analyze --questions "$QA/g0/questions.jsonl" --answers "${g0[@]}" \
  --out "$OUT/results/g0.json" >> "$OUT/qa_paths.log" 2>&1 || rc_all=1
log "analysis: x0 $(( ${#x0[@]} - 1 )) models, g0 $(( ${#g0[@]} - 1 )) models; rc=$rc_all"
exit "$rc_all"
