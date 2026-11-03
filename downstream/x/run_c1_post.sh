#!/usr/bin/env bash
# C1's evaluations after training (the user's decisions 5 of 2026-11-02 and 5 of 2026-11-03; design
# reports/C1_DESIGN_20261028.md "评测"), the one-card part, one sco job, in the order the user gave (X3 runs before it
# in its own 8-card jobs: run_x3.sh C1-B-030-s1 and DS-B-030-s1):
#   X4  1. C1's VLM exported once (run_x4_export.sh X4/hf C1-B-030-200k CKPT; the one copy the brief allows, ~16 GB);
#       2. the main line's v2 validation (413) and test (292) sets on that copy (the main checkout's
#          sco_interiorgs_v10_train.sh eval, no_think, as X4 did for E1), into X4/eval_{val,test}_v2/C1-B-030-200k;
#       3. the loss of the main line's expert answers (run_x4_ppl.sh): C1 from the copy, DS-B-030-s1 from its
#          checkpoint in memory (X4_STARVLA_PT, B's architecture; no copy);
#   X0  4. the 510 questions (q0_ask.py --model B --starvla-pt, both from their checkpoints, no copy) into
#          X/C1/qa/answers, then the paired analysis against B and against DS-B-030-s1 (x0b_qa.py analyze) into
#          X/C1/qa/results.
# Every step skips what is done (an export with starvla_export.json, an evaluation with done.json; scoring and asking
# resume), so a relaunch goes on.  Reads the checkpoints by path; writes under X4 and X/C1 only.
# Usage: run_c1_post.sh [C1_EXP] [REF_EXP]
set -uo pipefail
RUNS="${C1P_RUNS:-/mnt/umm/users/yinbaiqiao/downstream_runs}"  # C1P_* only change for the tests
C1="${1:-C1-B-030-s1}"; REF="${2:-DS-B-030-s1}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="${C1P_MAIN:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
VLM_PY="${C1P_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
B_PATH="${C1P_B_PATH:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/inputs/base_7b}"
X4="$RUNS/x/X4"; X0="$RUNS/x/X0/run_20261026/out"; QA="$RUNS/x/C1/qa"
CK="$RUNS/experiments/$C1/run/final_model/pytorch_model.pt"
RK="$RUNS/experiments/$REF/run/final_model/pytorch_model.pt"
NAME="${C1%-s1}-200k"; RNAME="$REF-200k"  # C1-B-030-200k, DS-B-030-s1-200k
mkdir -p "$QA/answers" "$QA/results" "$X4/jobs"
log() { echo "[c1-post] $* $(date -u +%FT%TZ)" | tee -a "$QA/c1_post.log"; }
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
for f in "$CK" "$RK"; do [[ -f $f ]] || { log "no checkpoint $f"; exit 6; }; done
cd "$WT"
rc_all=0
# X4 1: the one copy
bash downstream/x/run_x4_export.sh "$X4/hf" "$NAME" "$CK" || { log "export failed"; exit 7; }
log "export: $(du -sh "$X4/hf/$NAME" | cut -f1) in $X4/hf/$NAME"
# X4 2: the v2 sets
for set in val test; do
  if [[ -f $X4/eval_${set}_v2/$NAME/done.json ]]; then log "v2 $set: done before"; continue; fi
  mkdir -p "$X4/jobs/${set}_$NAME"
  bash "${C1P_EVAL:-$MAIN/examples/train/active_spatial/sco_interiorgs_v10_train.sh}" eval "$X4/jobs/${set}_$NAME" \
    "$X4/eval_${set}_v2" "$NAME" "$X4/hf/$NAME" no_think > "$X4/jobs/${set}_$NAME/eval.log" 2>&1
  rc=$?
  log "v2 $set: rc=$rc"
  (( rc )) && rc_all=$rc
done
# X4 3: the answers' loss
bash downstream/x/run_x4_ppl.sh "$NAME" "$X4/hf/$NAME" "$X4/ppl" || rc_all=$?
X4_STARVLA_PT="$RK" bash downstream/x/run_x4_ppl.sh "$RNAME" "$B_PATH" "$X4/ppl" || rc_all=$?
log "X4 answer loss done (rc so far $rc_all)"
# X0 4: the 510 questions
for e in "$C1" "$REF"; do
  CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/x/q0_ask.py --model B --starvla-pt "$RUNS/experiments/$e/run/final_model/pytorch_model.pt" \
    --smoke 2 --ask "$X0/questions.jsonl=$QA/answers/x0_$e.jsonl" >> "$QA/ask_$e.log" 2>&1
  rc=$?
  log "X0 asked $e: rc=$rc $(wc -l < "$QA/answers/x0_$e.jsonl" 2>/dev/null) answers"
  (( rc )) && rc_all=$rc
done
for ref in B "$REF"; do
  "$VLM_PY" downstream/x/x0b_qa.py analyze --questions "$X0/questions.jsonl" --answers "B=$X0/answers_B.jsonl" \
    "$REF=$QA/answers/x0_$REF.jsonl" "$C1=$QA/answers/x0_$C1.jsonl" --reference "$ref" \
    --out "$QA/results/x0_vs_$ref.json" >> "$QA/c1_post.log" 2>&1 || rc_all=$?
done
log "done (rc $rc_all)"
exit $rc_all
