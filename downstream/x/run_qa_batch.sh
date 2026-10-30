#!/usr/bin/env bash
# The question-answering batch of the user's decisions of 2026-10-30 (Q0, G0, X0b, linear probes, parameter
# distances), in two one-card jobs:
#   qwen25  1. G0's questions: X0's 253 scenes reset again (PARTS simulator processes, downstream/x/g0_qa.py
#              generate), each image checked against X0's; the ~300 questions (select) and a 20-question review page
#           2. X0b-1 (the QA bank's InteriorGS views, X0's three kinds) and X0b-2 (X0's questions in the main line's
#              tagged-answer layout): downstream/x/x0b_qa.py bank / format
#           3. parameter distances of B, S, RS, RD and E1's four (downstream/x/param_dist.py, in the background)
#           4. B, S, RS, RD asked X0b-1, X0b-2 and G0, and their hidden states on X0's questions (q0_ask.py)
#   qwen3   (needs the question files of qwen25) the Qwen3 base, SFT epoch 8, RS and RD asked X0 (Q0), X0b-1, X0b-2
#           and G0, their hidden states on X0's questions, and the Qwen3 parameter distances
# Both end with the analysis of whatever answers exist so far (analysis step below; rerun it alone with STAGE
# analyze).  Every step skips what is already done, so a relaunch goes on.  Models and weights are read by path,
# nothing is copied.  Writes only under OUT.  The qwen3 job runs as the container's root (sco_job.prepare
# as_root: the Qwen3-VL base snapshot in /mnt/umm/shared_model is readable by root only, as the model line's jobs
# read it); run as root, this script gives everything under OUT back to the owner (OWNER_UID:OWNER_GID) when it ends.
# Usage: run_qa_batch.sh OUT qwen25|qwen3|analyze
set -uo pipefail
OUT="${1:?Usage: run_qa_batch.sh OUT qwen25|qwen3|analyze}"; STAGE="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
X0="${QA_X0:-/mnt/umm/users/yinbaiqiao/downstream_runs/x/X0/run_20261026/out}"
VLM_PY="${QA_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
PARTS="${QA_PARTS:-10}"
mkdir -p "$OUT"/{g0,x0b1,x0b2,answers,hidden,pd,results}
log() { echo "[qa] $* $(date -u +%FT%TZ)" | tee -a "$OUT/qa_$STAGE.log"; }
cd "$WT"
source "$WT/downstream/eval/sim_env.sh"
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
finish() {
  rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"
  if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$OUT"; fi
}
trap finish EXIT
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

ask_model() {  # MODEL: every set it has not answered yet, then its hidden states on X0's questions
  local m=$1 sets=()
  [[ $m == Q3* ]] && sets+=(--ask "$X0/questions.jsonl=$OUT/answers/x0_$m.jsonl")
  sets+=(--ask "$OUT/x0b1/questions.jsonl=$OUT/answers/x0b1_$m.jsonl")
  sets+=(--ask "$OUT/x0b2/questions.jsonl=$OUT/answers/x0b2_$m.jsonl")
  [[ -s $OUT/g0/questions.jsonl ]] && sets+=(--ask "$OUT/g0/questions.jsonl=$OUT/answers/g0_$m.jsonl")
  sets+=(--hidden "$X0/questions.jsonl=$OUT/hidden/x0_$m.npz")
  CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/x/q0_ask.py --model "$m" --smoke 2 "${sets[@]}" >> "$OUT/ask_$m.log" 2>&1
  local rc=$?
  log "asked $m: rc=$rc"
}

analysis() {
  local a=() q3=() tag=() g=() h=()
  for m in B S RS RD Q3B Q3S Q3RS Q3RD; do
    if [[ $m == Q3* ]]; then f="$OUT/answers/x0_$m.jsonl"; else f="$X0/answers_$m.jsonl"; fi
    [[ -s $f ]] && a+=("$m=$f") && [[ $m == Q3* ]] && q3+=("$m=$f")
    [[ -s $OUT/answers/x0b1_$m.jsonl ]] && tag+=("$m")
    [[ -s $OUT/answers/g0_$m.jsonl ]] && g+=("$m=$OUT/answers/g0_$m.jsonl")
    [[ -s $OUT/hidden/x0_$m.npz ]] && h+=("$m=$OUT/hidden/x0_$m.npz")
  done
  (( ${#a[@]} )) && "$VLM_PY" downstream/x/x0b_qa.py analyze --questions "$X0/questions.jsonl" --answers "${a[@]}" \
    --reference B --out "$OUT/results/q0_vs_B.json" >> "$OUT/qa_$STAGE.log" 2>&1
  (( ${#q3[@]} )) && "$VLM_PY" downstream/x/x0b_qa.py analyze --questions "$X0/questions.jsonl" --answers "${q3[@]}" \
    --reference Q3B --out "$OUT/results/q0_vs_Q3B.json" >> "$OUT/qa_$STAGE.log" 2>&1
  for v in x0b1 x0b2; do
    local args=() flag=()
    [[ $v == x0b2 ]] && flag=(--tagged)
    for m in "${tag[@]}"; do [[ -s $OUT/answers/${v}_$m.jsonl ]] && args+=("$m=$OUT/answers/${v}_$m.jsonl"); done
    (( ${#args[@]} )) || continue
    "$VLM_PY" downstream/x/x0b_qa.py analyze --questions "$OUT/$v/questions.jsonl" --answers "${args[@]}" "${flag[@]}" \
      --reference B --out "$OUT/results/${v}_vs_B.json" >> "$OUT/qa_$STAGE.log" 2>&1
    local q3a=()
    for x in "${args[@]}"; do [[ $x == Q3* ]] && q3a+=("$x"); done
    (( ${#q3a[@]} )) && "$VLM_PY" downstream/x/x0b_qa.py analyze --questions "$OUT/$v/questions.jsonl" \
      --answers "${q3a[@]}" "${flag[@]}" --reference Q3B --out "$OUT/results/${v}_vs_Q3B.json" >> "$OUT/qa_$STAGE.log" 2>&1
  done
  (( ${#g[@]} )) && "$VLM_PY" downstream/x/g0_qa.py analyze --questions "$OUT/g0/questions.jsonl" --answers "${g[@]}" \
    --out "$OUT/results/g0.json" >> "$OUT/qa_$STAGE.log" 2>&1
  (( ${#h[@]} )) && "$VLM_PY" downstream/x/probe_lp.py --questions "$X0/questions.jsonl" --hidden "${h[@]}" \
    --out "$OUT/results/probe.json" >> "$OUT/qa_$STAGE.log" 2>&1
  log "analysis: x0 ${#a[@]} models, x0b ${#tag[@]}, g0 ${#g[@]}, probes ${#h[@]}"
}

case "$STAGE" in
  qwen25)
    if [[ ! -s $OUT/g0/questions.jsonl ]]; then
      if [[ ! -f $OUT/g0/gen/DONE ]]; then
        log "G0 generate: $PARTS processes"
        for i in $(seq 1 "$PARTS"); do
          mkdir -p "$OUT/g0/gen/part$i"
          sim 1 "$SIM_PY" downstream/x/g0_qa.py generate --x0 "$X0/questions.jsonl" --part "$i/$PARTS" \
            --out "$OUT/g0/gen/part$i" > "$OUT/g0/gen/part$i/generate.log" 2>&1 &
        done
        wait
        touch "$OUT/g0/gen/DONE"
      fi
      "$VLM_PY" downstream/x/g0_qa.py select --gen "$OUT"/g0/gen/part* --out "$OUT/g0/questions.jsonl" \
        >> "$OUT/qa_$STAGE.log" 2>&1 || log "G0 select refused (see the log): G0 is not asked"
      [[ -s $OUT/g0/questions.jsonl ]] && "$VLM_PY" downstream/x/g0_qa.py samples --questions "$OUT/g0/questions.jsonl" \
        --n 20 --out "$OUT/g0/samples.html" >> "$OUT/qa_$STAGE.log" 2>&1
      log "G0: $(wc -l < "$OUT/g0/questions.jsonl" 2>/dev/null || echo 0) questions"
    fi
    [[ -s $OUT/x0b1/questions.jsonl ]] || "$VLM_PY" downstream/x/x0b_qa.py bank --out "$OUT/x0b1/questions.jsonl" \
      >> "$OUT/qa_$STAGE.log" 2>&1 || log "X0b-1 failed"
    [[ -s $OUT/x0b2/questions.jsonl ]] || "$VLM_PY" downstream/x/x0b_qa.py format --x0 "$X0/questions.jsonl" \
      --out "$OUT/x0b2/questions.jsonl" >> "$OUT/qa_$STAGE.log" 2>&1 || log "X0b-2 failed"
    log "X0b-1: $(wc -l < "$OUT/x0b1/questions.jsonl" 2>/dev/null || echo 0), X0b-2: $(wc -l < "$OUT/x0b2/questions.jsonl" 2>/dev/null || echo 0) questions"
    [[ -s $OUT/x0b1/questions.jsonl && -s $OUT/x0b2/questions.jsonl ]] && touch "$OUT/questions.DONE"  # the qwen3 job waits for it
    PD=""
    if [[ ! -s $OUT/pd/qwen25.json ]]; then
      "$VLM_PY" downstream/x/param_dist.py compute --group qwen25 --threads 3 --out "$OUT/pd/qwen25.json" \
        > "$OUT/pd/qwen25.log" 2>&1 &
      PD=$!
    fi
    for m in B S RS RD; do ask_model "$m"; done
    [[ -n $PD ]] && { wait "$PD"; log "parameter distances qwen25: rc=$?"; }
    analysis
    ;;
  qwen3)
    [[ -f $OUT/questions.DONE ]] || { log "the question sets are not made yet: run the qwen25 stage first"; exit 4; }
    PD=""
    if [[ ! -s $OUT/pd/qwen3.json ]]; then
      "$VLM_PY" downstream/x/param_dist.py compute --group qwen3 --threads 3 --out "$OUT/pd/qwen3.json" \
        > "$OUT/pd/qwen3.log" 2>&1 &
      PD=$!
    fi
    for m in Q3B Q3S Q3RS Q3RD; do ask_model "$m"; done
    [[ -n $PD ]] && { wait "$PD"; log "parameter distances qwen3: rc=$?"; }
    analysis
    ;;
  analyze)
    analysis
    ;;
  *)
    echo "unknown stage $STAGE"; exit 2 ;;
esac
log "stage $STAGE done"
