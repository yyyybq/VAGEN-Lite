#!/usr/bin/env bash
# N series "can it answer, can it do" (the user's task book of 2026-11-05, item 7), one one-card sco job per family:
# each model of the family answers OUT/questions.jsonl zero-shot (downstream/x/q0_ask.py, greedy, X0's 16 new tokens)
# into OUT/answers/<MODEL>.jsonl; resumable (answered questions are skipped).  The qwen3 job runs as the container's
# root (the Qwen3-VL base snapshot is readable by root only) and gives OUT back to OWNER_UID:OWNER_GID (default
# 20325) at the end.  Writes only under OUT.
# Usage: run_n_qa_ask.sh OUT qwen25|qwen3
#   N_VLM_PY: another Python than the StarVLA environment's (tests).
set -uo pipefail
OUT="${1:?Usage: run_n_qa_ask.sh OUT qwen25|qwen3}"; FAMILY="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
case $FAMILY in
  qwen25) MODELS=(B S RS RD) ;;
  qwen3) MODELS=(Q3B Q3S Q3RS Q3RD) ;;
  *) echo "unknown family $FAMILY"; exit 2 ;;
esac
cd "$WT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
trap 'if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$OUT"; fi' EXIT
mkdir -p "$OUT/answers"
log() { echo "[n-qa-ask] $* $(date -u +%FT%TZ)" | tee -a "$OUT/ask_$FAMILY.log"; }
rc_all=0
for m in "${MODELS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/x/q0_ask.py --model "$m" --smoke 2 \
    --ask "$OUT/questions.jsonl=$OUT/answers/$m.jsonl" >> "$OUT/ask_$m.log" 2>&1
  rc=$?
  log "$m: rc=$rc, $(cat "$OUT/answers/$m.jsonl" 2>/dev/null | grep -c .) answers"
  (( rc )) && rc_all=$rc
done
exit $rc_all
