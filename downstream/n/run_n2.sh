#!/usr/bin/env bash
# N2 (few-shot; the user's decisions 2-3 of 2026-11-02 and 1 of 2026-11-03), one one-card sco job per model: for each
# demonstration count in TIERS, LoRA training (downstream/n/n2_train.py, the StarVLA environment's Python, GPU 0) to
# RUN/adapters/<n>/adapter.pt, then its evaluation on INSTANCES (run_n_episodes.sh with N_ADAPTER: the server merges
# the adapter into MODEL in memory; WORKERS episodes at a time) in RUN/eval/<n>.  A tier whose adapter exists is not
# trained again, and the evaluation resumes (episodes already in RUN/eval/<n>/episodes.jsonl are skipped); a tier
# whose training fails is skipped and the job ends with 1.  Writes only under RUN.
# Usage: run_n2.sh RUN MODEL DATA INSTANCES [TIERS] [WORKERS] [LIMIT]
#   DATA: n2_data.py build's folder; B, S, RS, RD train on DATA/qwen25/demos_<n>.jsonl, the Q3 models on
#   DATA/qwen3/demos_<n>.jsonl and are evaluated with the Qwen3 validation config (N_VAL_ENV), as in N1.
#   TIERS default "100 300 1000"; LIMIT 0 = all instances.
#   Qwen3 models: the job runs as the container's root (sco_job.prepare as_root; their base is readable by root only),
#   and everything under RUN is given back to OWNER_UID:OWNER_GID (default 20325) when the job ends.
set -uo pipefail
RUN="${1:?Usage: run_n2.sh RUN MODEL DATA INSTANCES [TIERS] [WORKERS] [LIMIT]}"; MODEL="${2:?}"; DATA="${3:?}"
INSTANCES="${4:?}"; TIERS="${5:-100 300 1000}"; WORKERS="${6:-12}"; LIMIT="${7:-0}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLM_PY="${N_VLM_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python}"
case $MODEL in
  Q3*) VARIANT=qwen3
       export N_VAL_ENV="${N_VAL_ENV_QWEN3:-/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_rl_20261026/rl_q3d/val_env.yaml}" ;;
  *) VARIANT=qwen25 ;;
esac
mkdir -p "$RUN/adapters" "$RUN/eval"
log() { echo "[n2] $* $(date -u +%FT%TZ)" | tee -a "$RUN/n2.log"; }
OWNER="${OWNER_UID:-20325}:${OWNER_GID:-20325}"
trap 'if [[ $(id -u) -eq 0 ]]; then chown -R "$OWNER" "$RUN"; fi' EXIT
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
cd "$WT"
log "$MODEL ($VARIANT), tiers $TIERS, instances $INSTANCES (limit $LIMIT)"
rc_all=0
for n in $TIERS; do
  A="$RUN/adapters/$n"
  if [[ ! -f $A/adapter.pt ]]; then
    log "train $MODEL on $DATA/$VARIANT/demos_$n.jsonl"
    CUDA_VISIBLE_DEVICES=0 "$VLM_PY" downstream/n/n2_train.py --model "$MODEL" --data "$DATA/$VARIANT/demos_$n.jsonl" \
      --out "$A" > "$RUN/adapters/train_$n.log" 2>&1
    rc=$?
    log "train $n rc=$rc"
    if [[ $rc -ne 0 || ! -f $A/adapter.pt ]]; then rc_all=1; continue; fi
  fi
  N_ADAPTER="$A/adapter.pt" bash downstream/n/run_n_episodes.sh "$RUN/eval/$n" "$INSTANCES" "$MODEL" "$WORKERS" "$LIMIT"
  log "evaluate $n rc=$? $(grep -c '"success": true' "$RUN/eval/$n/episodes.jsonl" 2>/dev/null) successes of $(wc -l < "$RUN/eval/$n/episodes.jsonl" 2>/dev/null)"
done
exit $rc_all
