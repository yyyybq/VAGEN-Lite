#!/usr/bin/env bash
# One stage-1 run (an 8-card sco job): segmented training with the evaluation inside the job (the user's decisions
# of 2026-10-22).  For each evaluation point P of EXP (downstream/train/full_state.py points; e.g. 50k, 100k, ..., the
# last step):
#   1. train to P: train_entry.py with trainer.stop_at=P; it resumes from the newest full state (DeepSpeed weights and
#      optimizer, LR schedule, random states, data position) and leaves after saving P's checkpoint and full state
#      (at the last step: final_model);
#   2. evaluate P's checkpoint here when full_state.py eval-mode says injob: 8 policy servers x 12 simulator
#      processes over the 950-episode plan (downstream/eval/run_eval_ckpt.sh CKPT EXP/evals/steps_P/out 8 96).
# Stage 1 (E1) runs have one point only, the last step (the user's decision 1 of 2026-10-22, second message: only the
# final checkpoint is evaluated in the job; E0 covers 25k-100k): train to the end, then evaluate the final checkpoint.
# Started again (the same job relaunched, or resubmitted), it continues: checkpoints that exist are not retrained,
# finished evaluations are not repeated (an interrupted one resumes), training resumes from the full state.
# Writes only under EXP: run/, evals/steps_P/out/, train.log, segments.log, stage_times.tsv, gpu.csv (GPU memory and
# use every 10 s; added 2026-10-26 for X1's memory report).
# Usage: run_segmented.sh EXP
#   TRAIN_ENTRY (2026-10-31, C1): the training entry instead of downstream/train/train_entry.py, e.g.
#   downstream/train/cotrain_entry.py (train_entry.py plus the text loss of the config's trainer.cotrain).
set -uo pipefail
EXP="${1:?Usage: run_segmented.sh EXP_DIR}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENTRY="${TRAIN_ENTRY:-$WT/downstream/train/train_entry.py}"
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
FS="$WT/downstream/train/full_state.py"
log() { echo "[segmented] $* $(date -u +%FT%TZ)" | tee -a "$EXP/segments.log"; }
stage() { echo -e "$1\t$(date -u -d @"$2" +%FT%TZ)\t$(( $(date +%s) - $2 ))\t$3" >> "$EXP/stage_times.tsv"; }

train_to() {  # train_to P: one training segment, in its own shell with the StarVLA environment
  (
    export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
    export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
    source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
    set +e
    export PYTHONPATH="$WT:$STARVLA_ROOT"
    cd "$STARVLA_ROOT"
    accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
      "$TRAIN_ENTRY" --config_yaml "$EXP/config.yaml" "trainer.stop_at=$1" \
      ${TRAIN_EXTRA:-} >> "$EXP/train.log" 2>&1
  )
}

checks() {
  (
    source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
    set +e
    export PYTHONPATH="$WT:$STARVLA_ROOT"
    read -r START DATA < <(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["start"]["name"], d["data"]["dir"])' "$EXP/experiment.json")
    python "$WT/downstream/verify_starts.py" verify --starts "$START" > "$EXP/verify_starts_at_start.json" \
      || { echo "start $START changed since it was locked"; exit 3; }
    python "$WT/downstream/demos/prepare_data.py" --out "$DATA" --check > "$EXP/data_check_at_start.json" \
      || { echo "prepared data differs from its manifest"; exit 3; }
  )
}

log "exp=$EXP points=$(python3 "$FS" points "$EXP")"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 10000 >> "$EXP/gpu.csv" &
GPU_PID=$!
trap 'kill $GPU_PID 2>/dev/null' EXIT
checks || { log "checks failed: stop"; exit 3; }
for P in $(python3 "$FS" points "$EXP"); do
  CK="$(python3 "$FS" ckpt "$EXP" "$P")"
  if [[ ! -f "$CK" ]]; then
    log "train to $P"
    t0=$(date +%s)
    train_to "$P"
    rc=$?
    stage "train_to_$P" "$t0" "$rc"
    [[ $rc -ne 0 ]] && { log "training to $P failed rc=$rc: stop"; exit $rc; }
    [[ -f "$CK" ]] || { log "training to $P ended without $CK: stop"; exit 5; }
  fi
  MODE="$(python3 "$FS" eval-mode "$EXP" "$P")"
  log "step $P: evaluation $MODE"
  if [[ "$MODE" == injob ]]; then
    t0=$(date +%s)
    bash "$WT/downstream/eval/run_eval_ckpt.sh" "$CK" "$EXP/evals/steps_$P/out" 8 96 >> "$EXP/segments.log" 2>&1
    rc=$?
    stage "eval_$P" "$t0" "$rc"
    [[ $rc -ne 0 ]] && log "evaluation of $P rc=$rc (training goes on; the watcher shows it)"
  fi
done
log "done"
