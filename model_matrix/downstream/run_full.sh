#!/usr/bin/env bash
# A full downstream run of the model line (an 8-card sco job; the user's approval of DS-Q3B-100-s1, 2026-10-26): the
# downstream line's stage-1 run (VAGEN-Lite-downstream/downstream/train/run_segmented.sh, read only, by path) for a
# start of this line -- the same segments (downstream/train/full_state.py points), training entry
# (downstream/train/train_entry.py with trainer.stop_at), full-state resume, in-job evaluation of the final
# checkpoint and plan.  Three differences, all about the start:
#   - the start check is this line's lock (make_config --check-start, the downstream line's fingerprint rule) instead
#     of the downstream registry, which does not hold this line's starts;
#   - a Cambrian-S start trains and serves through the Cambrian backbone entry (cambrian_env.sh, eval_ckpt.sh);
#   - the job runs as root (the Qwen3-VL shards are root-only) and gives EXP back to its owner every 30 minutes and at
#     the end (the downstream line runs as the owner).
# Started again (resubmitted), it continues as run_segmented.sh does.  Writes only under EXP.  Usage: run_full.sh EXP
set -uo pipefail
EXP="${1:?Usage: run_full.sh EXP_DIR}"
MWT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
DS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
FS="$DS/downstream/train/full_state.py"
OWNER=$(stat -c %u:%g "$EXP")
( while sleep 1800; do chown -R "$OWNER" "$EXP" 2>/dev/null; done ) &
CHOWN_PID=$!
trap 'kill "$CHOWN_PID" 2>/dev/null; chown -R "$OWNER" "$EXP" 2>/dev/null || true' EXIT
log() { echo "[full] $* $(date -u +%FT%TZ)" | tee -a "$EXP/segments.log"; }
stage() { echo -e "$1\t$(date -u -d @"$2" +%FT%TZ)\t$(( $(date +%s) - $2 ))\t$3" >> "$EXP/stage_times.tsv"; }
read -r START_PATH DATA < <(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); x=d["data"]; print(d["start"]["path"], x["dir"] if isinstance(x, dict) else x)' "$EXP/experiment.json")

train_to() {  # train_to P: one training segment, in its own shell with the StarVLA environment
  (
    export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
    export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
    source "$MWT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
    set +e
    export PYTHONPATH="$DS:$STARVLA_ROOT"
    source "$MWT/model_matrix/downstream/cambrian_env.sh"
    starvla_entry "$START_PATH"
    cd "$STARVLA_ROOT"
    accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
      ${ENTRY[@]+"${ENTRY[@]}"} "$DS/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" \
      "trainer.stop_at=$1" ${TRAIN_EXTRA:-} >> "$EXP/train.log" 2>&1
  )
}

checks() {
  (
    source "$MWT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
    set +e
    cd "$MWT"
    PYTHONPATH="$MWT" python -m model_matrix.downstream.make_config --check-start "$EXP" \
      > "$EXP/verify_start_at_start.json" || { echo "the start changed since it was locked"; exit 3; }
    PYTHONPATH="$DS:$STARVLA_ROOT" python "$DS/downstream/demos/prepare_data.py" --out "$DATA" --check \
      > "$EXP/data_check_at_start.json" || { echo "prepared data differs from its manifest"; exit 3; }
  )
}

log "exp=$EXP start=$START_PATH points=$(python3 "$FS" points "$EXP")"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
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
    bash "$MWT/model_matrix/downstream/eval_ckpt.sh" "$START_PATH" "$CK" "$EXP/evals/steps_$P/out" 8 96 \
      >> "$EXP/segments.log" 2>&1
    rc=$?
    stage "eval_$P" "$t0" "$rc"
    [[ $rc -ne 0 ]] && log "evaluation of $P rc=$rc"
  fi
done
log "done"
