#!/usr/bin/env bash
# The full-state resume test (the user's decision 1 of 2026-10-22), one 8-card sco job.  ROOT holds two experiments
# made by make_experiment.py with the same config (step-2 data, MAX steps, full state every INTERVAL steps, eval point
# STOP, logging every step):
#   A  ROOT/RESUME-A  trained from 0 to MAX without a break;
#   B  ROOT/RESUME-B  trained from 0, killed (SIGKILL to every process of the launch) once step KILL_AT is logged,
#                     resumed with trainer.stop_at=STOP (leaves after STOP's checkpoint and full state), resumed again
#                     to MAX: the two moves of a segmented stage-1 job;
#   then resume_check.py compares A and B (ROOT/resume_check.json), and, with EVAL_SHARD set, B's STOP checkpoint is
#   evaluated on that shard of the plan with 8 policy servers x 96 simulator processes (ROOT/eval_P96/), to check the
#   in-job evaluation and measure its time per episode;
#   last (EXPORT_TEST=1, default), the step-4 export test downstream/rlinf/export_test.sh into ROOT/export_test/.
# Every launch logs batch hashes (trainer.log_batch_digest=true).  Writes only under ROOT.
# Usage: resume_test.sh ROOT        (KILL_AT=150 STOP=200 EVAL_SHARD=1/10 by default)
set -uo pipefail
ROOT="${1:?Usage: resume_test.sh ROOT}"
KILL_AT="${KILL_AT:-150}"; STOP="${STOP:-200}"; EVAL_SHARD="${EVAL_SHARD-1/10}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
log() { echo "[resume-test] $* $(date -u +%FT%TZ)" | tee -a "$ROOT/resume_test.log"; }

if [[ "${2:-}" == "--launch" ]]; then  # one training launch: resume_test.sh ROOT --launch EXP [overrides...]
  EXP="$3"; shift 3
  export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
  export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
  source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
  set +e
  export PYTHONPATH="$WT:$STARVLA_ROOT"
  cd "$STARVLA_ROOT"
  exec accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
    "$WT/downstream/train/train_entry.py" --config_yaml "$EXP/config.yaml" trainer.log_batch_digest=true "$@" \
    >> "$EXP/train.log" 2>&1
fi

last_step() { python3 -c '
import json, sys
s = 0
try:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        s = r.get("step", s)
except (OSError, ValueError):
    pass
print(s)' "$1"; }

gpus_free() { for i in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  (( n == 0 )) && return 0; sleep 5; done; return 1; }

A="$ROOT/RESUME-A"; B="$ROOT/RESUME-B"
nvidia-smi --query-gpu=index,name --format=csv,noheader
log "A: 0 -> end without a break"
t0=$(date +%s); bash "$0" "$ROOT" --launch "$A"; log "A rc=$? $(( $(date +%s) - t0 ))s"
gpus_free || log "GPUs still busy after A"

log "B: 0 -> killed at step >= $KILL_AT"
setsid bash "$0" "$ROOT" --launch "$B" &
BPID=$!
PGID=$(ps -o pgid= -p "$BPID" | tr -d ' ')
while kill -0 "$BPID" 2>/dev/null; do
  s=$(last_step "$B/run/metrics.jsonl")
  if (( s >= KILL_AT )); then
    kill -9 -- "-$PGID" 2>/dev/null
    log "B killed at logged step $s (process group $PGID)"
    break
  fi
  sleep 1
done
wait "$BPID" 2>/dev/null
gpus_free || log "GPUs still busy after the kill"
log "B left: $(ls "$B/run/full_state" 2>/dev/null | tr '\n' ' ')"

log "B: resume -> stop at $STOP"
t0=$(date +%s); bash "$0" "$ROOT" --launch "$B" "trainer.stop_at=$STOP"; log "B segment rc=$? $(( $(date +%s) - t0 ))s"
gpus_free || true
log "B: resume -> end"
t0=$(date +%s); bash "$0" "$ROOT" --launch "$B"; log "B rest rc=$? $(( $(date +%s) - t0 ))s"
gpus_free || true

python3 "$WT/downstream/train/resume_check.py" "$ROOT" > "$ROOT/resume_check.out" 2>&1
log "resume_check rc=$? (ROOT/resume_check.json)"

if [[ -n "$EVAL_SHARD" ]]; then
  CK="$B/run/checkpoints/steps_${STOP}_pytorch_model.pt"
  log "in-job evaluation check: $CK, 8 servers x 96 processes, shard $EVAL_SHARD"
  t0=$(date +%s)
  bash "$WT/downstream/eval/run_eval_ckpt.sh" "$CK" "$ROOT/eval_P96/out" 8 96 "$EVAL_SHARD" >> "$ROOT/resume_test.log" 2>&1
  log "evaluation rc=$? $(( $(date +%s) - t0 ))s"
  python3 "$WT/downstream/train/resume_check.py" "$ROOT" > "$ROOT/resume_check.out" 2>&1
fi
if [[ "${EXPORT_TEST:-1}" == 1 && ! -e "$ROOT/export_test/export" ]]; then
  log "step-4 export test (downstream/rlinf/export_test.sh)"
  bash "$WT/downstream/rlinf/export_test.sh" "$ROOT/export_test" >> "$ROOT/resume_test.log" 2>&1
  log "export test rc=$?"
fi
log "done"
