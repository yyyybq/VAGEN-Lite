#!/usr/bin/env bash
# Resume test, second run (2026-10-22).  In the first run (resume_test.sh, job pt-i4bzrj4k) the kill reached only the
# launcher: torch's elastic launcher starts the training workers in sessions of their own, so SIGKILL to the launch's
# process group left them training to the end, and the kill -> resume path was not exercised.  Here the kill also hits
# every process whose command line names B's config (the elastic agent, the 8 workers, their data-loader workers) and
# checks that none is left before resuming.  A (ROOT/RESUME-A, the uninterrupted run of the first test) is reused.
#   B: ROOT/<B_NAME>, made like RESUME-A by make_experiment.py; trained from 0, killed once step KILL_AT is logged
#      (its full state at 100 is complete by then), resumed with trainer.stop_at=STOP (retrains from the state at 100,
#      leaves after STOP's checkpoint and full state), resumed again to the last step;
#   resume_check.py ROOT RESUME-A <B_NAME> -> ROOT/resume_check_<B_NAME>.json
# One 8-card sco job; writes only under ROOT/<B_NAME> and ROOT/resume_kill_test_<B_NAME>.log.
# Usage: resume_kill_test.sh ROOT [B_NAME]      (B_NAME RESUME-B2, KILL_AT=150, STOP=200 by default)
set -uo pipefail
ROOT="${1:?Usage: resume_kill_test.sh ROOT [B_NAME]}"
BN="${2:-RESUME-B2}"
B="$ROOT/$BN"
KILL_AT="${KILL_AT:-150}"; STOP="${STOP:-200}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
log() { echo "[resume-kill-test] $* $(date -u +%FT%TZ)" | tee -a "$ROOT/resume_kill_test_$BN.log"; }

if [[ "${3:-}" == "--launch" ]]; then  # one training launch of B: resume_kill_test.sh ROOT B_NAME --launch [overrides]
  shift 3
  export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=disabled WANDB_DISABLED=true NO_PROXY='*' no_proxy='*' STARVLA_ROOT
  export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
  source "$WT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
  set +e
  export PYTHONPATH="$WT:$STARVLA_ROOT"
  cd "$STARVLA_ROOT"
  exec accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
    "$WT/downstream/train/train_entry.py" --config_yaml "$B/config.yaml" trainer.log_batch_digest=true "$@" \
    >> "$B/train.log" 2>&1
fi

PATTERN="--config_yaml $B/config.yaml"
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
alive() { pgrep -f -- "$PATTERN" | grep -c .; }
gpus_free() { for i in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  (( n == 0 )) && return 0; sleep 5; done; return 1; }

nvidia-smi --query-gpu=index,name --format=csv,noheader
[[ -d "$ROOT/RESUME-A/run" ]] || { log "no uninterrupted run $ROOT/RESUME-A/run: stop"; exit 3; }
[[ -e "$B/run" ]] && { log "$B/run exists already: stop"; exit 3; }

log "B: 0 -> killed at step >= $KILL_AT"
setsid bash "$0" "$ROOT" "$BN" --launch &
BPID=$!
while kill -0 "$BPID" 2>/dev/null; do
  s=$(last_step "$B/run/metrics.jsonl")
  if (( s >= KILL_AT )); then
    PGID=$(ps -o pgid= -p "$BPID" 2>/dev/null | tr -d ' ')
    [[ -n "$PGID" ]] && kill -9 -- "-$PGID" 2>/dev/null
    pkill -9 -f -- "$PATTERN"
    for i in $(seq 1 60); do (( $(alive) == 0 )) && break; pkill -9 -f -- "$PATTERN"; sleep 2; done
    log "B killed at logged step $s; processes of B's launch left: $(alive)"
    break
  fi
  sleep 1
done
wait "$BPID" 2>/dev/null
gpus_free || log "GPUs still busy after the kill"
log "B left: full states $(ls "$B/run/full_state" 2>/dev/null | tr '\n' ' '); last logged step $(last_step "$B/run/metrics.jsonl")"

log "B: resume -> stop at $STOP"
t0=$(date +%s); bash "$0" "$ROOT" "$BN" --launch "trainer.stop_at=$STOP"; log "B segment rc=$? $(( $(date +%s) - t0 ))s"
gpus_free || true
log "B: resume -> end"
t0=$(date +%s); bash "$0" "$ROOT" "$BN" --launch; log "B rest rc=$? $(( $(date +%s) - t0 ))s"
gpus_free || true

python3 "$WT/downstream/train/resume_check.py" "$ROOT" RESUME-A "$BN" > "$ROOT/resume_check_$BN.out" 2>&1
log "resume_check rc=$? ($ROOT/resume_check_$BN.json)"
log "done"
