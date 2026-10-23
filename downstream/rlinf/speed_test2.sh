#!/usr/bin/env bash
# The second RL speed measurement (the user's decision 5 of 2026-10-23 evening: rollout and update epochs 2 each,
# 64 environments tried, 100 updates, at most 30 hours per start), one 8-card job.  RUN/cfg as for speed_test.sh,
# copied from RL-SPEED-1 (the 4 tasks of the stage-0 report, episode length 512).  Three launches of
# downstream/rlinf/run_rlinf.sh, one after the other, nothing saved:
#   envs64   2 PPO updates with 64 environments (1 rollout epoch, 1 update epoch): environment steps per second
#   full64   1 update of the new configuration: 64 environments, 2 rollout epochs, 2 update epochs, actor learning
#            rate 5e-6, gradient clip 1.0
#   full32   the same with 32 environments (if 64 do not pay)
# Each launch logs RUN/rlinf_<tag>.log, gpu_<tag>.csv, cpu_<tag>.csv, hooks_<tag>.jsonl, grad_norms_<tag>.jsonl.
# Usage: speed_test2.sh ENV RUN
set -uo pipefail
ENV="${1:?Usage: speed_test2.sh ENV RUN}"
RUN="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log() { echo "[speed-test-2] $* $(date -u +%FT%TZ)" | tee -a "$RUN/speed_test.log"; }
COMMON=(runner.save_interval=100000 runner.val_check_interval=100000 env.eval.total_num_envs=8)
FULL=(env.train.rollout_epoch=2 algorithm.update_epoch=2 actor.optim.lr=5e-6 actor.optim.clip_grad=1.0
      runner.max_epochs=1 runner.max_steps=1)
run() {  # run TAG OVERRIDES...
  local tag="$1"; shift
  log "$tag: $*"
  local t0=$(date +%s)
  RLINF_TAG="$tag" bash "$WT/downstream/rlinf/run_rlinf.sh" "$ENV" "$RUN" "${COMMON[@]}" \
    "runner.logger.log_path=$RUN/logs_$tag" "$@" >> "$RUN/speed_test.log" 2>&1
  log "$tag rc=$? $(( $(date +%s) - t0 ))s"
}
run envs64 env.train.total_num_envs=64 runner.max_epochs=2 runner.max_steps=2
run full64 env.train.total_num_envs=64 "${FULL[@]}"
run full32 env.train.total_num_envs=32 "${FULL[@]}"
log "done"
