#!/usr/bin/env bash
# The 1-hour RL speed measurement of step 4 (E3; the user's decision 3 of 2026-10-23), one 8-card job.  RUN/cfg is made
# beforehand with scripts/prepare_rlinf_robocasa.py prepare (episode length 512, so a rollout divides into the global
# batch of 64).  Four launches of downstream/rlinf/run_rlinf.sh, one after the other, nothing saved, no evaluation:
#   envs8, envs16, envs32   2 PPO updates each with 8 / 16 / 32 environments (1 rollout epoch, 1 update epoch):
#                           environment steps per second, update time, GPU memory, CPU load
#   full                    1 update with the full configuration of the plan: 32 environments, 4 rollout epochs,
#                           4 update epochs, actor learning rate 5e-6, gradient clip 1.0
# Each launch logs RUN/rlinf_<tag>.log, gpu_<tag>.csv, cpu_<tag>.csv, hooks_<tag>.jsonl, grad_norms_<tag>.jsonl.
# Usage: speed_test.sh ENV RUN
set -uo pipefail
ENV="${1:?Usage: speed_test.sh ENV RUN}"
RUN="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
log() { echo "[speed-test] $* $(date -u +%FT%TZ)" | tee -a "$RUN/speed_test.log"; }
COMMON=(runner.save_interval=100000 runner.val_check_interval=100000 env.eval.total_num_envs=8)
run() {  # run TAG OVERRIDES...
  local tag="$1"; shift
  log "$tag: $*"
  local t0=$(date +%s)
  RLINF_TAG="$tag" bash "$WT/downstream/rlinf/run_rlinf.sh" "$ENV" "$RUN" "${COMMON[@]}" \
    "runner.logger.log_path=$RUN/logs_$tag" "$@" >> "$RUN/speed_test.log" 2>&1
  log "$tag rc=$? $(( $(date +%s) - t0 ))s"
}
for n in 8 16 32; do
  run "envs$n" "env.train.total_num_envs=$n" runner.max_epochs=2 runner.max_steps=2
done
run full env.train.total_num_envs=32 env.train.rollout_epoch=4 algorithm.update_epoch=4 actor.optim.lr=5e-6 \
  actor.optim.clip_grad=1.0 runner.max_epochs=1 runner.max_steps=1
log "done"
