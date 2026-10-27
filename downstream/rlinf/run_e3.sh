#!/usr/bin/env bash
# Step 4, E3 (the user's decisions of 2026-10-23 evening and 2026-10-26): one PPO run of one start in RLinf, one
# 8-card job, three stages; a stage whose output exists is skipped, so a relaunch goes on where the last one stopped:
#   1. train: UPDATES (100) PPO updates on the tasks of RUN/cfg (made by scripts/prepare_rlinf_robocasa.py prepare from
#      the start's source checkpoint, read by path; since 10-26 evening the E2 30% final model and 3 tasks),
#      32 environments, rollout and update epochs 2, actor learning rate 5e-6, gradient clip 1.0, a checkpoint every
#      20 updates (about 63 GB each).  RLinf's own evaluation during training is off (runner.val_check_interval -1,
#      the user's decision of 2026-10-27: on 10-27 both E3-B and E3-RS hung for good right after RLinf's first
#      in-training evaluation, one evaluation environment worker never returning); the 944-episode evaluation of
#      stage 3 is the measurement.  A watchdog ends the training when the RLinf log shows no new PPO update
#      ("Global Step: N") for STALL_MIN (40) minutes, counted from the launch for the first update: it writes
#      RUN/stalled_<TAG>.json, stops RLinf (TERM, then KILL a minute later) and the job ends with exit code 7, so a
#      hang no longer holds the cards.  (RLinf asserts that the checkpoint interval is a multiple of the evaluation
#      interval when the evaluation is on: the plan's 25 / 10 failed all four runs on 2026-10-26; checked here before
#      RLinf starts.)  A relaunch resumes from the newest complete checkpoint (runner.resume_dir).  Once the last update
#      is saved, the older checkpoints are deleted (the plan of 2026-10-23: only the newest and the last are kept).
#   2. export: the last actor to the StarVLA format with scripts/export_rlinf_starvla.py, RUN/export (about 16.8 GB;
#      the user's go-ahead of 2026-10-26); a half-written export (no rlinf_export.json) is removed and redone.
#   3. evaluate: the exported policy with the 944-episode closed-loop protocol, 8 policy servers x 96 simulator
#      processes, RUN/evals/final/out: the same evaluation as the start's E1 final checkpoint before RL.
# Writes only under RUN: e3.log, logs/, export/, evals/, e3_summary.json (the stages' return codes, at the end).
# Usage: run_e3.sh ENV RUN SOURCE_PT
#   (MAIN_CHECKOUT, STARVLA_SETUP, EXPORT_PY, E3_UPDATES, E3_SAVE_EVERY, E3_EVAL_EVERY, E3_STALL_MIN, E3_WATCH_EVERY
#    and E3_KILL_PATTERN only change for the tests)
set -uo pipefail
ENV="${1:?Usage: run_e3.sh ENV RUN SOURCE_PT}"
RUN="${2:?}"
SOURCE="${3:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="${MAIN_CHECKOUT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
SETUP="${STARVLA_SETUP:-$WT/examples/train/robocasa/setup_starvla_env.sh}"
UPDATES="${E3_UPDATES:-100}"
SAVE_EVERY="${E3_SAVE_EVERY:-20}"
EVAL_EVERY="${E3_EVAL_EVERY:--1}"  # -1: RLinf's in-training evaluation off (10-27)
STALL_MIN="${E3_STALL_MIN:-40}"
STALL_SEC="${E3_STALL_SEC:-$(( STALL_MIN * 60 ))}"
WATCH_EVERY="${E3_WATCH_EVERY:-60}"
KILL_PATTERN="${E3_KILL_PATTERN:-train_embodied_agent.py}"
if (( (EVAL_EVERY > 0 && SAVE_EVERY % EVAL_EVERY != 0) || UPDATES % SAVE_EVERY != 0 )); then  # RLinf's check_progress
  echo "[e3] checkpoint interval $SAVE_EVERY must be a multiple of the evaluation interval $EVAL_EVERY and divide" \
       "the $UPDATES updates: stop" | tee -a "$RUN/e3.log"
  exit 6
fi
CKPTS="$RUN/logs/robocasa365_starvla_ppo/checkpoints"
FINAL="$CKPTS/global_step_$UPDATES"
POLICY="$RUN/export/checkpoints/policy.pt"
EVAL="$RUN/evals/final/out"
log() { echo "[e3] $* $(date -u +%FT%TZ)" | tee -a "$RUN/e3.log"; }
complete() { [[ -f "$1/actor/dcp_checkpoint/.metadata" && -f "$1/actor/model_state_dict/full_weights.pt" ]]; }
newest() {  # the newest complete checkpoint directory, or nothing
  local best="" n=-1 d s
  for d in "$CKPTS"/global_step_*; do
    [[ -d "$d" ]] || continue
    s="${d##*global_step_}"
    if [[ "$s" =~ ^[0-9]+$ ]] && complete "$d" && (( s > n )); then n=$s; best="$d"; fi
  done
  echo "$best"
}
last_update() {  # the last "Global Step: N" of an RLinf log, or nothing
  tr '\r' '\n' < "$1" 2>/dev/null | grep -ao 'Global Step: *[0-9]*' | tail -1 | grep -o '[0-9]*$'
}
watchdog() {  # watchdog PID LOG TAG: end the training when no new PPO update appears for STALL_MIN minutes
  local pid=$1 rlog=$2 tag=$3 seen=-1 since now step
  since=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$WATCH_EVERY"
    step=$(last_update "$rlog")
    now=$(date +%s)
    if [[ -n "$step" ]] && (( step > seen )); then seen=$step; since=$now; fi
    if (( now - since > STALL_SEC )); then
      log "watchdog: no new PPO update for $STALL_MIN minutes (last update ${step:-none}): ending the training"
      printf '{"utc": "%s", "tag": "%s", "last_update": "%s", "stall_minutes": %s}\n' "$(date -u +%FT%TZ)" "$tag" \
        "${step:-none}" "$STALL_MIN" > "$RUN/stalled_$tag.json"
      pkill -TERM -f "$KILL_PATTERN" 2>/dev/null; pkill -TERM -P "$pid" 2>/dev/null
      sleep "$WATCH_EVERY"
      pkill -KILL -f "$KILL_PATTERN" 2>/dev/null; pkill -KILL -P "$pid" 2>/dev/null
      return 0
    fi
  done
}
train_rc=0 export_rc=0 eval_rc=0
log "start: run=$RUN source=$SOURCE updates=$UPDATES"

# 1. train
if complete "$FINAL"; then
  log "train: $FINAL exists, skipped"
else
  RESUME=()
  last=$(newest)
  [[ -n "$last" ]] && RESUME=("runner.resume_dir=$last")
  TAG="train_$(date -u +%Y%m%dT%H%M%SZ)"
  log "train: $UPDATES updates${last:+, resuming from $last}; in-training evaluation every $EVAL_EVERY (-1: off);" \
      "watchdog after $STALL_MIN minutes without a new update"
  RLINF_TAG="$TAG" bash "$WT/downstream/rlinf/run_rlinf.sh" "$ENV" "$RUN" \
    "runner.max_epochs=$UPDATES" "runner.max_steps=$UPDATES" "runner.save_interval=$SAVE_EVERY" \
    "runner.val_check_interval=$EVAL_EVERY" \
    env.train.total_num_envs=32 env.train.rollout_epoch=2 algorithm.update_epoch=2 actor.optim.lr=5e-6 \
    actor.optim.clip_grad=1.0 env.eval.total_num_envs=8 "runner.logger.log_path=$RUN/logs" "${RESUME[@]}" \
    >> "$RUN/e3.log" 2>&1 &
  rlinf_pid=$!
  watchdog "$rlinf_pid" "$RUN/rlinf_$TAG.log" "$TAG" > /dev/null 2>&1 &  # log() still writes e3.log
  watch_pid=$!
  wait "$rlinf_pid"
  train_rc=$?
  kill "$watch_pid" 2>/dev/null
  wait "$watch_pid" 2>/dev/null
  log "train rc=$train_rc"
  if [[ -f "$RUN/stalled_$TAG.json" ]]; then
    log "train: ended by the watchdog (no new update for $STALL_MIN minutes): stop"
    exit 7
  fi
  if ! complete "$FINAL"; then
    log "train: $FINAL is missing: stop"
    exit 4
  fi
fi
for d in "$CKPTS"/global_step_*; do
  if [[ -d "$d" && "$d" != "$FINAL" ]]; then rm -rf "$d"; log "deleted the older checkpoint $d"; fi
done

# 2. export
if [[ -f "$POLICY" && -f "$RUN/export/rlinf_export.json" ]]; then
  log "export: $POLICY exists, skipped"
else
  [[ -e "$RUN/export" ]] && { rm -rf "$RUN/export"; log "export: removed a half-written export"; }
  log "export: $FINAL -> $RUN/export"
  (
    source "$SETUP" > /dev/null
    set +e
    cd "$MAIN"
    "${EXPORT_PY:-python}" scripts/export_rlinf_starvla.py --weights "$FINAL/actor/model_state_dict/full_weights.pt" \
      --source-ckpt "$SOURCE" --out "$RUN/export"
  ) >> "$RUN/e3.log" 2>&1
  export_rc=$?
  log "export rc=$export_rc $(du -sh "$RUN/export" 2>/dev/null | cut -f1)"
  if [[ $export_rc -ne 0 || ! -f "$POLICY" ]]; then
    log "export failed: stop"
    exit 5
  fi
fi

# 3. evaluate (944 episodes; run_eval_ckpt.sh resumes an evaluation that stopped half way)
if [[ -f "$EVAL/episodes/summary.json" ]] && python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s.get("complete") == s.get("episodes") and "per_task" in s else 1)' "$EVAL/episodes/summary.json"; then
  log "evaluate: complete, skipped"
else
  log "evaluate: $POLICY, 8 servers x 96 processes"
  bash "$WT/downstream/eval/run_eval_ckpt.sh" "$POLICY" "$EVAL" 8 96 >> "$RUN/e3.log" 2>&1
  eval_rc=$?
  log "evaluate rc=$eval_rc"
fi
python3 - "$RUN" "$train_rc" "$export_rc" "$eval_rc" "$FINAL" <<'EOF'
import json, sys
from pathlib import Path
run, final = Path(sys.argv[1]), sys.argv[5]
summary = run / "evals/final/out/episodes/summary.json"
res = {"train_rc": int(sys.argv[2]), "export_rc": int(sys.argv[3]), "eval_rc": int(sys.argv[4]), "final": final,
       "eval": ({k: v for k, v in json.loads(summary.read_text()).items() if k != "per_task"}
                if summary.is_file() else None)}
(run / "e3_summary.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res))
EOF
log "done"
exit $eval_rc
