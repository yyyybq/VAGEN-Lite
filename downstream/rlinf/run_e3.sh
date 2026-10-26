#!/usr/bin/env bash
# Step 4, E3 (the user's decisions of 2026-10-23 evening and 2026-10-26): one PPO run of one start in RLinf, one
# 8-card job, three stages; a stage whose output exists is skipped, so a relaunch goes on where the last one stopped:
#   1. train: UPDATES (100) PPO updates on the tasks of RUN/cfg (made by scripts/prepare_rlinf_robocasa.py prepare from
#      the start's E1 final checkpoint, read by path; the 5 spatial tasks of the user's decision 1 of 2026-10-26),
#      32 environments, rollout and update epochs 2, actor learning rate 5e-6, gradient clip 1.0, RLinf's own
#      evaluation every 10 updates (8 environments), a checkpoint every 25 updates (about 63 GB each).  A relaunch
#      resumes from the newest complete checkpoint (runner.resume_dir).  Once the last update is saved, the older
#      checkpoints are deleted (the plan of 2026-10-23: only the newest and the last are kept).
#   2. export: the last actor to the StarVLA format with scripts/export_rlinf_starvla.py, RUN/export (about 16.8 GB;
#      the user's go-ahead of 2026-10-26); a half-written export (no rlinf_export.json) is removed and redone.
#   3. evaluate: the exported policy with the 944-episode closed-loop protocol, 8 policy servers x 96 simulator
#      processes, RUN/evals/final/out: the same evaluation as the start's E1 final checkpoint before RL.
# Writes only under RUN: e3.log, logs/, export/, evals/, e3_summary.json (the stages' return codes, at the end).
# Usage: run_e3.sh ENV RUN SOURCE_PT
#   (MAIN_CHECKOUT, STARVLA_SETUP, EXPORT_PY and E3_UPDATES only change for the tests)
set -uo pipefail
ENV="${1:?Usage: run_e3.sh ENV RUN SOURCE_PT}"
RUN="${2:?}"
SOURCE="${3:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN="${MAIN_CHECKOUT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
SETUP="${STARVLA_SETUP:-$WT/examples/train/robocasa/setup_starvla_env.sh}"
UPDATES="${E3_UPDATES:-100}"
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
train_rc=0 export_rc=0 eval_rc=0
log "start: run=$RUN source=$SOURCE updates=$UPDATES"

# 1. train
if complete "$FINAL"; then
  log "train: $FINAL exists, skipped"
else
  RESUME=()
  last=$(newest)
  [[ -n "$last" ]] && RESUME=("runner.resume_dir=$last")
  log "train: $UPDATES updates${last:+, resuming from $last}"
  RLINF_TAG="train_$(date -u +%Y%m%dT%H%M%SZ)" bash "$WT/downstream/rlinf/run_rlinf.sh" "$ENV" "$RUN" \
    "runner.max_epochs=$UPDATES" "runner.max_steps=$UPDATES" runner.save_interval=25 runner.val_check_interval=10 \
    env.train.total_num_envs=32 env.train.rollout_epoch=2 algorithm.update_epoch=2 actor.optim.lr=5e-6 \
    actor.optim.clip_grad=1.0 env.eval.total_num_envs=8 "runner.logger.log_path=$RUN/logs" "${RESUME[@]}" \
    >> "$RUN/e3.log" 2>&1
  train_rc=$?
  log "train rc=$train_rc"
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
