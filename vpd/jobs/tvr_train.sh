#!/usr/bin/env bash
# sco payload of jobs vpd-tvr-train-q3b / -q3s / -e2 (CONTRACT_wave2 7.4; DECISIONS G10(b)(c)(i)): the pilot adapters
# of one start, after the prereg is frozen and the smoke passed; with VPD_TVR_ROOT=explore/rl (below) the EXPLORATORY
# jobs vpd-tvr-xtrain-q3rd / -q3rs150 (DECISIONS G18).  as_root, one H800, gate tvr, from pilot_tvr.py:
#     env VPD_JOB_DEADLINE_S=<soft> timeout -k 120 <outer> bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/tvr_train.sh START RUN
# (outer / soft from the est_high DERIVED from the calibration, 5.3.1; e.g. N_top ALL 3.0 -> 10500 / 10320.)
# P = OUT/tvr/runs/pilot/START (made by pilot_tvr prepare as the login user).  Steps:
#   node, prereg_committed (vpd_snapshot_file vpd/PREREG_TVR_PILOT.md), guard (pilot_tvr guard tvr-train-<s>: prereg
#   block == pilot_cfg, smoke PASS, futility false or override, missing() == [], data shas, trainer prefix)
#   train_0W     tiers/v1/warmup0w20.jsonl, --phase 0W, seed 1001               -> P/adapters/0W
#   train_100    demos_100.jsonl, --init-adapter P/adapters/0W, --phase tier100 -> P/adapters/100
#   train_<TOP>  demos_<TOP>.jsonl (TOP = pilot_cfg tiers.n_top: ALL or 400), --init-adapter 0W, --phase tier<TOP>
#   Q3B only (the seed replicate, O4 / O13): train_0W_s1002 (--seed 1002 --phase 0W_s1002 -> P/adapters/0W_s1002),
#                train_100_s1002 (--init-adapter P/adapters/0W_s1002, --seed 1002, --phase tier100_s1002)
#   expect       pilot_tvr expect tvr-train-<s> -> RUN/expect.json; RUN/DONE only if every step passed
# Same flags as the calibration (--epochs auto --threads 2 --num-workers 3; LoRA r16/a32 defaults of the frozen
# trainer).  Idempotent via `pilot adapter-ok` (skip complete adapters; exit 3 = trained with other code: the job
# stops, nothing overwritten, exit 3).  As root, RUN and P/adapters are chowned to 20325 at EXIT (never P itself).
# Deadlines: soft VPD_JOB_DEADLINE_S (default 10320); every training under `timeout -k 30 <left>`.
# Env: VPD_DRYRUN=1 prints the commands; test hooks honoured ONLY with VPD_DRYRUN=1: VPD_TVR_OUT, VPD_DRY_NTOP (ALL).
# VPD_TVR_ROOT (DECISIONS G18): the output root under OUT/tvr/runs.  Unset or "pilot": the pilot jobs as above
# (START Q3B Q3S E2, job tvr-train-<s>, P = runs/pilot/START).  "explore/rl": the EXPLORATORY RL starts (START Q3RD
# Q3RS150, job tvr-xtrain-<s>, P = runs/explore/rl/START; the same steps, data, seed and flags; pilot_tvr's explore
# payload sets it).  Any other value: exit 2.  The third explore start Q3SG (PREREG section 17, row 2026-11-13;
# DECISIONS G20) runs under explore/rl the same way (job tvr-xtrain-q3sg, P = runs/explore/rl/Q3SG).

START=${1:-}
RUN=${2:-}
TVR_ROOT=${VPD_TVR_ROOT:-pilot}
case $TVR_ROOT in
  pilot)
    case $START in
      Q3B|Q3S|E2) ;;
      *) echo "usage: tvr_train.sh START RUN   (START: Q3B Q3S E2)" >&2; exit 2 ;;
    esac
    JOB=tvr-train-${START,,} ;;
  explore/rl)
    case $START in
      Q3RD|Q3RS150|Q3SG) ;;
      *) echo "usage: VPD_TVR_ROOT=explore/rl tvr_train.sh START RUN   (START: Q3RD Q3RS150 Q3SG)" >&2; exit 2 ;;
    esac
    JOB=tvr-xtrain-${START,,} ;;
  *) echo "VPD_TVR_ROOT must be pilot or explore/rl, got '$TVR_ROOT'" >&2; exit 2 ;;
esac
if [[ -z $RUN || $# -ne 2 ]]; then
  echo "usage: tvr_train.sh START RUN" >&2
  exit 2
fi
TAG=tvr_train
NEED_PREREG=1
DEF_DEADLINE=10320
export RUN
export VPD_TVR_ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"  # before common_env.sh (lvp profile "envlib+orig")
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
unset ACTIVESPATIAL_USE_X11 ACTIVESPATIAL_X11_FORCE_VULKAN DISPLAY LP_NUM_THREADS
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

DRY=${VPD_DRYRUN:-0}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-$DEF_DEADLINE}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
THOR_RESERVE_S=240
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_MIN_STEP_S must be a positive integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))
TVR=$VPD_OUT/tvr
if [[ $DRY == 1 && -n ${VPD_TVR_OUT:-} ]]; then
  TVR=$VPD_TVR_OUT  # test hook (dry run only)
fi
TVR_PY=${VPD_TVR_PY:-$VPD_OUT/envs/tvr/bin/python}
PUBLIC=$TVR/hf_meta/tvr-sft-va/sft_train.json
REPLAY=$TVR/replay/sft1600_lvp
DEMOS=$TVR/demos/sft1600_lvp
TIERS=$TVR/tiers/v1
CAL=$TVR/runs/explore/cal/Q3B
FALLBACK_FILE=$CAL/shake/fallback_used.json
FAILS=()
BUILD=?
FB=?
mkdir -p "$RUN/logs"
cd "$VPD_WORKTREE"
[[ $DRY == 1 ]] || rm -f -- "$RUN/DONE"  # DONE describes this run only
P=$TVR/runs/$TVR_ROOT/$START
AD=$P/adapters
PARITY_DIR=""
OUTROOT=""
PREFIX=pilot
TASKSET=eval500
MERGED_TIERS=""
REPLICATE_START=Q3B   # pilot_cfg tiers.replicate (start Q3B, tier 100, seed 1002)
REPLICATE_SEED=1002
if [[ $DRY != 1 ]]; then
  mkdir -p "$AD"
  vpd_chown_trap "$RUN" "$AD"   # never P or runs/pilot (shared parents, made by pilot_tvr prepare)
fi

# ---- helpers (identical in tvr_cal.sh, tvr_cal400.sh, tvr_mech_smoke.sh, tvr_train.sh, tvr_eval.sh) ----
x() { if [[ $DRY == 1 ]]; then echo "[dry] $*"; else "$@"; fi; }
left() { echo $((DEADLINE - $(date +%s))); }
# part FUNC ARGS...: one block; a failure is recorded (vpd_step line rc != 0 and a FAILS entry) and never ends the
# payload under set -e (an || context), as wb_variants.sh
part() {
  local n=${#FAILS[@]} rc=0
  "$@" || rc=$?
  if ((rc != 0 && ${#FAILS[@]} == n)); then
    vpd_step "$1:failed" false || true
    FAILS+=("$1")
  fi
  return 0
}
# timed NAME CMD...: vpd_step NAME under `timeout -k 30 <seconds left>`; NAME:no_time when < MIN_STEP_S are left
timed() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t < MIN_STEP_S)); then
    echo "[$TAG] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@"
}
# timed_thor NAME CMD...: as timed, plus --deadline-s <seconds left - 240> (run_tvr's own fast shutdown, 5.2.2)
timed_thor() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t - THOR_RESERVE_S < MIN_STEP_S)); then
    echo "[$TAG] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@" --deadline-s $((t - THOR_RESERVE_S))
}
step() {  # NAME plain|thor CMD...: timed / timed_thor; a failure is a FAILS entry named NAME
  local name=$1 kind=$2; shift 2
  if [[ $kind == thor ]]; then
    timed_thor "$name" "$@" || { FAILS+=("$name"); return 1; }
  else
    timed "$name" "$@" || { FAILS+=("$name"); return 1; }
  fi
}
cfg() {  # KEY...: pilot_cfg values (dry run: <KEY> placeholders)
  if [[ $DRY == 1 ]]; then
    local k out=()
    for k in "$@"; do out+=("<$k>"); done
    echo "${out[*]}"
    return 0
  fi
  "$PY_EVAL" -m vpd.tvr.pilot_tvr cfg "$@"
}
# reap_unity WHY: kill lingering Unity players of the frozen build (TERM, 5 s, KILL); RUN/logs/reap.log
reap_unity() {
  if [[ $DRY == 1 ]]; then echo "[dry] reap_unity $1"; return 0; fi
  local pids k pat="/.ai2thor/releases/$BUILD/$BUILD"
  if ! command -v pgrep > /dev/null || ! command -v pkill > /dev/null; then
    echo "[$TAG] $1: pgrep/pkill missing, cannot reap Unity players" | tee -a "$RUN/logs/reap.log"
    return 0
  fi
  pids=$(pgrep -f -- "$pat" | tr '\n' ' ' || true)
  [[ -n ${pids// /} ]] || return 0
  echo "[$TAG] $(date -u +%TZ) $1: TERM lingering Unity player(s): $pids" | tee -a "$RUN/logs/reap.log"
  pkill -TERM -f -- "$pat" || true
  for ((k = 0; k < 50; k++)); do
    pgrep -f -- "$pat" > /dev/null || return 0
    sleep 0.1
  done
  echo "[$TAG] $(date -u +%TZ) $1: KILL Unity player(s) still alive" | tee -a "$RUN/logs/reap.log"
  pkill -KILL -f -- "$pat" || true
  return 0
}
reap() { vpd_step "reap_$1" reap_unity "$1"; }
pick_merge_dir() {  # /tmp if >= 40 GB free, else /dev/shm (DECISIONS B11); empty if neither
  local base avail
  for base in /tmp /dev/shm; do
    avail=$(df -P -B1G "$base" 2>/dev/null | awk 'NR==2{print $4}')
    if [[ -n $avail && $avail -ge 40 ]]; then
      echo "$base/vpd_merged"
      return 0
    fi
  done
  echo ""
}
safe_rm_merged() {  # only job-local merged folders
  case $1 in
    /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) x rm -rf -- "$1" ;;
    *) echo "[$TAG] refusing to remove $1" >&2; return 1 ;;
  esac
}
stop_job() {  # RC: end the payload now (the EXIT handlers still remove merged weights, reap and chown); no DONE
  df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
  echo "[$TAG] stopped $(date -u +%FT%TZ) rc=$1; failed steps: ${FAILS[*]:-none}"
  exit "$1"
}

GUARD_OK=0
# guard: (msmoke / train / eval: NEED_PREREG=1) prereg_committed = the snapshot's PREREG_TVR_PILOT.md is the
# committed file (vpd_snapshot_file), then `pilot_tvr guard JOB` (missing(), check_prereg, smoke PASS, futility,
# data shas, trainer prefix, adapters); a failure skips every model step (expect still runs)
guard() {
  if ((NEED_PREREG)); then
    vpd_step prereg_committed x vpd_snapshot_file vpd/PREREG_TVR_PILOT.md || { FAILS+=(prereg_committed); return 1; }
  fi
  vpd_step guard x "$PY_EVAL" -m vpd.tvr.pilot_tvr guard "$JOB" --run "$RUN" || { FAILS+=(guard); return 1; }
  GUARD_OK=1
}
merge_disk() { vpd_step merge_disk test -n "$VPD_MERGE_DIR" || { FAILS+=(merge_disk); return 1; }; }

# adapter_state DIR [INIT_DIR] -> 0 complete with the current code (and from INIT_DIR/adapter.pt as it is now),
# 1 missing / unfinished / other init (retrain), 3 trained with other code (stop)
adapter_state() {
  local args=("$1")
  [[ -n ${2:-} ]] && args+=(--init "$2")
  [[ $DRY == 1 ]] && { echo "[dry] $PY_EVAL -m vpd.walkerbench.pilot adapter-ok ${args[*]}"; return 1; }
  local rc=0
  "$PY_EVAL" -m vpd.walkerbench.pilot adapter-ok "${args[@]}" || rc=$?
  return "$rc"
}
declare -A TRAINED=()
# train LABEL DATA PHASE SEED [INIT_LABEL]: AD/LABEL from AD/INIT_LABEL (skip rule: adapter-ok)
train() {
  local label=$1 data=$2 phase=$3 seed=$4 init=${5:-} st=0
  local out=$AD/$label initdir=""
  [[ -n $init ]] && initdir=$AD/$init
  if [[ -n $init && -z ${TRAINED[$init]:-} ]]; then
    vpd_step "train_$label:no_init" false || true
    FAILS+=("train_$label")
    return 1
  fi
  adapter_state "$out" "$initdir" || st=$?
  if ((st == 0)); then
    vpd_step "train_$label:skip" true
    TRAINED[$label]=1
    return 0
  fi
  if ((st == 3)); then
    echo "[$TAG] $out was trained with other code: stopping (nothing overwritten)" >&2
    vpd_step "train_$label:code_mismatch" false || true
    FAILS+=("train_$label")
    stop_job 3
  fi
  local args=(--start "$START" --data "$data" --out "$out" --epochs auto --seed "$seed" --phase "$phase"
              --threads "$VPD_THREADS" --num-workers 3)
  [[ -n $initdir ]] && args+=(--init-adapter "$initdir")
  if timed "train_$label" "$PY_TRAIN" vpd/train/n2_train_vpd.py "${args[@]}"; then
    TRAINED[$label]=1
  else
    FAILS+=("train_$label")
    return 1
  fi
}
# need_adapter LABEL [INIT_LABEL]: an adapter another job trained must be complete (adapter-ok 0) to be merged / used
need_adapter() {
  local st=0 initdir=""
  [[ -n ${2:-} ]] && initdir=$AD/$2
  adapter_state "$AD/$1" "$initdir" || st=$?
  if ((st == 0)) || [[ $DRY == 1 ]]; then
    vpd_step "adapter_$1:ok" true
    TRAINED[$1]=1
    return 0
  fi
  vpd_step "adapter_$1:not_ready" false || true
  FAILS+=("adapter_$1")
  ((st == 3)) && stop_job 3
  return 1
}

# eval_state DIR ADAPTER_DIR -> 0 fresh results.json (it served ADAPTER_DIR/adapter.pt as it is now), 1 none, 3 stale
# or unreadable (EVALOK_RC = eval-ok's own exit code: 3 stale; anything else = unreadable results.json / a crash)
EVALOK_RC=0
eval_state() {
  local rc=0
  EVALOK_RC=0
  [[ -f $1/results.json ]] || return 1
  "$PY_EVAL" -m vpd.walkerbench.pilot eval-ok "$1" --adapter "$2" || rc=$?
  EVALOK_RC=$rc
  ((rc == 0 || rc == 3)) || rc=3
  return "$rc"
}
# parity_state FILE ADAPTER_DIR -> 0 current + pass, 4 current + failed (not rerun), 1 missing / other adapter sha
parity_state() {
  [[ $DRY == 1 ]] && { echo "[dry] $PY_EVAL -m vpd.tvr.pilot_tvr parity-ok $1 --adapter $2"; return 1; }
  local rc=0
  "$PY_EVAL" -m vpd.tvr.pilot_tvr parity-ok "$1" --adapter "$2" || rc=$?
  return "$rc"
}

MML=()
CANARY_OK=1
# session TIER ADAPTER_LABEL PARITY(0|1) ARMS [RUN_TVR ARGS...]: merge AD/ADAPTER_LABEL on job-local disk, merge parity
# (PARITY=1; PY_TRAIN, before vLLM), ONE run_tvr session for ARMS on the merged folder (--resume; the full arm list
# always), remove the folder.  Fresh evals (and a current merge parity) are skipped; a stale eval is kept and flagged
# unless VPD_RERUN_STALE=1 (then moved aside to DIR.stale-<UTC>).  No eval after a failed canary.
session() {
  local tier=$1 label=$2 want_parity=$3 arms=$4; shift 4
  local addir=$AD/$label arm st todo=0 stale=0 dir pst=0 pfile=$PARITY_DIR/merge_$tier.json
  if ((CANARY_OK != 1)); then
    vpd_step "eval_$tier:canary_failed" false || true
    FAILS+=("eval_$tier")
    return 1
  fi
  for arm in ${arms//,/ }; do
    dir=$OUTROOT/${PREFIX}__${START}__${tier}__$arm
    st=0
    eval_state "$dir" "$addir" || st=$?
    if ((st == 3)); then
      if ((EVALOK_RC != 3)); then
        echo "[$TAG] eval-ok on $dir exited $EVALOK_RC (unreadable results.json or a broken interpreter): kept" >&2
        vpd_step "eval_${tier}_$arm:evalok_rc$EVALOK_RC" false || true
        stale=1
      elif [[ ${VPD_RERUN_STALE:-0} == 1 ]]; then
        vpd_step "stale_aside_${tier}_$arm" x mv -T -- "$dir" "$dir.stale-$(date -u +%Y%m%dT%H%M%SZ)" || stale=1
        todo=1
      else
        echo "[$TAG] $dir served other weights than $addir/adapter.pt; VPD_RERUN_STALE=1 re-runs it" >&2
        stale=1
      fi
    elif ((st != 0)); then
      todo=1
    fi
  done
  if ((stale)); then
    vpd_step "eval_$tier:stale_kept" false || true
    FAILS+=("eval_$tier")
    return 1
  fi
  if ((want_parity)); then
    pst=0
    parity_state "$pfile" "$addir" || pst=$?
  fi
  if ((todo == 0 && pst != 1)); then
    vpd_step "merge_$tier:skip" true
    if ((want_parity && pst == 4)); then
      vpd_step "merge_parity_$tier:failed_kept" false || true
      FAILS+=("merge_parity_$tier")
    elif ((want_parity)); then
      vpd_step "merge_parity_$tier:skip" true
    fi
    vpd_step "eval_$tier:skip" true
    return 0
  fi
  if [[ -z ${TRAINED[$label]:-} ]]; then
    vpd_step "merge_$tier:no_adapter" false || true
    FAILS+=("merge_$tier")
    return 1
  fi
  if [[ -z $VPD_MERGE_DIR ]]; then
    vpd_step "merge_$tier:no_disk" false || true
    FAILS+=("merge_$tier")
    return 1
  fi
  local local_dir=$VPD_MERGE_DIR/${PREFIX}__${START}__$tier
  safe_rm_merged "$local_dir" || true
  if ! timed "merge_$tier" "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$addir" --out "$local_dir"; then
    FAILS+=("merge_$tier")
    vpd_step "rm_merged_$tier" safe_rm_merged "$local_dir" || true
    return 1
  fi
  if ((want_parity && pst == 1)); then
    timed "merge_parity_$tier" "$PY_TRAIN" -B -m vpd.tvr.merge_parity --start "$START" --adapter "$addir" \
      --merged "$local_dir" --demos "$TIERS/warmup0w20.jsonl" --n 20 --out "$pfile" || FAILS+=("merge_parity_$tier")
  elif ((want_parity && pst == 4)); then
    vpd_step "merge_parity_$tier:failed_kept" false || true
    FAILS+=("merge_parity_$tier")
  elif ((want_parity)); then
    vpd_step "merge_parity_$tier:skip" true
  fi
  if ((todo)); then
    timed_thor "eval_$tier" "$PY_Q3" -B -m vpd.tvr.run_tvr eval --start "$START" --tier "$tier" --arms "$arms" \
      --task-set "$TASKSET" --name-prefix "$PREFIX" --out-root "$OUTROOT" --model-dir "$local_dir" \
      --adapter-meta "$addir/meta.json" "$@" ${MML[@]+"${MML[@]}"} --resume || FAILS+=("eval_$tier")
    reap "eval_$tier"
  else
    vpd_step "eval_$tier:skip" true
  fi
  vpd_step "rm_merged_$tier" safe_rm_merged "$local_dir" || FAILS+=("rm_merged_$tier")
}

# canary DIR: run_tvr replay sft2 into DIR (as this job's user and node), then the frozen tolerance check against the
# data job's replay (TVR_PY render_parity, 5.2.6); CANARY_OK=0 on any failure (the job's evals are then not run).
# The canary is always rendered by THIS run: an earlier run's DIR is moved aside to DIR.prev-<UTC> first (a resubmitted
# job may land on another node, and --resume would skip a complete replay and recheck the old frames, O22).  Only when
# this run has the time to render it (timed_thor's test): otherwise DIR stays (canary_replay:no_time, no eval runs).
canary() {
  local d=$1
  if [[ $DRY != 1 && -d $d && -n $(ls -A -- "$d" 2> /dev/null) ]] && (($(left) - THOR_RESERVE_S >= MIN_STEP_S)); then
    if ! vpd_step canary_aside mv -T -- "$d" "$d.prev-$(date -u +%Y%m%dT%H%M%SZ)"; then
      FAILS+=(canary_aside)
      CANARY_OK=0
      return 1
    fi
    mkdir -p -- "$d"
  fi
  timed_thor canary_replay "$PY_Q3" -B -m vpd.tvr.run_tvr replay --task-set sft2 --public "$PUBLIC" --run-dir "$d" \
    --images-dir "$d/images" --resume || { FAILS+=(canary_replay); CANARY_OK=0; }
  reap canary_replay
  if ((CANARY_OK == 1)); then
    timed canary_parity "$TVR_PY" -B -m vpd.tvr.render_parity --replay "$d" --ref "$REPLAY" --ref-images \
      "$DEMOS/images" --images "$d/images" --task-set sft2 --out "$d/render_parity.json" \
      || { FAILS+=(canary_parity); CANARY_OK=0; }
  else
    vpd_step canary_parity:no_replay false || true
  fi
  ((CANARY_OK == 1))
}

EXPECT_OK=0
expect_list() {
  if vpd_step expect x "$PY_EVAL" -m vpd.tvr.pilot_tvr expect "$JOB" --run "$RUN"; then
    EXPECT_OK=1
  else
    FAILS+=(expect)
    return 1
  fi
}
finish() {  # df_end, RUN/DONE only when every step (expect included) passed; exit 0 / 1
  df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
  if ((${#FAILS[@]} == 0 && EXPECT_OK == 1)); then
    if [[ $DRY == 1 ]]; then echo "[dry] date -u > $RUN/DONE"; else date -u +%FT%TZ > "$RUN/DONE"; fi
  fi
  echo "[$TAG] done $(date -u +%FT%TZ); failed steps: ${FAILS[*]:-none}"
  ((${#FAILS[@]} == 0))
}
# common start: node facts, cfg values, merge dir, TERM trap, EXIT cleanup of merged folders (MERGED_TIERS)
begin() {
  echo "[$TAG] start $(date -u +%FT%TZ) RUN=$RUN uid=$(id -u) dry=$DRY deadline=${DEADLINE_S}s $*"
  vpd_step node vpd_node_facts "$RUN/node.txt" || true
  read -r BUILD FB < <(cfg render.build protocol.max_model_len_fallback) || true
  export VPD_MERGE_DIR=${VPD_MERGE_DIR:-$(pick_merge_dir)}
  echo "[$TAG] VPD_MERGE_DIR=${VPD_MERGE_DIR:-<none with 40 GB free>}"
  if [[ $DRY != 1 ]]; then
    trap 'echo "[$TAG] TERM: reaping"; reap_unity term; exit 143' TERM
    vpd_on_exit "reap_unity exit"
    if [[ -n $VPD_MERGE_DIR && -n $MERGED_TIERS ]]; then
      local t list=()
      for t in $MERGED_TIERS; do list+=("$VPD_MERGE_DIR/${PREFIX}__${START}__$t"); done
      vpd_on_exit "for d in $(printf '%q ' "${list[@]}"); do case \$d in /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) rm -rf -- \"\$d\";; esac; done"
    fi
  fi
  # the O9 fallback, decided by the calibration's shakedown before the freeze, applies to every later eval
  if [[ -f $FALLBACK_FILE ]]; then
    MML=(--max-model-len "$FB")
    echo "[$TAG] $FALLBACK_FILE exists: every eval runs with --max-model-len $FB (O9)"
  fi
}
# ---- end of the shared helpers ----

begin "START=$START P=$P"
NTOP=?
if [[ $DRY == 1 ]]; then NTOP=${VPD_DRY_NTOP:-ALL}; else read -r NTOP < <(cfg tiers.n_top) || true; fi
echo "[$TAG] N_top=$NTOP"
part guard
if ((GUARD_OK == 1)); then
  if [[ $NTOP != ALL && $NTOP != 400 ]]; then
    vpd_step "n_top:bad" false || true
    FAILS+=(n_top)
  else
    part train 0W "$TIERS/warmup0w20.jsonl" 0W 1001
    part train 100 "$TIERS/demos_100.jsonl" tier100 1001 0W
    part train "$NTOP" "$TIERS/demos_$NTOP.jsonl" "tier$NTOP" 1001 0W
    if [[ $START == "$REPLICATE_START" ]]; then
      part train "0W_s$REPLICATE_SEED" "$TIERS/warmup0w20.jsonl" "0W_s$REPLICATE_SEED" "$REPLICATE_SEED"
      part train "100_s$REPLICATE_SEED" "$TIERS/demos_100.jsonl" "tier100_s$REPLICATE_SEED" "$REPLICATE_SEED" \
        "0W_s$REPLICATE_SEED"
    fi
  fi
else
  vpd_step steps:guard_failed false || true
fi
part expect_list
finish
