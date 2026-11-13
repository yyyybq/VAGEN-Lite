#!/usr/bin/env bash
# sco payload of jobs vpd-tvr-eval-<s>-top and vpd-tvr-eval-<s>-low (CONTRACT_wave2 7.5; DECISIONS G10(g)(h); O2, O20,
# O22): the pilot evals of one start on all 500 eval.json tasks, always split in two jobs that may run in parallel
# after train-<s>; with VPD_TVR_ROOT=explore/rl (below) the EXPLORATORY jobs vpd-tvr-xeval-<q3rd|q3rs150>-<part>
# (DECISIONS G18).  as_root, one H800, gate tvr, from pilot_tvr.py:
#     env VPD_JOB_DEADLINE_S=<soft> timeout -k 120 <outer> bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/tvr_eval.sh START top|low RUN
# (outer / soft from the est_high DERIVED from the calibration, 5.3.1; refused above 5.5 h.)  P = OUT/tvr/runs/pilot/START.
#   node, prereg_committed, guard (pilot_tvr guard tvr-eval-<s>-<part>: prereg, smoke PASS, missing() == [], data
#   shas, the train-<s> adapters adapter-ok with the right init), merge_disk
#   canary   run_tvr replay sft2 -> P/canary/<part> + TVR_PY render_parity vs the data job's replay; a FAIL stops
#            this job's evals (renderer inconsistency: report and ask, never tune).  Rendered by every run on its
#            own node: an earlier run's P/canary/<part> is moved aside to P/canary/<part>.prev-<UTC> first.
#   top:  merge_<TOP> -> merge_parity_<TOP> (PY_TRAIN, before vLLM) -> eval_<TOP> = ONE run_tvr session, arms
#         c,u,blind -> rm -> official (per eval dir)
#   low:  merge_0W -> merge_parity_0W -> eval_0W (c) -> rm -> merge_100 -> merge_parity_100 -> eval_100 (c) -> rm
#         [Q3B: merge_100s1002 -> merge_parity_100s1002 -> eval_100s1002 (c, adapter 100_s1002) -> rm] -> official
#   official  one step per eval dir: PY_Q3 -m vpd.tvr.score_tvr official --run-dir D --served CODE
#             --bfs OUT/tvr/bfs/eval500_opt.json (TVRBench eval_va.aggregate in the TVR env; no Controller)
#   expect    pilot_tvr expect tvr-eval-<s>-<part> -> RUN/expect.json; RUN/DONE only if every step passed
# Eval dirs P/evals/pilot__<S>__<tier>__<arm>; merge parity P/parity/merge_<tier>.json (every merged eval, G0(7)).
# Skips: an eval with a fresh results.json (`pilot eval-ok DIR --adapter ADIR`); a stale one is flagged and kept
# (VPD_RERUN_STALE=1 moves it aside and redoes it; an eval-ok crash / unreadable results.json is recorded as
# eval_<tier>_<arm>:evalok_rc<N> and kept); a partial / invalid run dir resumes (run_tvr --resume, the full
# arm list always); merge parity reruns only if its json is missing or of another adapter sha.  Merged weights only
# on job-local disk, removed after each session and at EXIT; Unity reaped after every THOR step and at EXIT.  As root,
# RUN, P/evals, P/parity and P/canary are chowned to 20325 at EXIT (never P or runs/pilot).
# CPU: vLLM 4 + harness 1 + E x L THOR workers (image arms); the blind arm 16 harness threads, no THOR.
# Deadlines: soft VPD_JOB_DEADLINE_S (default 8160); `timeout -k 30 <left>`; THOR / vLLM --deadline-s <left - 240>.
# Env: VPD_DRYRUN=1 prints the commands; test hooks honoured ONLY with VPD_DRYRUN=1: VPD_TVR_OUT, VPD_DRY_NTOP (ALL).
# VPD_TVR_ROOT (DECISIONS G18): the output root under OUT/tvr/runs.  Unset or "pilot": the pilot jobs as above
# (START Q3B Q3S E2, job tvr-eval-<s>-<part>, P = runs/pilot/START, eval prefix pilot; a failed merge parity is a
# failure).  "explore/rl": the EXPLORATORY RL starts (START Q3RD Q3RS150, job tvr-xeval-<s>-<part>, P =
# runs/explore/rl/START, eval dirs P/evals/xrl__<S>__<tier>__<arm>, merged folders VPD_MERGE_DIR/xrl__<S>__<tier>; the
# same canary, merge, merge parity, eval, official and expect steps; merge parity is REPORT ONLY (DECISIONS G16 (1)):
# a json of the served adapter that says FAILED is recorded as merge_parity_<tier>:report_only, rc 0, while a crash
# or a missing json is still a failure).  Any other value: exit 2.  The third explore start Q3SG (PREREG section 17,
# row 2026-11-13; DECISIONS G20) runs under explore/rl the same way (jobs tvr-xeval-q3sg-<part>, P =
# runs/explore/rl/Q3SG, prefix xrl).

START=${1:-}
PART=${2:-}
RUN=${3:-}
TVR_ROOT=${VPD_TVR_ROOT:-pilot}
case $TVR_ROOT in
  pilot)
    case $START in
      Q3B|Q3S|E2) ;;
      *) echo "usage: tvr_eval.sh START top|low RUN   (START: Q3B Q3S E2)" >&2; exit 2 ;;
    esac
    JOB=tvr-eval-${START,,}-$PART
    PREFIX=pilot
    PARITY_GATED=1 ;;
  explore/rl)
    case $START in
      Q3RD|Q3RS150|Q3SG) ;;
      *) echo "usage: VPD_TVR_ROOT=explore/rl tvr_eval.sh START top|low RUN   (START: Q3RD Q3RS150 Q3SG)" >&2; exit 2 ;;
    esac
    JOB=tvr-xeval-${START,,}-$PART
    PREFIX=xrl
    PARITY_GATED=0 ;;
  *) echo "VPD_TVR_ROOT must be pilot or explore/rl, got '$TVR_ROOT'" >&2; exit 2 ;;
esac
case $PART in
  top|low) ;;
  *) echo "usage: tvr_eval.sh START top|low RUN" >&2; exit 2 ;;
esac
if [[ -z $RUN || $# -ne 3 ]]; then
  echo "usage: tvr_eval.sh START top|low RUN" >&2
  exit 2
fi
TAG=tvr_eval
NEED_PREREG=1
DEF_DEADLINE=8160
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
PARITY_DIR=$P/parity
OUTROOT=$P/evals
TASKSET=eval500
REPLICATE_START=Q3B   # pilot_cfg tiers.replicate (start Q3B, tier 100, seed 1002): eval label 100s1002
REPLICATE_SEED=1002
NTOP=?
if [[ $DRY == 1 ]]; then
  NTOP=${VPD_DRY_NTOP:-ALL}
else
  read -r NTOP < <("$PY_EVAL" -m vpd.tvr.pilot_tvr cfg tiers.n_top) || true
fi
if [[ $PART == top ]]; then
  MERGED_TIERS="$NTOP"
else
  MERGED_TIERS="0W 100 100s$REPLICATE_SEED"
fi
if [[ $DRY != 1 ]]; then
  mkdir -p "$P/evals" "$P/parity" "$P/canary/$PART"
  vpd_chown_trap "$RUN" "$P/evals" "$P/parity" "$P/canary"   # never P or runs/pilot (shared parents)
fi

# ---- helpers (identical in tvr_cal.sh, tvr_cal400.sh, tvr_mech_smoke.sh, tvr_train.sh, tvr_eval.sh; except that
# tvr_eval.sh adds parity_failed_kept / parity_step_failed, which its session() calls for the merge parity) ----
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
# merge parity failures (tvr_eval.sh only; session() below calls them).  PARITY_GATED=1 (pilot): a failure, as
# before.  PARITY_GATED=0 (explore, DECISIONS G16 (1): the bf16 U-M is reported, not gated): a json of THIS adapter
# that says FAILED (parity-ok 4) is recorded as merge_parity_<tier>:report_only (rc 0); a crash or a missing json /
# one of another adapter (parity-ok 1) is still a failure.
parity_failed_kept() {  # TIER: a current json says FAILED and is not rerun
  if ((PARITY_GATED == 0)); then
    vpd_step "merge_parity_$1:report_only" true
    return 0
  fi
  vpd_step "merge_parity_$1:failed_kept" false || true
  FAILS+=("merge_parity_$1")
}
parity_step_failed() {  # TIER FILE ADAPTER_DIR: the merge_parity step exited non-zero
  local st=0
  if ((PARITY_GATED == 0)); then
    parity_state "$2" "$3" || st=$?
    if ((st == 4)); then
      vpd_step "merge_parity_$1:report_only" true
      return 0
    fi
  fi
  FAILS+=("merge_parity_$1")
}
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
      parity_failed_kept "$tier"
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
      --merged "$local_dir" --demos "$TIERS/warmup0w20.jsonl" --n 20 --out "$pfile" \
      || parity_step_failed "$tier" "$pfile" "$addir"
  elif ((want_parity && pst == 4)); then
    parity_failed_kept "$tier"
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

official() {  # TIER ARM: TVRBench's own eval_va.aggregate() on the eval dir vs score_tvr (G0(12))
  local d=$OUTROOT/${PREFIX}__${START}__$1__$2
  if [[ $DRY != 1 && ! -f $d/results.json ]]; then
    vpd_step "official_$1_$2:no_results" false || true
    FAILS+=("official_$1_$2")
    return 1
  fi
  step "official_$1_$2" plain "$PY_Q3" -B -m vpd.tvr.score_tvr official --run-dir "$d" --served "$SERVED" \
    --bfs "$TVR/bfs/eval500_opt.json"
}

begin "START=$START PART=$PART P=$P N_top=$NTOP"
SERVED=?
if [[ $DRY == 1 ]]; then
  SERVED="<code:$START>"
else
  read -r SERVED < <("$PY_EVAL" -m vpd.tvr.pilot_tvr served "$START") || true
fi
part guard
part merge_disk
if ((GUARD_OK == 1)); then
  if [[ $NTOP != ALL && $NTOP != 400 ]]; then
    vpd_step "n_top:bad" false || true
    FAILS+=(n_top)
  else
    part canary "$P/canary/$PART"
    if [[ $PART == top ]]; then
      part need_adapter 0W
      part need_adapter "$NTOP" 0W
      part session "$NTOP" "$NTOP" 1 c,u,blind
      for A in c u blind; do
        part official "$NTOP" "$A"
      done
    else
      part need_adapter 0W
      part need_adapter 100 0W
      part session 0W 0W 1 c
      part session 100 100 1 c
      if [[ $START == "$REPLICATE_START" ]]; then
        part need_adapter "0W_s$REPLICATE_SEED"
        part need_adapter "100_s$REPLICATE_SEED" "0W_s$REPLICATE_SEED"
        part session "100s$REPLICATE_SEED" "100_s$REPLICATE_SEED" 1 c
      fi
      part official 0W c
      part official 100 c
      if [[ $START == "$REPLICATE_START" ]]; then
        part official "100s$REPLICATE_SEED" c
      fi
    fi
  fi
else
  vpd_step steps:guard_failed false || true
fi
part expect_list
finish
