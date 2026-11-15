#!/usr/bin/env bash
# sco payload of the EXPLORATORY round-5 experiment B, "forced opening actions" (job tvr-xint-<s>-<set>): START's pilot
# ALL adapter (arm c) re-run on the tasks of SET with its first K actions FORCED to the other pilot start's (the
# donor's) pilot actions, K = 0,1,2,3 (smoke: 0,2; K = 0 is a plain re-run).  Not gated, outside the prereg's decision
# table; never writes runs/pilot or runs/explore/rl.  Modelled on tvr_eval.sh (its helpers below).  as_root, one H800:
#     env VPD_JOB_DEADLINE_S=<soft> timeout -k 120 <outer> bash <code>/vpd/jobs/tvr_intervene.sh START SET RUN
# START Q3B|Q3S (donor D: the other one); SET T101|T55|smoke.  P = OUT/tvr/runs/explore/intervene/START/SET.
# Inputs, built on the login node by `python -m vpd.tvr.intervene sets|pick|donors` (never by this job):
#   I = OUT/tvr/runs/explore/intervene/inputs: I/SET.json (the task subset) and I/forced_<START>_<SET>.json (the donor's
#   actions of those tasks: vpd.tvr.forced/1 from D's pilot dir runs/pilot/D/evals/pilot__D__ALL__c).
#   node, prereg_committed (the snapshot's PREREG file is the committed one), provenance (run_tvr_forced provenance
#   -> RUN/provenance.json: run_tvr's code_sha256, the module's sha256, harness_r5_sha256), guard (`intervene check
#   --set SET`: the set file is the sets manifest's (T101 = 101, T55 = 55; smoke = 8 tasks of them), the manifest's
#   A / B dirs are the pilot's Q3S / Q3B ALL c dirs, the forced file was built from it with the donor's actions of
#   every task, every donor trajectory unchanged; RUN/guard_inputs.json records the inputs' sha256), merge_disk
#   (VPD_MERGE_DIR is /tmp/vpd_merged or /dev/shm/vpd_merged, or below them: pick_merge_dir's choices)
#   canary    run_tvr replay sft2 -> P/canary/sft2 + TVR_PY render_parity vs the data job's replay (tvr_eval.sh's
#             canary; a FAIL stops the eval: report and ask, never tune)
#   adapter   runs/pilot/START/adapters/ALL (read only): pilot_tvr.x_adapter_check on the pilot root (complete, the
#             pilot's registered trainer prefix, meta start / seed / phase / data_sha256, from adapters/0W as it is
#             now) and `intervene adapter-match` (its adapter.pt sha256 == the one the pilot ALL c eval served).
#             tvr_eval.sh's `pilot adapter-ok` judges by this tree's trainer code, which changed with the Q3SG row
#             (2026-11-13): it calls every pilot adapter "other code" (exit 3).
#   session   merge ALL -> merge parity P/parity/merge_ALL.json (PY_TRAIN, before vLLM; REPORT ONLY, as explore/rl)
#             -> ONE run_tvr_forced eval session: --tier ALL --arms c --task-set eval500 --task-subset I/SET.json
#             --forced I/forced_<START>_<SET>.json --forced-k 0,1,2,3 (smoke 0,2) --resume -> rm merged weights
#   official  per run dir: PY_Q3 -m vpd.tvr.score_tvr official --run-dir D --served CODE
#             --bfs OUT/tvr/bfs/eval500_opt.json (tvr_eval.sh's official step)
#   expect    `intervene expect` -> RUN/expect.json; RUN/DONE only if every step passed
# Run dirs P/evals/xint__<START>__ALL__c__f<K>.  Before the session, every K dir: one run with other inputs or code
# (`intervene runstate`: its protocol "forced" / "task_subset" are not those of I/SET.json, the forced file and this
# harness_r5) or one with a results.json that served other weights (`pilot eval-ok DIR --adapter ADIR`) is stale:
# flagged and kept (the step fails), or, with VPD_RERUN_STALE=1, moved aside to DIR.stale-<UTC> and redone.  When
# every K has a fresh results.json the merge and the session are skipped; otherwise the session resumes
# (run_tvr_forced --resume, every K always).  Merged weights only on job-local disk (VPD_MERGE_DIR/
# xint__<START>__<SET>__ALL), removed after the session and at EXIT; Unity reaped after every THOR step and at EXIT.
# As root, at EXIT: chown -R 20325 of RUN, P/evals, P/parity, P/canary (this job's outputs) and chown (not -R) of the
# parent directories this job created (intervene, intervene/START, P); never runs/pilot, runs/explore or a parent that
# existed before.
# CPU: vLLM 4 + harness 1 + E x L THOR workers.  Deadlines: soft VPD_JOB_DEADLINE_S (default 8160); `timeout -k 30
# <left>`; THOR / vLLM --deadline-s <left - 240>.
# Env: VPD_DRYRUN=1 prints the commands; test hooks honoured ONLY with VPD_DRYRUN=1: VPD_TVR_OUT.

START=${1:-}
SET=${2:-}
RUN=${3:-}
case $START in
  Q3B) DONOR=Q3S ;;
  Q3S) DONOR=Q3B ;;
  *) echo "usage: tvr_intervene.sh START SET RUN   (START: Q3B Q3S; SET: T101 T55 smoke)" >&2; exit 2 ;;
esac
case $SET in
  T101|T55) KS=0,1,2,3 ;;
  smoke) KS=0,2 ;;
  *) echo "usage: tvr_intervene.sh START SET RUN   (START: Q3B Q3S; SET: T101 T55 smoke)" >&2; exit 2 ;;
esac
if [[ -z $RUN || $# -ne 3 ]]; then
  echo "usage: tvr_intervene.sh START SET RUN" >&2
  exit 2
fi
JOB=tvr-xint-${START,,}-${SET,,}
PREFIX=xint
TIER=ALL
PARITY_GATED=0   # merge parity is report only (exploratory, as explore/rl: DECISIONS G16 (1))
TAG=tvr_intervene
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
XROOT=$TVR/runs/explore/intervene
P=$XROOT/$START/$SET
PILOT=$TVR/runs/pilot                # read only: the pilot's adapters and eval dirs
AD=$PILOT/$START/adapters
OWN_DIR=$PILOT/$START/evals/pilot__${START}__ALL__c
DONOR_DIR=$PILOT/$DONOR/evals/pilot__${DONOR}__ALL__c
PARITY_DIR=$P/parity
OUTROOT=$P/evals
TASKSET=eval500
INPUTS=$XROOT/inputs
SUBSET=$INPUTS/$SET.json
FORCED=$INPUTS/forced_${START}_$SET.json
SETS_MANIFEST=$INPUTS/sets_manifest.json
MERGED_NAME=${PREFIX}__${START}__${SET}__$TIER   # the job's merged folder (VPD_MERGE_DIR/MERGED_NAME)
HARNESS_R5=
if [[ $DRY != 1 ]]; then
  NEW_PARENTS=()
  for d in "$XROOT" "$XROOT/$START" "$P"; do
    [[ -e $d ]] || NEW_PARENTS+=("$d")
  done
  mkdir -p "$P/evals" "$P/parity" "$P/canary"
  vpd_chown_trap "$RUN" "$P/evals" "$P/parity" "$P/canary"   # this job's outputs (never runs/pilot, explore/rl)
  if ((${#NEW_PARENTS[@]})) && [[ $(id -u) -eq 0 ]]; then    # the parents this job created: themselves, not -R
    q=$(printf '%q ' "${NEW_PARENTS[@]}")
    vpd_on_exit "chown ${VPD_JOB_UID}:${VPD_JOB_UID} $q || echo '[vpd] chown failed' >&2"
  fi
fi

# ---- helpers (tvr_eval.sh's shared helpers, those this payload uses, unchanged; tvr_eval.sh's session() / guard() /
# expect_list() / need_adapter() are replaced by session_forced() / guard() / expect_list() / need_pilot_adapter()) ----
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

GUARD_OK=0
# provenance: the forced harness's (run_tvr_forced provenance -> RUN/provenance.json); HARNESS_R5 = its
# harness_r5_sha256 (dry run: a placeholder)
provenance() {
  if [[ $DRY == 1 ]]; then
    echo "[dry] $PY_Q3 -B -m vpd.tvr.run_tvr_forced provenance --out $RUN/provenance.json"
    HARNESS_R5='<harness_r5>'
    return 0
  fi
  HARNESS_R5=$("$PY_Q3" -B -m vpd.tvr.run_tvr_forced provenance --out "$RUN/provenance.json") || return 1
  [[ $HARNESS_R5 =~ ^[0-9a-f]{64}$ ]]
}
# guard: prereg_committed = the snapshot's PREREG_TVR_PILOT.md is the committed file (vpd_snapshot_file), the
# provenance, then `intervene check` on the inputs (head); a failure skips every model step (expect still runs)
guard() {
  if ((NEED_PREREG)); then
    vpd_step prereg_committed x vpd_snapshot_file vpd/PREREG_TVR_PILOT.md || { FAILS+=(prereg_committed); return 1; }
  fi
  vpd_step provenance provenance || { FAILS+=(provenance); return 1; }
  vpd_step guard x "$PY_EVAL" -B -m vpd.tvr.intervene check --set "$SET" --tasks "$SUBSET" --sets-manifest \
    "$SETS_MANIFEST" --pilot-root "$PILOT" --start "$START" --forced "$FORCED" --donor "$DONOR" --source "$DONOR_DIR" \
    --out "$RUN/guard_inputs.json" || { FAILS+=(guard); return 1; }
  GUARD_OK=1
}
# merge_disk: a job-local merge dir, one of pick_merge_dir's (/tmp/vpd_merged, /dev/shm/vpd_merged) or below them
merge_disk_ok() {
  case $VPD_MERGE_DIR in
    */..|*/../*|*/.|*/./*) return 1 ;;
    /tmp/vpd_merged|/tmp/vpd_merged/?*|/dev/shm/vpd_merged|/dev/shm/vpd_merged/?*) return 0 ;;
    *) return 1 ;;
  esac
}
merge_disk() {
  vpd_step merge_disk merge_disk_ok || {
    echo "[$TAG] VPD_MERGE_DIR='${VPD_MERGE_DIR:-}' is not /tmp/vpd_merged or /dev/shm/vpd_merged (or below)" >&2
    FAILS+=(merge_disk)
    return 1
  }
}

# need_pilot_adapter: AD/ALL complete and as registered (pilot_tvr.x_adapter_check with the pilot root) and the one
# the pilot ALL c eval served (intervene adapter-match); see the head
ADAPTER_OK=0
XADAPTER_PY="import sys; from vpd.tvr import pilot_tvr as P; L = P.Layout(); "
XADAPTER_PY+="rc, msg = P.x_adapter_check(sys.argv[1], sys.argv[2], sys.argv[3], L, adapter_root=L.pilot_root); "
XADAPTER_PY+="print('[xadapter-ok]', msg); sys.exit(rc)"
need_pilot_adapter() {
  if vpd_step "adapter_$TIER" x "$PY_EVAL" -B -c "$XADAPTER_PY" "$START" "$TIER" 0W \
      && vpd_step "adapter_match_$TIER" x "$PY_EVAL" -B -m vpd.tvr.intervene adapter-match --adapter-dir "$AD/$TIER" \
        --eval-dir "$OWN_DIR"; then
    ADAPTER_OK=1
    return 0
  fi
  vpd_step "adapter_$TIER:not_ready" false || true
  FAILS+=("adapter_$TIER")
  return 1
}
# run_state DIR K -> 0 no run yet or a run of these inputs and this harness, 3 a run of others (intervene runstate)
run_state() {
  local rc=0
  x "$PY_EVAL" -B -m vpd.tvr.intervene runstate --dir "$1" --k "$2" --tasks "$SUBSET" --forced "$FORCED" \
    --harness-r5 "$HARNESS_R5" || rc=$?
  return "$rc"
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
# merge parity failures (tvr_eval.sh's).  PARITY_GATED=0 here (exploratory: the bf16 U-M is reported, not gated): a
# json of THIS adapter that says FAILED (parity-ok 4) is recorded as merge_parity_<tier>:report_only (rc 0); a crash or
# a missing json / one of another adapter (parity-ok 1) is still a failure.
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
# session_forced: tvr_eval.sh's session() for the forced runs.  The run dirs of every K (fresh: skip; stale inputs /
# code or weights: kept and flagged unless VPD_RERUN_STALE=1), merge AD/ALL on job-local disk, merge parity (PY_TRAIN,
# before vLLM; report only), ONE run_tvr_forced session for every K on the merged folder (--resume; every K always),
# remove the folder.  No eval after a failed canary.
session_forced() {
  local addir=$AD/$TIER k st todo=0 stale=0 dir pst=0 pfile=$PARITY_DIR/merge_$TIER.json
  if ((CANARY_OK != 1)); then
    vpd_step "eval_$TIER:canary_failed" false || true
    FAILS+=("eval_$TIER")
    return 1
  fi
  for k in ${KS//,/ }; do
    dir=$OUTROOT/${PREFIX}__${START}__${TIER}__c__f$k
    st=0
    run_state "$dir" "$k" || st=$?
    if ((st == 3)) && [[ ${VPD_RERUN_STALE:-0} == 1 ]]; then
      vpd_step "stale_aside_${TIER}_f$k" x mv -T -- "$dir" "$dir.stale-$(date -u +%Y%m%dT%H%M%SZ)" || stale=1
      todo=1
      continue
    elif ((st == 3)); then
      echo "[$TAG] $dir holds a run of other inputs or code (its protocol forced / task_subset are not those of" \
        "$SUBSET, $FORCED and harness_r5 $HARNESS_R5): kept; VPD_RERUN_STALE=1 moves it aside and re-runs it" >&2
      vpd_step "eval_${TIER}_f$k:stale_inputs" false || true
      stale=1
      continue
    elif ((st != 0)); then
      echo "[$TAG] intervene runstate on $dir exited $st: kept" >&2
      vpd_step "eval_${TIER}_f$k:runstate_rc$st" false || true
      stale=1
      continue
    fi
    eval_state "$dir" "$addir" || st=$?
    if ((st == 3)); then
      if ((EVALOK_RC != 3)); then
        echo "[$TAG] eval-ok on $dir exited $EVALOK_RC (unreadable results.json or a broken interpreter): kept" >&2
        vpd_step "eval_${TIER}_f$k:evalok_rc$EVALOK_RC" false || true
        stale=1
      elif [[ ${VPD_RERUN_STALE:-0} == 1 ]]; then
        vpd_step "stale_aside_${TIER}_f$k" x mv -T -- "$dir" "$dir.stale-$(date -u +%Y%m%dT%H%M%SZ)" || stale=1
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
    vpd_step "eval_$TIER:stale_kept" false || true
    FAILS+=("eval_$TIER")
    return 1
  fi
  pst=0
  parity_state "$pfile" "$addir" || pst=$?
  if ((todo == 0 && pst != 1)); then
    vpd_step "merge_$TIER:skip" true
    if ((pst == 4)); then
      parity_failed_kept "$TIER"
    else
      vpd_step "merge_parity_$TIER:skip" true
    fi
    vpd_step "eval_$TIER:skip" true
    return 0
  fi
  if ((ADAPTER_OK != 1)); then
    vpd_step "merge_$TIER:no_adapter" false || true
    FAILS+=("merge_$TIER")
    return 1
  fi
  if ! merge_disk_ok; then
    vpd_step "merge_$TIER:no_disk" false || true
    FAILS+=("merge_$TIER")
    return 1
  fi
  local local_dir=$VPD_MERGE_DIR/$MERGED_NAME
  safe_rm_merged "$local_dir" || true
  if ! timed "merge_$TIER" "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$addir" --out "$local_dir"; then
    FAILS+=("merge_$TIER")
    vpd_step "rm_merged_$TIER" safe_rm_merged "$local_dir" || true
    return 1
  fi
  if ((pst == 1)); then
    timed "merge_parity_$TIER" "$PY_TRAIN" -B -m vpd.tvr.merge_parity --start "$START" --adapter "$addir" \
      --merged "$local_dir" --demos "$TIERS/warmup0w20.jsonl" --n 20 --out "$pfile" \
      || parity_step_failed "$TIER" "$pfile" "$addir"
  elif ((pst == 4)); then
    parity_failed_kept "$TIER"
  else
    vpd_step "merge_parity_$TIER:skip" true
  fi
  if ((todo)); then
    timed_thor "eval_$TIER" "$PY_Q3" -B -m vpd.tvr.run_tvr_forced eval --start "$START" --tier "$TIER" --arms c \
      --task-set "$TASKSET" --name-prefix "$PREFIX" --out-root "$OUTROOT" --model-dir "$local_dir" \
      --adapter-meta "$addir/meta.json" --task-subset "$SUBSET" --forced "$FORCED" --forced-k "$KS" \
      ${MML[@]+"${MML[@]}"} --resume || FAILS+=("eval_$TIER")
    reap "eval_$TIER"
  else
    vpd_step "eval_$TIER:skip" true
  fi
  vpd_step "rm_merged_$TIER" safe_rm_merged "$local_dir" || FAILS+=("rm_merged_$TIER")
}

# canary DIR: tvr_eval.sh's (run_tvr replay sft2 into DIR as this job's user and node, then the frozen tolerance check
# against the data job's replay; CANARY_OK=0 on any failure; an earlier run's DIR is moved aside to DIR.prev-<UTC>)
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

official() {  # K: TVRBench's own eval_va.aggregate() on the run dir vs score_tvr (tvr_eval.sh's official)
  local d=$OUTROOT/${PREFIX}__${START}__${TIER}__c__f$1
  if [[ $DRY != 1 && ! -f $d/results.json ]]; then
    vpd_step "official_f$1:no_results" false || true
    FAILS+=("official_f$1")
    return 1
  fi
  step "official_f$1" plain "$PY_Q3" -B -m vpd.tvr.score_tvr official --run-dir "$d" --served "$SERVED" \
    --bfs "$TVR/bfs/eval500_opt.json"
}

EXPECT_OK=0
expect_list() {  # `intervene expect`: every K valid, its protocol / official check / forced prefixes (module doc)
  if vpd_step expect x "$PY_EVAL" -B -m vpd.tvr.intervene expect --out-root "$OUTROOT" --prefix "$PREFIX" \
      --start "$START" --tier "$TIER" --ks "$KS" --tasks "$SUBSET" --forced "$FORCED" --harness-r5 "$HARNESS_R5" \
      --out "$RUN/expect.json"; then
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
# common start: node facts, cfg values, merge dir, TERM trap, EXIT cleanup of the merged folder (MERGED_NAME)
begin() {
  echo "[$TAG] start $(date -u +%FT%TZ) RUN=$RUN uid=$(id -u) dry=$DRY deadline=${DEADLINE_S}s $*"
  vpd_step node vpd_node_facts "$RUN/node.txt" || true
  read -r BUILD FB < <(cfg render.build protocol.max_model_len_fallback) || true
  export VPD_MERGE_DIR=${VPD_MERGE_DIR:-$(pick_merge_dir)}
  echo "[$TAG] VPD_MERGE_DIR=${VPD_MERGE_DIR:-<none with 40 GB free>}"
  if [[ $DRY != 1 ]]; then
    trap 'echo "[$TAG] TERM: reaping"; reap_unity term; exit 143' TERM
    vpd_on_exit "reap_unity exit"
    if [[ -n $VPD_MERGE_DIR ]]; then
      local q
      q=$(printf '%q ' "$VPD_MERGE_DIR/$MERGED_NAME")
      vpd_on_exit "for d in $q; do case \$d in /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) rm -rf -- \"\$d\";;" \
        "esac; done"
    fi
  fi
  # the O9 fallback, decided by the calibration's shakedown before the freeze, applies to every later eval
  if [[ -f $FALLBACK_FILE ]]; then
    MML=(--max-model-len "$FB")
    echo "[$TAG] $FALLBACK_FILE exists: every eval runs with --max-model-len $FB (O9)"
  fi
}
# ---- end of the helpers ----

begin "JOB=$JOB START=$START SET=$SET DONOR=$DONOR K=$KS P=$P"
SERVED=?
if [[ $DRY == 1 ]]; then
  SERVED="<code:$START>"
else
  read -r SERVED < <("$PY_EVAL" -m vpd.tvr.pilot_tvr served "$START") || true
fi
part guard
part merge_disk
if ((GUARD_OK == 1)); then
  part canary "$P/canary/sft2"
  part need_pilot_adapter
  part session_forced
  for K in ${KS//,/ }; do
    part official "$K"
  done
else
  vpd_step steps:guard_failed false || true
fi
part expect_list
finish
