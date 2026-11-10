#!/usr/bin/env bash
# sco payload of jobs vpd-wb-q3b / vpd-wb-q3s / vpd-wb-e2 (as_root; one H800; one start per job), from pilot.py:
#     env VPD_OPTIONAL=0|1 VPD_JOB_DEADLINE_S=<outer-180> timeout -k 120 <outer> \
#         bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_pilot_start.sh START RUN
# outer = floor(est_high*3600) - 300 for the est_high the job is submitted at (plan: 3.4 -> 11940 s, soft 11760 s),
# so the job never holds the card past the GPU-h charged for it (DECISIONS B2).  Soft deadline: every train /
# merge / eval runs under `timeout -k 30 <seconds left>`; a step with < VPD_MIN_STEP_S (default 60) left is
# recorded as NAME:no_time and not started.  The 180 s after the soft deadline are for the cleanup that must run
# (rm of merged weights, df_end, chown); a resubmitted job then continues after the cut (idempotent, see below).
# START in Q3B Q3S E2.  Outputs: P = OUT/walkerbench/runs/pilot/START/{adapters/{0W,50,200}, evals/<name>/}.
# Idempotent: each step is skipped when its output exists and is current, so a resubmitted job continues where the
# last one stopped.  Adapter (`pilot adapter-ok DIR [--init 0W]`): finished with the CURRENT training code, and
# for 50 / 200 started from the 0W adapter.pt now on disk (else retrained).  Eval (`pilot eval-ok`): results.json,
# and for a merged eval the adapter.pt it served is the one now on disk (else stale: flagged, kept, re-run only with
# VPD_RERUN_STALE=1 after the fix is logged in PREREG section 12).
#   1 train_0W        warmup20, --epochs auto, seed 1001 (retrained in every pilot job, DECISIONS B7)
#   2 train_50        --init-adapter 0W, tiers/seed1001/demos_50.jsonl
#   3 train_200       --init-adapter 0W, tiers/seed1001/demos_200.jsonl
#   4 eval raw 640 empty
#   5 merge 200 -> eval 200 640 empty -> eval 200 640 off -> rm
#   6 merge 50  -> eval 50 640 empty -> rm
#   7 eval raw 1280 empty
#   8 VPD_OPTIONAL=1 only: merge 0W -> eval 0W 640 empty -> rm
# Eval names pilot__<START>__<tier>__<res>__<mode> under P/evals (PREREG 7.2).  A failing eval or merge does not
# stop the other steps; a failing training step skips what depends on it.  The exit code is 1 if anything failed;
# RUN/DONE is written only when every step passed.  An adapter trained with other code (adapter-ok exit 3) stops
# the job: nothing is overwritten or deleted, a person decides.  An earlier results_invalid.json is re-run only
# with VPD_RERUN_INVALID=1 (a G0 fix is logged in PREREG section 12 first).
# Threads: training VPD_THREADS=2 (--threads 2) with 3 loader workers; vLLM 4 (VLLMServer); 8 eval workers (B9).
# Env: VPD_DRYRUN=1 prints the commands without running them (writes only RUN/steps.jsonl and RUN/node.txt);
#      VPD_JOB_DEADLINE_S (default 11760 = plan est_high 3.4), VPD_MIN_STEP_S (default 60).
#      Test hook honoured ONLY with VPD_DRYRUN=1: VPD_WB_ROOT (instead of OUT/walkerbench), so a dry run never sees
#      (and skips on) the real pilot outputs; a real job always reads and writes OUT/walkerbench.

START=${1:-}
RUN=${2:-${RUN:-}}
case $START in
  Q3B|Q3S|E2) ;;
  *) echo "usage: wb_pilot_start.sh START RUN   (START: Q3B Q3S E2)" >&2; exit 2 ;;
esac
if [[ -z $RUN ]]; then
  echo "usage: wb_pilot_start.sh START RUN" >&2
  exit 2
fi
export RUN START
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

DRY=${VPD_DRYRUN:-0}
OPT=${VPD_OPTIONAL:-0}
[[ $OPT == 0 || $OPT == 1 ]] || { echo "VPD_OPTIONAL must be 0 or 1" >&2; exit 2; }
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-11760}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_MIN_STEP_S must be a positive integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))
WB=$VPD_OUT/walkerbench
if [[ $DRY == 1 && -n ${VPD_WB_ROOT:-} ]]; then
  WB=$VPD_WB_ROOT   # test hook (dry run only)
fi
P=$WB/runs/pilot/$START
AD=$P/adapters
EV=$P/evals
WARM=$WB/warmup/warmup20.jsonl
TIERS=$WB/tiers/seed1001
FAILS=()

mkdir -p "$RUN"
cd "$VPD_WORKTREE"
if [[ $DRY != 1 ]]; then
  mkdir -p "$AD" "$EV"
  vpd_chown_trap "$RUN" "$P"
  # pilot prepare creates runs/ and runs/pilot as the login user; should they still be missing (created here, as
  # root), hand them back too.  Not recursive: the other starts' dirs may be live.
  if [[ $(id -u) -eq 0 ]]; then
    vpd_on_exit "chown ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q ' "$WB/runs" "$WB/runs/pilot")|| true"
  fi
fi

x() {  # run, or print under VPD_DRYRUN=1
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*"
  else
    "$@"
  fi
}

left() { echo $((DEADLINE - $(date +%s))); }

# timed NAME CMD...: vpd_step NAME under the soft deadline (timeout -k 30 <seconds left>); with fewer than
# MIN_STEP_S seconds left the step is not started and is recorded as NAME:no_time.  Returns the step's status.
timed() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t < MIN_STEP_S)); then
    echo "[wb_pilot] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@"
}

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
  local d=$1
  case $d in
    /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) x rm -rf -- "$d" ;;
    *) echo "[wb_pilot] refusing to remove $d" >&2; return 1 ;;
  esac
}

# adapter_state DIR [INIT_DIR] -> 0 complete with the current code (and started from INIT_DIR/adapter.pt as it is
# now), 1 missing / unfinished / started from another 0W (retrain), 3 other code (stop)
adapter_state() {
  local args=("$1")
  [[ -n ${2:-} ]] && args+=(--init "$2")
  [[ $DRY == 1 ]] && { echo "[dry] $PY_EVAL -m vpd.walkerbench.pilot adapter-ok ${args[*]}"; return 1; }
  local rc=0
  "$PY_EVAL" -m vpd.walkerbench.pilot adapter-ok "${args[@]}" || rc=$?
  return "$rc"
}

declare -A TRAINED=()
train() {  # TIER DATA [INIT_DIR]
  local tier=$1 data=$2 init=${3:-} out=$AD/$1 st=0 phase=$1
  [[ $tier != 0W ]] && phase=tier$tier
  adapter_state "$out" "$init" || st=$?
  if ((st == 0)); then
    vpd_step "train_$tier:skip" true
    TRAINED[$tier]=1
    return 0
  fi
  if ((st == 3)); then
    echo "[wb_pilot] $out was trained with other code: stopping (nothing overwritten)" >&2
    vpd_step "train_$tier:code_mismatch" false || true
    exit 3
  fi
  local args=(--start "$START" --data "$data" --out "$out" --epochs auto --seed 1001 --phase "$phase"
              --threads "$VPD_THREADS" --num-workers 3)
  [[ -n $init ]] && args+=(--init-adapter "$init")
  if timed "train_$tier" "$PY_TRAIN" vpd/train/n2_train_vpd.py "${args[@]}"; then
    TRAINED[$tier]=1
  else
    FAILS+=("train_$tier")
  fi
}

# eval_state TIER RES MODE -> 0 results.json present (merged: it served adapters/TIER/adapter.pt as it is now),
# 1 none, 3 stale (served other weights, e.g. the tier was retrained since: flagged, re-run only with
# VPD_RERUN_STALE=1 after the fix is logged in PREREG section 12)
eval_state() {
  local dir=$EV/pilot__${START}__$1__$2__$3 rc=0
  [[ -f $dir/results.json ]] || return 1
  [[ $1 == raw ]] && return 0
  "$PY_EVAL" -m vpd.walkerbench.pilot eval-ok "$dir" --adapter "$AD/$1" || rc=$?
  ((rc == 0 || rc == 3)) || rc=3  # results.json present but not judged fresh: never re-run without a person
  return "$rc"
}

evalrun() {  # TIER RES MODE [MODEL_DIR ADAPTER_META]
  local tier=$1 res=$2 mode=$3 md=${4:-} meta=${5:-} st=0
  local name="pilot__${START}__${tier}__${res}__${mode}" size=640x400
  [[ $res == 1280 ]] && size=1280x800
  local dir=$EV/$name
  eval_state "$tier" "$res" "$mode" || st=$?
  if ((st == 0)); then
    vpd_step "eval_${tier}_${res}_${mode}:skip" true
    return 0
  fi
  local args=(--model "$START" --task-set s2test --render-size "$size" --constrained "$mode" --history-turns 0
              --workers 8 --run-name "$name" --out-root "$EV")
  if ((st == 3)); then
    if [[ ${VPD_RERUN_STALE:-0} != 1 ]]; then
      echo "[wb_pilot] $dir served other weights than $AD/$tier/adapter.pt; set VPD_RERUN_STALE=1 to re-run it" >&2
      vpd_step "eval_${tier}_${res}_${mode}:stale_kept" false || true
      FAILS+=("eval_${tier}_${res}_${mode}")
      return 1
    fi
    args+=(--overwrite)
  elif [[ -f $dir/results_invalid.json ]]; then
    if [[ ${VPD_RERUN_INVALID:-0} != 1 ]]; then
      echo "[wb_pilot] $dir has results_invalid.json; set VPD_RERUN_INVALID=1 to re-run it" >&2
      vpd_step "eval_${tier}_${res}_${mode}:invalid_kept" false || true
      FAILS+=("eval_${tier}_${res}_${mode}")
      return 1
    fi
    args+=(--overwrite)
  fi
  [[ -n $md ]] && args+=(--model-dir "$md" --adapter-meta "$meta")
  if ! timed "eval_${tier}_${res}_${mode}" "$PY_Q3" -m vpd.walkerbench.run_vpd "${args[@]}"; then
    FAILS+=("eval_${tier}_${res}_${mode}")
    return 1
  fi
}

merged_evals() {  # TIER RES:MODE...   merge TIER, run its evals on the merged folder, remove the folder
  local tier=$1; shift
  local spec st todo=()
  for spec in "$@"; do  # fresh evals are skipped; stale ones are flagged here unless VPD_RERUN_STALE=1
    st=0
    eval_state "$tier" "${spec%%:*}" "${spec##*:}" || st=$?
    if ((st == 0)); then
      vpd_step "eval_${tier}_${spec%%:*}_${spec##*:}:skip" true
    elif ((st == 3)) && [[ ${VPD_RERUN_STALE:-0} != 1 ]]; then
      echo "[wb_pilot] eval $tier $spec served other weights than $AD/$tier/adapter.pt; VPD_RERUN_STALE=1 re-runs it" >&2
      vpd_step "eval_${tier}_${spec%%:*}_${spec##*:}:stale_kept" false || true
      FAILS+=("eval_${tier}_${spec%%:*}_${spec##*:}")
    else
      todo+=("$spec")
    fi
  done
  if ((${#todo[@]} == 0)); then
    vpd_step "merge_$tier:skip" true
    return 0
  fi
  set -- "${todo[@]}"
  if [[ -z ${TRAINED[$tier]:-} ]]; then
    echo "[wb_pilot] adapter $tier not trained: its evals are skipped" >&2
    vpd_step "merge_$tier:no_adapter" false || true
    FAILS+=("merge_$tier")
    return 1
  fi
  if [[ -z $VPD_MERGE_DIR ]]; then
    echo "[wb_pilot] no job-local disk with 40 GB free: merge_$tier skipped" >&2
    vpd_step "merge_$tier:no_disk" false || true
    FAILS+=("merge_$tier")
    return 1
  fi
  local local_dir=$VPD_MERGE_DIR/${START}__$tier
  safe_rm_merged "$local_dir" || true
  if timed "merge_$tier" "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$AD/$tier" --out "$local_dir"; then
    for spec in "$@"; do
      evalrun "$tier" "${spec%%:*}" "${spec##*:}" "$local_dir" "$AD/$tier/meta.json" || true
    done
  else
    FAILS+=("merge_$tier")
  fi
  vpd_step "rm_merged_$tier" safe_rm_merged "$local_dir" || FAILS+=("rm_merged_$tier")
}

echo "[wb_pilot] start $(date -u +%FT%TZ) START=$START RUN=$RUN P=$P optional=$OPT dry=$DRY deadline=${DEADLINE_S}s"
vpd_node_facts "$RUN/node.txt"
for f in "$WARM" "$TIERS/demos_50.jsonl" "$TIERS/demos_200.jsonl"; do
  if [[ ! -f $f ]]; then
    echo "[wb_pilot] missing $f: run job wb-demos first" >&2
    [[ $DRY == 1 ]] || exit 2
  fi
done
export VPD_MERGE_DIR=${VPD_MERGE_DIR:-$(pick_merge_dir)}
echo "[wb_pilot] VPD_MERGE_DIR=${VPD_MERGE_DIR:-<none with 40 GB free>}"
if [[ -n $VPD_MERGE_DIR && $DRY != 1 ]]; then  # a timeout must not leave merged weights on the node
  vpd_on_exit "for d in '$VPD_MERGE_DIR/${START}__0W' '$VPD_MERGE_DIR/${START}__50' '$VPD_MERGE_DIR/${START}__200'; do case \$d in /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) rm -rf -- \"\$d\";; esac; done"
fi

# 1-3 training (0W first; 50 and 200 start from it)
train 0W "$WARM"
if [[ -n ${TRAINED[0W]:-} ]]; then
  train 50 "$TIERS/demos_50.jsonl" "$AD/0W"
  train 200 "$TIERS/demos_200.jsonl" "$AD/0W"
else
  echo "[wb_pilot] 0W failed: tiers 50 and 200 are not trained" >&2
fi
# 4 raw 640
evalrun raw 640 empty || true
# 5 tier 200: constrained (G1-G3) and unconstrained (secondary)
merged_evals 200 640:empty 640:off || true
# 6 tier 50
merged_evals 50 640:empty || true
# 7 raw 1280
evalrun raw 1280 empty || true
# 8 optional 0W
if [[ $OPT == 1 ]]; then
  merged_evals 0W 640:empty || true
fi

df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]} == 0)); then
  if [[ $DRY == 1 ]]; then
    echo "[dry] date -u > $RUN/DONE"
  else
    date -u +%FT%TZ > "$RUN/DONE"
  fi
fi
echo "[wb_pilot] done $(date -u +%FT%TZ) START=$START; failed steps: ${FAILS[*]:-none}"
((${#FAILS[@]} == 0))
