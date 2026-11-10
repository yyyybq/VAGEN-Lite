#!/usr/bin/env bash
# EXPLORATORY (not part of the pre-registered pilot; no decision rule, numbers only): WalkerBench learnability
# check of Q3B on all 483 train demos (user request 2026-11-09).  See vpd/EXPLORATORY_WB_LEARN483.md.
# sco payload of vpd-wb-x483a (ARM A) and vpd-wb-x483b (ARM B), as_root (Q3B blobs are root-only), one H800 each:
#     env VPD_JOB_DEADLINE_S=15720 timeout -k 120 15900 \
#         bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_learn483.sh A|B RUN
# est_high 4.5 GPU-h -> outer = floor(4.5*3600) - 300 = 15900 s, soft deadline = outer - 180 = 15720 s (the pilot's
# rule, pilot.job_timeouts).  Every train / merge / eval / probe runs under `timeout -k 30 <seconds left>`; a step
# with < VPD_MIN_STEP_S (60) left is recorded as NAME:no_time and not started; the 180 s after the soft deadline are
# for the cleanup (merged weights, df_end, chown).  A resubmitted job (new RUN dir) continues where this one stopped.
#
# Root X = OUT/walkerbench/runs/explore/x483/Q3B (VPD_X483_ROOT overrides: tests only).  The pilot's Q3B adapters
# P/adapters/{0W,50,200} (P = OUT/walkerbench/runs/pilot/Q3B; VPD_X483_PILOT overrides: tests only) are read only:
# nothing is written under runs/pilot/ (X there is refused) and no pilot adapter is ever (re)trained here.
# ARM A (the user-requested number first):
#   0 setup        learn483 setup (s2in50 task set created or verified; X dirs)
#   1 floors       STOP0 / FWD / ODO / RM on s2val and s2in50 (CPU, rendering stubbed) -> X/floors_<set>.json
#   2 train_483A   learn483 init-check (P/0W adapter.pt sha256 5cdbfd4366a3, trainer code 63a3a8b92ce7), then
#                  n2_train_vpd.py --start Q3B --data tiers/seed1001/demos_all.jsonl --init-adapter P/adapters/0W
#                  --out X/adapters/483A --epochs auto --seed 1001 --phase tier483 --threads 2 --num-workers 3
#                  (auto = max(3, ceil(240/483)) = 3 epochs: 1449 micro-steps, 182 optimizer steps)
#   3 merge 483A -> X1 s2test, X9 s2test --blind, X2 s2val, X5 s2in50, X12 s2test --workers 4 (rep2);
#                  merge parity (learn483 tf-parity: TF reply CE, first 20 VAL56 demos, merged vs LoRA); rm
#   4 tf_A         learn483 tf: 483A, raw, 0W, 50, 200 on VAL56 / TR200 / TR283 (greedy ROTATION on VAL56)
#   5 merge 200  -> X3 s2val, X6 s2in50; rm
#   6 merge 50   -> X7 s2in50; rm
#   7 raw        X4 s2val, X8 s2in50, X10 s2test 640 --constrained off --max-tokens 1024,
#                X11 s2test 1280x800 --constrained off --max-tokens 2048
# ARM B: setup; train_483B (= 483A with --epochs 10: 4830 micro-steps, 604 optimizer steps);
#        merge 483B -> Y1 s2test, Y2 s2val, Y3 s2in50; rm; tf_B (483B on the three demo sets).
# Evals: `PY_Q3 -m vpd.walkerbench.learn483 eval` = run_vpd.main with the task set s2in50 registered at run time;
#   --model Q3B --history-turns 0 --temperature 0 --workers 8 (rep2: 4), 640x400 --constrained empty
#   --max-tokens 512 unless stated, merged weights with --model-dir LOCAL --adapter-meta DIR/meta.json; names
#   explore__Q3B__<model>__<set>__<res>__<mode>[__blind|__rep2] under X/evals.
# Idempotent as wb_pilot_start.sh: an adapter is skipped when `pilot adapter-ok DIR --init P/adapters/0W` passes,
# an eval when results.json exists (merged: `pilot eval-ok DIR --adapter DIR` = it served the adapter.pt now on
# disk; stale evals are flagged and kept unless VPD_RERUN_STALE=1, results_invalid.json kept unless
# VPD_RERUN_INVALID=1); the probe skips finished (adapter, set) files and resumes a cut set at the next demo.  A
# failed step does not stop the others: every block of the step list runs through `part` (an `||` context, so
# `set -e` cannot fire; e.g. a failed init-check records train_483A:no_init and the floors, raw evals and probe still
# run).  A 483A/483B trained with other code (adapter-ok exit 3) stops the job (nothing overwritten).  Exit 1 if any
# step failed; RUN/DONE only when every step passed.
# Threads: training 2 (--threads 2) + 3 loader workers; vLLM 4 (VLLMServer); 8 eval workers; TF probe 2 + 3 workers.
# Env: VPD_DRYRUN=1 prints the commands (writes only RUN/{steps.jsonl,node.txt,df_end.txt}); VPD_JOB_DEADLINE_S
#      (default 15720), VPD_MIN_STEP_S (60), VPD_MERGE_DIR (default /tmp or /dev/shm with >= 40 GB free).

ARM=${1:-}
RUN=${2:-${RUN:-}}
case $ARM in
  A|B) ;;
  *) echo "usage: wb_learn483.sh ARM RUN   (ARM: A B)" >&2; exit 2 ;;
esac
if [[ -z $RUN ]]; then
  echo "usage: wb_learn483.sh ARM RUN" >&2
  exit 2
fi
START=Q3B
export RUN START
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

DRY=${VPD_DRYRUN:-0}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-15720}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_MIN_STEP_S must be a positive integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))
WB=$VPD_OUT/walkerbench
X=${VPD_X483_ROOT:-$WB/runs/explore/x483/$START}
P=${VPD_X483_PILOT:-$WB/runs/pilot/$START}
case "$(realpath -m -- "$X")/" in
  "$(realpath -m -- "$WB/runs/pilot")"/*)
    echo "[wb_learn483] refusing X=$X: under runs/pilot (the pilot's outputs are read only)" >&2; exit 2 ;;
esac
PAD=$P/adapters
INIT=$PAD/0W
XAD=$X/adapters
XEV=$X/evals
TF=$X/tf
DATA=$WB/tiers/seed1001/demos_all.jsonl
S2IN50=$WB/split/s2in50
L483=(-m vpd.walkerbench.learn483)
FAILS=()

mkdir -p "$RUN"
cd "$VPD_WORKTREE"
if [[ $DRY != 1 ]]; then
  mkdir -p "$XAD" "$XEV" "$TF"
  vpd_chown_trap "$RUN" "$X"
  # setup on the login node creates these as the login user; should they be created here (as root), hand them back
  # (not recursive for the parents: the other arm may be live)
  if [[ $(id -u) -eq 0 ]]; then
    vpd_on_exit "chown ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q ' "$WB/runs/explore" "$WB/runs/explore/x483")|| true"
    vpd_on_exit "[[ -d $(printf '%q' "$S2IN50") ]] && chown -R ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q' "$S2IN50") || true"
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

# timed NAME CMD...: vpd_step NAME under the soft deadline; NAME:no_time when fewer than MIN_STEP_S seconds are left
timed() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t < MIN_STEP_S)); then
    echo "[wb_learn483] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@"
}

# part FUNC ARGS...: one block of the step list (wb_variants.sh's helper).  FUNC runs in an `||` context, so neither
# its non-zero return nor a failing command inside it can end the payload under `set -e`; every block records its own
# failures (a vpd_step line with rc != 0 and a FAILS entry), and one that returns non-zero without having recorded any
# gets FUNC:failed.
part() {
  local n=${#FAILS[@]} rc=0
  "$@" || rc=$?
  if ((rc != 0 && ${#FAILS[@]} == n)); then
    vpd_step "$1:failed" false || true
    FAILS+=("$1")
  fi
  return 0
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
    *) echo "[wb_learn483] refusing to remove $d" >&2; return 1 ;;
  esac
}

# adapter_state DIR INIT_DIR -> 0 complete (current code, started from INIT/adapter.pt as now), 1 retrain, 3 other code
adapter_state() {
  [[ $DRY == 1 ]] && { echo "[dry] $PY_EVAL -m vpd.walkerbench.pilot adapter-ok $1 --init $2"; return 1; }
  local rc=0
  "$PY_EVAL" -m vpd.walkerbench.pilot adapter-ok "$1" --init "$2" || rc=$?
  return "$rc"
}

declare -A OK=()  # adapters usable for merges / the probe

check_pilot_adapter() {  # TIER [INIT]: the pilot's adapter, read only (never retrained here)
  local t=$1 rc=0 args=("$PAD/$1")
  [[ -n ${2:-} ]] && args+=(--init "$2")
  if [[ $DRY == 1 ]]; then
    echo "[dry] $PY_EVAL -m vpd.walkerbench.pilot adapter-ok ${args[*]}"
    OK[$t]=1
    return 0
  fi
  "$PY_EVAL" -m vpd.walkerbench.pilot adapter-ok "${args[@]}" || rc=$?
  if ((rc == 0)); then
    OK[$t]=1
    vpd_step "pilot_adapter_$t:ok" true
  else
    echo "[wb_learn483] pilot adapter $PAD/$t not usable (adapter-ok $rc): its evals / probe are skipped" >&2
    vpd_step "pilot_adapter_$t:not_ok" false || true
    FAILS+=("pilot_adapter_$t")
  fi
}

INIT_OK=0
init_check() {
  if vpd_step init_check x "$PY_EVAL" "${L483[@]}" init-check --init "$INIT"; then
    INIT_OK=1
  else
    echo "[wb_learn483] init-check failed: $INIT is not the pilot's 0W or the trainer changed; no training" >&2
    FAILS+=(init_check)
  fi
}

train() {  # NAME EPOCHS
  local name=$1 epochs=$2 out=$XAD/$1 st=0
  if ((INIT_OK != 1)); then
    vpd_step "train_$name:no_init" false || true
    FAILS+=("train_$name")
    return 1
  fi
  adapter_state "$out" "$INIT" || st=$?
  if ((st == 0)); then
    vpd_step "train_$name:skip" true
    OK[$name]=1
    return 0
  fi
  if ((st == 3)); then
    echo "[wb_learn483] $out was trained with other code: stopping (nothing overwritten)" >&2
    vpd_step "train_$name:code_mismatch" false || true
    exit 3
  fi
  if timed "train_$name" "$PY_TRAIN" vpd/train/n2_train_vpd.py --start "$START" --data "$DATA" --init-adapter "$INIT" \
      --out "$out" --epochs "$epochs" --seed 1001 --phase tier483 --threads "$VPD_THREADS" --num-workers 3; then
    OK[$name]=1
  else
    FAILS+=("train_$name")
  fi
}

# SPEC = SET:RES:MODE[:VARIANT]  (VARIANT: blind | rep2)
ename() {  # MODEL SPEC
  local tset res mode variant
  IFS=: read -r tset res mode variant <<< "$2"
  echo "explore__${START}__$1__${tset}__${res}__${mode}${variant:+__$variant}"
}
etag() {  # MODEL SPEC -> step name
  local tset res mode variant
  IFS=: read -r tset res mode variant <<< "$2"
  echo "eval_$1_${tset}_${res}_${mode}${variant:+_$variant}"
}

# eval_state DIR [ADAPTER_DIR] -> 0 results.json present (merged: it served ADAPTER_DIR/adapter.pt as it is now),
# 1 none, 3 stale (served other weights: flagged, re-run only with VPD_RERUN_STALE=1)
eval_state() {
  local dir=$1 rc=0
  [[ -f $dir/results.json ]] || return 1
  [[ -z ${2:-} ]] && return 0
  "$PY_EVAL" -m vpd.walkerbench.pilot eval-ok "$dir" --adapter "$2" || rc=$?
  ((rc == 0 || rc == 3)) || rc=3
  return "$rc"
}

evalrun() {  # MODEL SPEC [MODEL_DIR ADAPTER_DIR]
  local model=$1 spec=$2 md=${3:-} ad=${4:-} st=0
  local tset res mode variant
  IFS=: read -r tset res mode variant <<< "$spec"
  local name tag size=640x400 workers=8 mt=512 extra=()
  name=$(ename "$model" "$spec")
  tag=$(etag "$model" "$spec")
  [[ $res == 1280 ]] && size=1280x800
  if [[ $mode == off ]]; then
    mt=1024
    [[ $res == 1280 ]] && mt=2048
  fi
  case $variant in
    "") ;;
    blind) extra+=(--blind) ;;
    rep2) workers=4 ;;
    *) echo "[wb_learn483] unknown variant $variant" >&2; FAILS+=("$tag"); return 1 ;;
  esac
  local dir=$XEV/$name
  if [[ $model == raw ]]; then
    eval_state "$dir" || st=$?
  else
    eval_state "$dir" "$ad" || st=$?
  fi
  if ((st == 0)); then
    vpd_step "$tag:skip" true
    return 0
  fi
  local args=("${L483[@]}" eval --model "$START" --task-set "$tset" --render-size "$size" --constrained "$mode"
              --max-tokens "$mt" --history-turns 0 --workers "$workers" --temperature 0 --run-name "$name"
              --out-root "$XEV")
  if ((st == 3)); then
    if [[ ${VPD_RERUN_STALE:-0} != 1 ]]; then
      echo "[wb_learn483] $dir served other weights than $ad/adapter.pt; VPD_RERUN_STALE=1 re-runs it" >&2
      vpd_step "$tag:stale_kept" false || true
      FAILS+=("$tag")
      return 1
    fi
    args+=(--overwrite)
  elif [[ -f $dir/results_invalid.json ]]; then
    if [[ ${VPD_RERUN_INVALID:-0} != 1 ]]; then
      echo "[wb_learn483] $dir has results_invalid.json; VPD_RERUN_INVALID=1 re-runs it" >&2
      vpd_step "$tag:invalid_kept" false || true
      FAILS+=("$tag")
      return 1
    fi
    args+=(--overwrite)
  fi
  args+=(${extra[@]+"${extra[@]}"})
  [[ -n $md ]] && args+=(--model-dir "$md" --adapter-meta "$ad/meta.json")
  if ! timed "$tag" "$PY_Q3" "${args[@]}"; then
    FAILS+=("$tag")
    return 1
  fi
}

parity_needed() {  # the merge parity of 483A is missing or for another adapter.pt
  [[ $DRY == 1 ]] && return 0
  local rc=0
  "$PY_EVAL" "${L483[@]}" parity-ok --file "$TF/merge_parity.json" --adapter "$XAD/483A" || rc=$?
  ((rc != 0))
}

merged_evals() {  # MODEL ADAPTER_DIR SPEC...: merge, the evals on the merged folder (483A: + parity), remove it
  local model=$1 ad=$2; shift 2
  local spec st todo=() parity=0
  for spec in "$@"; do  # fresh evals are skipped; stale ones are flagged here unless VPD_RERUN_STALE=1
    st=0
    eval_state "$XEV/$(ename "$model" "$spec")" "$ad" || st=$?
    if ((st == 0)); then
      vpd_step "$(etag "$model" "$spec"):skip" true
    elif ((st == 3)) && [[ ${VPD_RERUN_STALE:-0} != 1 ]]; then
      echo "[wb_learn483] eval $model $spec served other weights than $ad/adapter.pt; VPD_RERUN_STALE=1 re-runs it" >&2
      vpd_step "$(etag "$model" "$spec"):stale_kept" false || true
      FAILS+=("$(etag "$model" "$spec")")
    else
      todo+=("$spec")
    fi
  done
  if [[ $model == 483A ]] && parity_needed; then
    parity=1
  fi
  if ((${#todo[@]} == 0 && parity == 0)); then
    vpd_step "merge_$model:skip" true
    return 0
  fi
  if [[ -z ${OK[$model]:-} ]]; then
    echo "[wb_learn483] adapter $model not available: its evals are skipped" >&2
    vpd_step "merge_$model:no_adapter" false || true
    FAILS+=("merge_$model")
    return 1
  fi
  if [[ -z $VPD_MERGE_DIR ]]; then
    echo "[wb_learn483] no job-local disk with 40 GB free: merge_$model skipped" >&2
    vpd_step "merge_$model:no_disk" false || true
    FAILS+=("merge_$model")
    return 1
  fi
  local local_dir=$VPD_MERGE_DIR/x483__${START}__$model
  safe_rm_merged "$local_dir" || true
  if timed "merge_$model" "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$ad" --out "$local_dir"; then
    for spec in ${todo[@]+"${todo[@]}"}; do
      evalrun "$model" "$spec" "$local_dir" "$ad" || true
    done
    if ((parity)); then
      timed "parity_$model" "$PY_TRAIN" "${L483[@]}" tf-parity --merged "$local_dir" --adapter "$ad" \
        --out "$TF/merge_parity.json" --threads "$VPD_THREADS" --num-workers 3 || FAILS+=("parity_$model")
    fi
  else
    FAILS+=("merge_$model")
  fi
  vpd_step "rm_merged_$model" safe_rm_merged "$local_dir" || FAILS+=("rm_merged_$model")
}

floors() {  # SET (CPU, rendering stubbed; skipped when X/floors_SET.json exists)
  local set=$1 out=$X/floors_$1.json
  if [[ -f $out ]]; then
    vpd_step "floors_$set:skip" true
    return 0
  fi
  timed "floors_$set" "$PY_EVAL" "${L483[@]}" floors --task-set "$set" --out "$out" --logs "$X/floors_logs/$set" \
    --work "$RUN/floors_work" || FAILS+=("floors_$set")
}

tf_step() {  # NAME ADAPTER_SPEC... (NAME=DIR, raw=)
  local name=$1; shift
  if (($# == 0)); then
    vpd_step "$name:no_adapter" false || true
    FAILS+=("$name")
    return 1
  fi
  local a args=("${L483[@]}" tf --out "$TF" --threads "$VPD_THREADS" --num-workers 3)
  for a in "$@"; do
    args+=(--adapter "$a")
  done
  timed "$name" "$PY_TRAIN" "${args[@]}" || { FAILS+=("$name"); return 1; }
}

echo "[wb_learn483] start $(date -u +%FT%TZ) ARM=$ARM START=$START RUN=$RUN X=$X P=$P dry=$DRY" \
     "deadline=${DEADLINE_S}s (EXPLORATORY: numbers only, no decision rule)"
vpd_node_facts "$RUN/node.txt"
for f in "$DATA" "$INIT/adapter.pt"; do
  if [[ ! -f $f ]]; then
    echo "[wb_learn483] missing $f" >&2
    [[ $DRY == 1 ]] || exit 2
  fi
done
export VPD_MERGE_DIR=${VPD_MERGE_DIR:-$(pick_merge_dir)}
echo "[wb_learn483] VPD_MERGE_DIR=${VPD_MERGE_DIR:-<none with 40 GB free>}"
if [[ -n $VPD_MERGE_DIR && $DRY != 1 ]]; then  # a timeout must not leave merged weights on the node
  vpd_on_exit "for d in '$VPD_MERGE_DIR/x483__${START}__483A' '$VPD_MERGE_DIR/x483__${START}__483B' '$VPD_MERGE_DIR/x483__${START}__200' '$VPD_MERGE_DIR/x483__${START}__50'; do case \$d in /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) rm -rf -- \"\$d\";; esac; done"
fi

vpd_step setup x "$PY_EVAL" "${L483[@]}" setup --x "$X" || FAILS+=(setup)
if [[ $ARM == A ]]; then
  # 1 floors on the two new sets (s2test's are baselines/baselines_s2test.json)
  part floors s2val
  part floors s2in50
  # 2 the user-requested training
  part init_check
  part check_pilot_adapter 0W
  part check_pilot_adapter 50 "$INIT"
  part check_pilot_adapter 200 "$INIT"
  part train 483A auto
  # 3 483A closed loop (X1 first), then the merge parity
  part merged_evals 483A "$XAD/483A" s2test:640:empty s2test:640:empty:blind s2val:640:empty s2in50:640:empty \
    s2test:640:empty:rep2
  # 4 teacher-forced probe
  TFA=()
  [[ -n ${OK[483A]:-} ]] && TFA+=("483A=$XAD/483A")
  TFA+=("raw=")
  [[ -n ${OK[0W]:-} ]] && TFA+=("0W=$PAD/0W")
  [[ -n ${OK[50]:-} ]] && TFA+=("50=$PAD/50")
  [[ -n ${OK[200]:-} ]] && TFA+=("200=$PAD/200")
  part tf_step tf_A "${TFA[@]}"
  # 5-6 the pilot's tiers on the new sets
  part merged_evals 200 "$PAD/200" s2val:640:empty s2in50:640:empty
  part merged_evals 50 "$PAD/50" s2in50:640:empty
  # 7 raw
  for spec in s2val:640:empty s2in50:640:empty s2test:640:off s2test:1280:off; do
    part evalrun raw "$spec"
  done
else
  part init_check
  part train 483B 10
  part merged_evals 483B "$XAD/483B" s2test:640:empty s2val:640:empty s2in50:640:empty
  if [[ -n ${OK[483B]:-} ]]; then
    part tf_step tf_B "483B=$XAD/483B"
  else
    vpd_step "tf_B:no_adapter" false || true
    FAILS+=(tf_B)
  fi
fi

df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]} == 0)); then
  if [[ $DRY == 1 ]]; then
    echo "[dry] date -u > $RUN/DONE"
  else
    date -u +%FT%TZ > "$RUN/DONE"
  fi
fi
echo "[wb_learn483] done $(date -u +%FT%TZ) ARM=$ARM; failed steps: ${FAILS[*]:-none}"
((${#FAILS[@]} == 0))
