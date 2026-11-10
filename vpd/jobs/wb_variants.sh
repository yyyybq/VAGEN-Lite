#!/usr/bin/env bash
# EXPLORATORY (DECISIONS G4, G6; not part of the pre-registered pilot, no gate decided here): WalkerBench limited
# exploration, Q3B recipe variants on SEL = s2val + s2dev74.  See vpd/EXPLORATORY_WB_VARIANTS.md.
# sco payload, as_root (Q3B blobs are root-only), one H800:
#     env VPD_JOB_DEADLINE_S=<soft> timeout -k 120 <outer> \
#         bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_variants.sh VARIANT RUN [BASE]
# (vpd.sco prepare rewrites the worktree prefix to RUN/code: the job runs from a code snapshot.)
#   VARIANT V0  (job A, vpd-wb-v0v1): B1 single-demo rebuild FIRST (G6), setup, floors on s2sel, V0 = the current
#               recipe on the 409 exploration tasks (3 epochs from the pilot 0W), evals SEL with / without images,
#               then the V1 eval-only probes: V0 merged + --odometer line; raw Q3B --constrained off --max-tokens 1024
#               without / with --odometer line; raw constrained-empty.
#   VARIANT V1T (job B, vpd-wb-v1t): odometer line in every demo user turn; 3 epochs; SEL --odometer line with /
#               without images.
#   VARIANT V2  (job C, vpd-wb-v2): odometer THOUGHT in every demo reply; 3 epochs; SEL --constrained short
#               --thought-max-chars 32 with / without images.
#   VARIANT V5 BASE (BASE V0|V1T|V2: the incumbent): BASE's data + per-reply loss_weight; 3 epochs; BASE's eval mode.
#   VARIANT V4 BASE (BASE V0|V1T|V2|V5): BASE's data, --epochs 10; BASE's eval mode.
# Soft deadline = outer - 180 s, outer = floor(est_high * 3600) - 300 s (the pilot's rule); every step runs under
# `timeout -k 30 <seconds left>`; a step with < VPD_MIN_STEP_S (60) left is recorded as NAME:no_time and not started.
# Root X = OUT/walkerbench/runs/explore/wbv/Q3B (VPD_WBV_ROOT overrides: tests only); the pilot's Q3B 0W
# (P/adapters/0W, P = OUT/walkerbench/runs/pilot/Q3B; VPD_WBV_PILOT overrides: tests only) is read only.  Nothing is
# written under runs/pilot/ (X there is refused).  The variant writes only X/<VARIANT>/ (+ X/V1/, X/floors/, X/b1/
# and the B1 demos dir for V0; X/analysis/ for every job).
# Idempotent: data is written once (a different existing file is refused); an adapter is skipped when
# `variants adapter-ok` passes (exit 3 = trained with other code: not retrained and not evaluated, nothing
# overwritten); an eval is skipped when results.json exists (merged: `pilot eval-ok DIR --adapter DIR` = it served
# the adapter.pt on disk; stale evals are flagged and kept unless VPD_RERUN_STALE=1, results_invalid.json kept
# unless VPD_RERUN_INVALID=1).  B1: a B1 out dir without its shard manifest (a cut render; our own generated dir) is
# moved aside to <B1_OUT>.partial-<UTC> and rebuilt; a B1 out dir that is, holds or lies in the frozen
# demos/s2train_640x400 is refused.
# init-check: the pilot 0W (sha prefix, finished); V5 also a trainer that reads loss_weight; V5 / V4 also the current
# trainer code_sha256 == that of X/V0/adapter and X/BASE/adapter where they exist (exit 4 = other code: no training,
# step train_V:code_differs).
# A failed step never ends the payload: every block of the step list runs through `part` (an `||` context, so
# `set -e` cannot fire), each failure is a steps.jsonl line (rc != 0) and a FAILS entry, and the steps that do not
# need the failed output still run (the raw probes, evals of adapters that exist, analyze, expect).  Exit 3 if an
# adapter was trained with other code, else 1 if any step failed, else 0.  RUN/DONE (a stale one is removed at the
# start) only when every step passed, which includes `variants expect --check` (RUN/expect.txt, 0 missing).  The
# last steps are `variants analyze` (numbers only) and `variants expect --check`.
# Threads: training 2 (--threads) + 3 loader workers; vLLM 4 (VLLMServer); 8 eval workers; CPU steps VPD_THREADS.
# Env: VPD_DRYRUN=1 prints the commands (writes only RUN/{steps.jsonl,node.txt,df_end.txt}); VPD_JOB_DEADLINE_S,
#      VPD_MIN_STEP_S (60), VPD_MERGE_DIR (default /tmp or /dev/shm with >= 40 GB free).

usage() { echo "usage: wb_variants.sh VARIANT RUN [BASE]   (VARIANT V0 V1T V2; V5 BASE=V0|V1T|V2; V4 BASE=V0|V1T|V2|V5)" >&2; exit 2; }
VARIANT=${1:-}
RUN=${2:-${RUN:-}}
BASE=${3:-}
case $VARIANT in
  V0|V1T|V2) [[ -z $BASE ]] || usage ;;
  V5) [[ $BASE =~ ^(V0|V1T|V2)$ ]] || usage ;;
  V4) [[ $BASE =~ ^(V0|V1T|V2|V5)$ ]] || usage ;;
  *) usage ;;
esac
[[ -n $RUN ]] || usage
START=Q3B
export RUN START
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

case $VARIANT in  # default soft deadlines of the submit commands (est_high 2.5 / 1.8 / 4.5 GPU-h)
  V0) DEF_DEADLINE=8520 ;;
  V4) DEF_DEADLINE=15720 ;;
  *) DEF_DEADLINE=6000 ;;
esac
DRY=${VPD_DRYRUN:-0}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-$DEF_DEADLINE}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_MIN_STEP_S must be a positive integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))
WB=$VPD_OUT/walkerbench
X=${VPD_WBV_ROOT:-$WB/runs/explore/wbv/$START}
P=${VPD_WBV_PILOT:-$WB/runs/pilot/$START}
B1_OUT=${VPD_WBV_B1_OUT:-$WB/demos/s2train_640x400_B1_vis0578}
B1_OUT=${B1_OUT%/}
B1_STEM=vis_0578_target_20260127_1951_33
FROZEN_DEMOS_DIR=$WB/demos/s2train_640x400  # never written, moved or removed here (G6)
case "$(realpath -m -- "$X")/" in
  "$(realpath -m -- "$WB/runs/pilot")"/*)
    echo "[wb_variants] refusing X=$X: under runs/pilot (the pilot's outputs are read only)" >&2; exit 2 ;;
esac
INIT=$P/adapters/0W
VX=$X/$VARIANT
AD=$VX/adapter
DATA=$VX/data/train.jsonl
VAR=(-m vpd.walkerbench.variants)
BASEARG=()
[[ -n $BASE ]] && BASEARG=(--base "$BASE")
case $VARIANT in
  V4) EPOCHS=10 ;;
  *) EPOCHS=3 ;;
esac
FAILS=()

mkdir -p "$RUN"
cd "$VPD_WORKTREE"
if [[ $DRY != 1 ]]; then
  rm -f -- "$RUN/DONE"  # DONE describes this run only
  mkdir -p "$VX" "$X/analysis"
  OWN=("$RUN" "$VX" "$X/analysis")
  if [[ $VARIANT == V0 ]]; then
    mkdir -p "$X/V1" "$X/floors" "$X/b1"
    OWN+=("$X/V1" "$X/floors" "$X/b1")
  elif [[ $VARIANT == V5 || $VARIANT == V4 ]]; then
    mkdir -p "$X/floors"
    OWN+=("$X/floors")
  fi
  vpd_chown_trap "${OWN[@]}"
  if [[ $(id -u) -eq 0 ]]; then
    if [[ $VARIANT == V0 ]]; then  # the B1 demos dir exists only once demos.py ran (job A)
      vpd_on_exit "[[ -d $(printf '%q' "$B1_OUT") ]] && chown -R ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q' "$B1_OUT"); true"
    fi
    # parents and setup outputs created here as root are handed back (not recursive for X: other jobs may be live)
    vpd_on_exit "chown ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q ' "$WB/runs/explore" "$WB/runs/explore/wbv" "$X")|| true"
    vpd_on_exit "for d in $(printf '%q ' "$WB/split/s2dev74" "$WB/split/s2sel"); do [[ -d \$d ]] && chown -R ${VPD_JOB_UID}:${VPD_JOB_UID} \"\$d\"; done; true"
    vpd_on_exit "[[ -f $(printf '%q' "$X/EXPLORATORY.md") ]] && chown ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q' "$X/EXPLORATORY.md"); true"
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

# part FUNC ARGS...: one block of the step list.  FUNC runs in an `||` context, so neither its non-zero return nor a
# failing command inside it can end the payload under `set -e`; every block records its own failures (a vpd_step line
# with rc != 0 and a FAILS entry), and one that returns non-zero without having recorded any gets FUNC:failed.
part() {
  local n=${#FAILS[@]} rc=0
  "$@" || rc=$?
  if ((rc != 0 && ${#FAILS[@]} == n)); then
    vpd_step "$1:failed" false || true
    FAILS+=("$1")
  fi
  return 0
}

same_or_nested() {  # A B: true when A and B are one directory or one holds the other (realpath -m)
  local a b
  a=$(realpath -m -- "$1")
  b=$(realpath -m -- "$2")
  [[ $a == "$b" || $a == "$b"/* || $b == "$a"/* ]]
}

# timed NAME CMD...: vpd_step NAME under the soft deadline; NAME:no_time when fewer than MIN_STEP_S seconds are left
timed() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t < MIN_STEP_S)); then
    echo "[wb_variants] $name skipped: ${t}s left before the soft deadline" >&2
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
    *) echo "[wb_variants] refusing to remove $d" >&2; return 1 ;;
  esac
}

# ------------------------------------------------------------------ B1 (G6): the one dropped val demo, rebuilt
b1_rebuild() {
  if same_or_nested "$B1_OUT" "$FROZEN_DEMOS_DIR"; then
    echo "[wb_variants] refusing B1 out dir $B1_OUT: it is, holds or lies in the frozen $FROZEN_DEMOS_DIR" >&2
    vpd_step "b1:frozen_dir_refused" false || true
    FAILS+=(b1)
    return 1
  fi
  if [[ $DRY != 1 ]] && "$PY_EVAL" "${VAR[@]}" b1 --stage done --x "$X" --out-dir "$B1_OUT" >/dev/null; then
    vpd_step "b1:skip" true
    return 0
  fi
  vpd_step b1_precheck x "$PY_EVAL" "${VAR[@]}" b1 --stage precheck --x "$X" || { FAILS+=(b1_precheck); return 1; }
  vpd_step b1_before x "$PY_EVAL" "${VAR[@]}" b1 --stage before --x "$X" || { FAILS+=(b1_before); return 1; }
  if [[ -f $B1_OUT/shards/shard0of1.json ]]; then
    vpd_step "b1_render:skip" true
  else
    if [[ -e $B1_OUT || -L $B1_OUT ]]; then
      # a cut render left our own dir without its shard manifest (demos.py would refuse its images): move it aside
      local aside
      aside="$B1_OUT.partial-$(date -u +%Y%m%dT%H%M%SZ)"
      if [[ -e $aside || -L $aside ]]; then
        echo "[wb_variants] $aside exists: the partial B1 dir is not moved" >&2
        vpd_step "b1_partial_aside:exists" false || true
        FAILS+=(b1_partial_aside)
        return 1
      fi
      echo "[wb_variants] $B1_OUT has no shards/shard0of1.json (a cut render): moving it to $aside" >&2
      vpd_step b1_partial_aside x mv -T -- "$B1_OUT" "$aside" || { FAILS+=(b1_partial_aside); return 1; }
      if [[ $DRY != 1 && $(id -u) -eq 0 ]]; then
        vpd_on_exit "chown -R ${VPD_JOB_UID}:${VPD_JOB_UID} $(printf '%q' "$aside") || true"
      fi
    fi
    timed b1_render "$PY_EVAL" -m vpd.walkerbench.demos --stems "$B1_STEM" --out "$B1_OUT" --work "$RUN/b1_work" \
      --strict || { FAILS+=(b1_render); return 1; }
  fi
  timed b1_merge "$PY_EVAL" -m vpd.walkerbench.demos --merge --allow-partial --out "$B1_OUT" \
    || { FAILS+=(b1_merge); return 1; }
  vpd_step b1_after x "$PY_EVAL" "${VAR[@]}" b1 --stage after --x "$X" --out-dir "$B1_OUT" || FAILS+=(b1_after)
}

floors() {  # STOP0 / FWD / ODO / RM on s2sel (CPU, rendering stubbed); skipped when X/floors/floors_s2sel.json exists
  local out=$X/floors/floors_s2sel.json
  if [[ -f $out ]]; then
    vpd_step "floors_s2sel:skip" true
    return 0
  fi
  timed floors_s2sel "$PY_EVAL" "${VAR[@]}" floors --out "$out" --logs "$X/floors/logs" --work "$RUN/floors_work" \
    || FAILS+=(floors_s2sel)
}

INIT_OK=0
INIT_RC=0
init_check() {  # exit 0 ok; 4 V5 / V4: the current trainer code differs from V0's / BASE's adapter; else 1
  local extra=() rc=0
  if [[ $VARIANT == V5 || ( $VARIANT == V4 && $BASE == V5 ) ]]; then
    extra=(--need-loss-weight)
  fi
  vpd_step init_check x "$PY_EVAL" "${VAR[@]}" init-check --init "$INIT" --variant "$VARIANT" \
    ${BASEARG[@]+"${BASEARG[@]}"} --x "$X" ${extra[@]+"${extra[@]}"} || rc=$?
  INIT_RC=$rc
  if ((rc == 0)); then
    INIT_OK=1
    return 0
  fi
  if ((rc == 4)); then
    echo "[wb_variants] init-check refused $VARIANT: the current trainer code_sha256 differs from the one X/V0 (or" \
         "X/$BASE) was trained with, so $VARIANT would not be comparable; no training (see its problems above)" >&2
  else
    echo "[wb_variants] init-check failed: $INIT is not the pilot's 0W (or V5's trainer lacks loss_weight); no training" >&2
  fi
  FAILS+=(init_check)
  return 1
}

# adapter_state -> 0 complete, 1 (re)train, 3 trained with other code
adapter_state() {
  [[ $DRY == 1 ]] && { echo "[dry] $PY_EVAL ${VAR[*]} adapter-ok --variant $VARIANT ${BASE:+--base $BASE} --x $X --init $INIT"; return 1; }
  local rc=0
  "$PY_EVAL" "${VAR[@]}" adapter-ok --variant "$VARIANT" ${BASEARG[@]+"${BASEARG[@]}"} --x "$X" --init "$INIT" || rc=$?
  return "$rc"
}

TRAIN_OK=0
CODE_MISMATCH=0
train() {
  local st=0
  if ((INIT_OK != 1)); then
    if ((INIT_RC == 4)); then
      vpd_step "train_$VARIANT:code_differs" false || true
    else
      vpd_step "train_$VARIANT:no_init" false || true
    fi
    FAILS+=("train_$VARIANT")
    return 1
  fi
  if [[ $DRY != 1 && ! -f $DATA ]]; then
    vpd_step "train_$VARIANT:no_data" false || true
    FAILS+=("train_$VARIANT")
    return 1
  fi
  adapter_state || st=$?
  if ((st == 0)); then
    vpd_step "train_$VARIANT:skip" true
    TRAIN_OK=1
    return 0
  fi
  if ((st == 3)); then
    echo "[wb_variants] $AD was trained with other code: not retrained and not evaluated (nothing overwritten);" \
         "the independent steps still run and the job exits 3" >&2
    vpd_step "train_$VARIANT:code_mismatch" false || true
    FAILS+=("train_$VARIANT")
    CODE_MISMATCH=1
    return 1
  fi
  if timed "train_$VARIANT" "$PY_TRAIN" vpd/train/n2_train_vpd.py --start "$START" --data "$DATA" --init-adapter "$INIT" \
      --out "$AD" --epochs "$EPOCHS" --seed 1001 --phase "wbv_$VARIANT" --threads "$VPD_THREADS" --num-workers 3; then
    TRAIN_OK=1
  else
    FAILS+=("train_$VARIANT")
  fi
}

# SPEC = OWNER:SET:CONSTRAINED:ODOMETER[:blind]  (names as variants.eval_name)
ename() {  # MODEL SPEC
  local owner tset cm odo blind suf=""
  IFS=: read -r owner tset cm odo blind <<< "$2"
  if [[ $odo == line ]]; then suf=__odo; fi
  echo "wbv__${START}__$1__${tset}__640__${cm}${suf}${blind:+__$blind}"
}
edir() {  # MODEL SPEC
  local owner rest
  IFS=: read -r owner rest <<< "$2"
  echo "$X/$owner/evals/$(ename "$1" "$2")"
}
etag() {  # MODEL SPEC -> step name
  local owner tset cm odo blind suf=""
  IFS=: read -r owner tset cm odo blind <<< "$2"
  if [[ $odo == line ]]; then suf=_odo; fi
  echo "eval_${owner}_$1_${tset}_${cm}${suf}${blind:+_$blind}"
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
  local owner tset cm odo blind
  IFS=: read -r owner tset cm odo blind <<< "$spec"
  local name tag dir mt=512 extra=()
  name=$(ename "$model" "$spec")
  tag=$(etag "$model" "$spec")
  dir=$(edir "$model" "$spec")
  [[ $cm == off ]] && mt=1024
  [[ $cm == short ]] && extra+=(--thought-max-chars 32)
  case $blind in
    "") ;;
    blind) extra+=(--blind) ;;
    *) echo "[wb_variants] unknown eval flag $blind" >&2; FAILS+=("$tag"); return 1 ;;
  esac
  if [[ $model == raw ]]; then
    eval_state "$dir" || st=$?
  else
    eval_state "$dir" "$ad" || st=$?
  fi
  if ((st == 0)); then
    vpd_step "$tag:skip" true
    return 0
  fi
  local args=("${VAR[@]}" eval --odometer "$odo" --model "$START" --task-set "$tset" --render-size 640x400
              --constrained "$cm" --max-tokens "$mt" --history-turns 0 --workers 8 --temperature 0 --run-name "$name"
              --out-root "$X/$owner/evals")
  if ((st == 3)); then
    if [[ ${VPD_RERUN_STALE:-0} != 1 ]]; then
      echo "[wb_variants] $dir served other weights than $ad/adapter.pt; VPD_RERUN_STALE=1 re-runs it" >&2
      vpd_step "$tag:stale_kept" false || true
      FAILS+=("$tag")
      return 1
    fi
    args+=(--overwrite)
  elif [[ -f $dir/results_invalid.json ]]; then
    if [[ ${VPD_RERUN_INVALID:-0} != 1 ]]; then
      echo "[wb_variants] $dir has results_invalid.json; VPD_RERUN_INVALID=1 re-runs it" >&2
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

merged_evals() {  # SPEC...: merge X/<VARIANT>/adapter, the evals on the merged folder, remove it
  local spec st todo=()
  for spec in "$@"; do  # fresh evals are skipped; stale ones are flagged here unless VPD_RERUN_STALE=1
    st=0
    eval_state "$(edir "$VARIANT" "$spec")" "$AD" || st=$?
    if ((st == 0)); then
      vpd_step "$(etag "$VARIANT" "$spec"):skip" true
    elif ((st == 3)) && [[ ${VPD_RERUN_STALE:-0} != 1 ]]; then
      echo "[wb_variants] eval $spec served other weights than $AD/adapter.pt; VPD_RERUN_STALE=1 re-runs it" >&2
      vpd_step "$(etag "$VARIANT" "$spec"):stale_kept" false || true
      FAILS+=("$(etag "$VARIANT" "$spec")")
    else
      todo+=("$spec")
    fi
  done
  if ((${#todo[@]} == 0)); then
    vpd_step "merge_$VARIANT:skip" true
    return 0
  fi
  if ((TRAIN_OK != 1)); then
    echo "[wb_variants] adapter $VARIANT not available: its evals are skipped" >&2
    vpd_step "merge_$VARIANT:no_adapter" false || true
    FAILS+=("merge_$VARIANT")
    return 1
  fi
  if [[ -z $VPD_MERGE_DIR ]]; then
    echo "[wb_variants] no job-local disk with 40 GB free: merge_$VARIANT skipped" >&2
    vpd_step "merge_$VARIANT:no_disk" false || true
    FAILS+=("merge_$VARIANT")
    return 1
  fi
  local local_dir=$VPD_MERGE_DIR/wbv__${START}__$VARIANT
  safe_rm_merged "$local_dir" || true
  if timed "merge_$VARIANT" "$PY_Q3" vpd/train/merge.py --start "$START" --adapter "$AD" --out "$local_dir"; then
    for spec in ${todo[@]+"${todo[@]}"}; do
      evalrun "$VARIANT" "$spec" "$local_dir" "$AD" || true
    done
  else
    FAILS+=("merge_$VARIANT")
  fi
  vpd_step "rm_merged_$VARIANT" safe_rm_merged "$local_dir" || FAILS+=("rm_merged_$VARIANT")
}

variant_evals() {  # 4 the variant's own evals (mode from `variants eval-mode`: one source with the data manifest)
  local mode cmode odo specs
  if ! mode=$("$PY_EVAL" "${VAR[@]}" eval-mode --variant "$VARIANT" ${BASEARG[@]+"${BASEARG[@]}"} --x "$X"); then
    echo "[wb_variants] eval-mode failed for $VARIANT ${BASE:-}" >&2
    vpd_step "eval_mode:failed" false || true
    FAILS+=(eval_mode)
    return 1
  fi
  read -r cmode odo <<< "${mode:-x x}"
  if ! [[ $cmode =~ ^(empty|short|off)$ && $odo =~ ^(off|line)$ ]]; then
    echo "[wb_variants] eval-mode printed '$mode' for $VARIANT ${BASE:-}: no evals" >&2
    vpd_step "eval_mode:invalid" false || true
    FAILS+=(eval_mode)
    return 1
  fi
  specs=("$VARIANT:s2sel:$cmode:$odo" "$VARIANT:s2sel:$cmode:$odo:blind")
  if [[ $VARIANT == V0 ]]; then
    specs+=("V1:s2sel:empty:line")  # V1 probe: V0 merged + the eval-time odometer line
  fi
  merged_evals "${specs[@]}"
}

raw_probes() {  # 5 V1 raw probes (job A): no adapter needed, so they run whatever happened to the training
  local spec rc=0
  for spec in V1:s2sel:off:off V1:s2sel:off:line V1:s2sel:empty:off; do
    evalrun raw "$spec" || rc=1
  done
  return "$rc"
}

EXPECT_OK=0
expect_list() {  # 6b RUN/expect.txt; EXPECT_OK=1 only when nothing is missing
  if [[ $DRY == 1 ]]; then
    echo "[dry] $PY_EVAL ${VAR[*]} expect --variant $VARIANT ${BASE:+--base $BASE} --x $X --b1-out $B1_OUT --check > $RUN/expect.txt"
    EXPECT_OK=1
    return 0
  fi
  if vpd_step expect bash -c "$(printf '%q ' "$PY_EVAL" "${VAR[@]}" expect --variant "$VARIANT" \
      ${BASEARG[@]+"${BASEARG[@]}"} --x "$X" --b1-out "$B1_OUT" --check) > $(printf '%q' "$RUN/expect.txt")"; then
    EXPECT_OK=1
  else
    FAILS+=(expect)
    return 1
  fi
}

echo "[wb_variants] start $(date -u +%FT%TZ) VARIANT=$VARIANT BASE=${BASE:-none} START=$START RUN=$RUN X=$X P=$P" \
     "dry=$DRY deadline=${DEADLINE_S}s (EXPLORATORY: numbers only, no gate decided here)"
vpd_node_facts "$RUN/node.txt" || true
MISSING=()
for f in "$INIT/adapter.pt" "$WB/split/split_s2.json" "$WB/tiers/order_seed1001.json" "$WB/tiers/seed1001/demos_all.jsonl"; do
  [[ -f $f ]] || MISSING+=("$f")
done
if ((${#MISSING[@]})); then  # recorded, not fatal: the steps that need them fail on their own, the rest still run
  echo "[wb_variants] missing ${MISSING[*]}" >&2
  if [[ $DRY != 1 ]]; then
    vpd_step "inputs:missing" false || true
    FAILS+=(inputs)
  fi
fi
export VPD_MERGE_DIR=${VPD_MERGE_DIR:-$(pick_merge_dir)}
echo "[wb_variants] VPD_MERGE_DIR=${VPD_MERGE_DIR:-<none with 40 GB free>}"
if [[ -n $VPD_MERGE_DIR && $DRY != 1 ]]; then  # a timeout must not leave merged weights on the node
  vpd_on_exit "d='$VPD_MERGE_DIR/wbv__${START}__$VARIANT'; case \$d in /tmp/vpd_merged/?*|/dev/shm/vpd_merged/?*) rm -rf -- \"\$d\";; esac"
fi

# The step list.  Every block goes through `part`: a failure is recorded and the next block still runs.
# 0 the B1 single-demo rebuild, first (job A only; G6)
if [[ $VARIANT == V0 ]]; then
  part b1_rebuild
fi
# 1 task sets (verified; created if missing) and X dirs
vpd_step setup x "$PY_EVAL" "${VAR[@]}" setup --x "$X" || FAILS+=(setup)
# 2 floors on s2sel (job A; V5 / V4 only if still missing)
if [[ $VARIANT == V0 || $VARIANT == V5 || $VARIANT == V4 ]]; then
  part floors
fi
# 3 data, init check, training
vpd_step "data_$VARIANT" x "$PY_EVAL" "${VAR[@]}" data --variant "$VARIANT" ${BASEARG[@]+"${BASEARG[@]}"} --x "$X" \
  || FAILS+=("data_$VARIANT")
part init_check
part train
# 4 the variant's own evals (+ the V1 merged probe in job A); skipped and recorded when there is no adapter
part variant_evals
# 5 the raw probes (job A)
if [[ $VARIANT == V0 ]]; then
  part raw_probes
fi
# 6 numbers (no verdict) and the expect list
vpd_step analyze x "$PY_EVAL" "${VAR[@]}" analyze --x "$X" || FAILS+=(analyze)
part expect_list

df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]} == 0 && EXPECT_OK == 1)); then  # the whole step list ran and passed, expect included
  if [[ $DRY == 1 ]]; then
    echo "[dry] date -u > $RUN/DONE"
  else
    date -u +%FT%TZ > "$RUN/DONE"
  fi
fi
echo "[wb_variants] done $(date -u +%FT%TZ) VARIANT=$VARIANT; failed steps: ${FAILS[*]:-none}"
if ((CODE_MISMATCH)); then
  exit 3
fi
((${#FAILS[@]} == 0))
