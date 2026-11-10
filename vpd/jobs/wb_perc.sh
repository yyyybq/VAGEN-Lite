#!/usr/bin/env bash
# EXPLORATORY (DECISIONS G5; not part of any pre-registered pilot; numbers only, absolute per start, no between-start
# difference, no gate, no decision rule; counted separately from the WB 30 GPU-h).  WalkerBench perception zero-shot:
# raw Q3B, Q3S and E2 on angle / dis / height (task set perc294).  See vpd/walkerbench/perc.py.
# sco payload of vpd-wb-perc (as_root: the Qwen3 starts' blobs are root-only; one H800):
#     env VPD_JOB_DEADLINE_S=5280 timeout -k 120 5460 \
#         bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_perc.sh RUN
# (vpd.sco prepare rewrites the worktree prefix to RUN/code).  est 1.6 GPU-h -> outer = floor(1.6*3600) - 300 = 5460 s,
# soft deadline = outer - 180 = 5280 s (the pilot's rule, pilot.job_timeouts).  Every step runs under
# `timeout -k 30 <seconds left>` (an eval: seconds left minus VPD_TAIL_RESERVE_S = 120, kept for analyze / expect); a
# step with < VPD_MIN_STEP_S (60) left is recorded as NAME:no_time and not started.
# Expected wall time (INFERENCE from the Q3B s2test off run, 0.166 s/step with 8 workers): ~3,000-6,700 steps per
# start -> 10-25 min each plus ~3-5 min server start / sandbox / token audit, i.e. ~45-90 min for the three.
# Root R = OUT/walkerbench/runs/explore/perc (never under runs/pilot: refused).  Steps:
#   check_set        perc build --check: perc294 (built on the login node before prepare) matches the checkout
#   eval_<START>     for START in Q3B Q3S E2 (this order): perc eval --start START = run_vpd.main with IDENTICAL
#                    flags for every start: --task-set perc294 --render-size 640x400 --constrained off --lenient
#                    --max-tokens 1024 --history-turns 0 --temperature 0 --workers 8; run name
#                    explore__<START>__raw__perc294__640__off_lenient under R/<START>/evals
#   analyze          perc analyze -> R/PERC.json, R/REPORT_EXPLORATORY.md
#   expect           perc expect --check --out RUN/expect.txt: the output list (results.json, per_task.jsonl,
#                    manifest.json, explore.json per start; PERC.json; REPORT_EXPLORATORY.md) and per start state ok /
#                    protocol_ok / consistent
# Idempotent: an eval with results.json is skipped, so a resubmitted job (new RUN dir) continues after a cut; an
# earlier results_invalid.json is kept (step NAME:invalid_kept, failed) unless VPD_RERUN_INVALID=1 (then --overwrite).
# A failed eval does not stop the other starts.  Exit 1 if any step failed; RUN/DONE only when every step passed.
# Threads: vLLM 4 (VLLMServer), 8 eval worker threads, other processes VPD_THREADS=2 (common_env.sh).
# Env: VPD_DRYRUN=1 prints the commands (writes only RUN/{steps.jsonl,node.txt,df_end.txt}); VPD_JOB_DEADLINE_S
#      (default 5280), VPD_MIN_STEP_S (60), VPD_TAIL_RESERVE_S (120), VPD_RERUN_INVALID (0).  Test hook honoured
#      ONLY with VPD_DRYRUN=1:
#      VPD_PERC_ROOT (instead of R).

RUN=${1:-${RUN:-}}
if [[ -z $RUN ]]; then
  echo "usage: wb_perc.sh RUN" >&2
  exit 2
fi
export RUN
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
cd "$VPD_WORKTREE"

DRY=${VPD_DRYRUN:-0}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-5280}
MIN_STEP_S=${VPD_MIN_STEP_S:-60}
TAIL_S=${VPD_TAIL_RESERVE_S:-120}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
[[ $MIN_STEP_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_MIN_STEP_S must be a positive integer" >&2; exit 2; }
[[ $TAIL_S =~ ^[0-9]+$ ]] || { echo "VPD_TAIL_RESERVE_S must be a non-negative integer" >&2; exit 2; }
DEADLINE=$((T0 + DEADLINE_S))
WB=$VPD_OUT/walkerbench
R=$WB/runs/explore/perc
if [[ $DRY == 1 && -n ${VPD_PERC_ROOT:-} ]]; then
  R=$VPD_PERC_ROOT   # test hook (dry run only)
fi
case "$(realpath -m -- "$R")/" in
  "$(realpath -m -- "$WB/runs/pilot")"/*)
    echo "[wb_perc] refusing R=$R: under runs/pilot (the pilot's outputs are read only)" >&2; exit 2 ;;
esac
if [[ $DRY != 1 && $(id -u) -ne 0 ]]; then
  echo "[wb_perc] must run as root (prepare with --as-root): the Qwen3 starts' blobs are root-only" >&2
  exit 2
fi
STARTS=(Q3B Q3S E2)
PERC=(-m vpd.walkerbench.perc)
FAILS=()

mkdir -p "$RUN"
if [[ $DRY != 1 ]]; then
  for s in "${STARTS[@]}"; do
    mkdir -p "$R/$s/evals"
  done
  vpd_chown_trap "$RUN" "$R"
fi

x() {  # run, or print under VPD_DRYRUN=1
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*"
  else
    "$@"
  fi
}

left() { echo $((DEADLINE - $(date +%s))); }

# timed NAME CMD...: vpd_step NAME under the soft deadline minus ${RESERVE:-0} s (kept for the steps after it);
# NAME:no_time when fewer than MIN_STEP_S seconds are left
timed() {
  local name=$1; shift
  local t
  t=$(($(left) - ${RESERVE:-0}))
  if ((t < MIN_STEP_S)); then
    echo "[wb_perc] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@"
}

ename() { echo "explore__$1__raw__perc294__640__off_lenient"; }

evalrun() {  # START
  local s=$1 dir args
  dir=$R/$s/evals/$(ename "$s")
  if [[ -f $dir/results.json ]]; then
    vpd_step "eval_$s:skip" true
    return 0
  fi
  args=("${PERC[@]}" eval --start "$s" --root "$R")
  if [[ -f $dir/results_invalid.json ]]; then
    if [[ ${VPD_RERUN_INVALID:-0} != 1 ]]; then
      echo "[wb_perc] $dir has results_invalid.json; VPD_RERUN_INVALID=1 re-runs it" >&2
      vpd_step "eval_$s:invalid_kept" false || true
      FAILS+=("eval_$s")
      return 1
    fi
    args+=(--overwrite)
  fi
  if ! RESERVE=$TAIL_S timed "eval_$s" "$PY_Q3" "${args[@]}"; then
    FAILS+=("eval_$s")
    return 1
  fi
}

echo "[wb_perc] start $(date -u +%FT%TZ) RUN=$RUN R=$R worktree=$VPD_WORKTREE dry=$DRY deadline=${DEADLINE_S}s" \
     "(EXPLORATORY: numbers only, absolute per start, no between-start differences)"
vpd_node_facts "$RUN/node.txt"

if timed check_set "$PY_EVAL" "${PERC[@]}" build --check; then
  for s in "${STARTS[@]}"; do
    evalrun "$s" || true
  done
else
  echo "[wb_perc] perc294 check failed: no eval" >&2
  FAILS+=(check_set)
  for s in "${STARTS[@]}"; do
    vpd_step "eval_$s:no_set" false || true
  done
fi
timed analyze "$PY_EVAL" "${PERC[@]}" analyze --root "$R" || FAILS+=(analyze)
timed expect "$PY_EVAL" "${PERC[@]}" expect --root "$R" --check --out "$RUN/expect.txt" || FAILS+=(expect)

df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]} == 0)); then
  if [[ $DRY == 1 ]]; then
    echo "[dry] date -u > $RUN/DONE"
  else
    date -u +%FT%TZ > "$RUN/DONE"
  fi
fi
echo "[wb_perc] done $(date -u +%FT%TZ); failed steps: ${FAILS[*]:-none}"
((${#FAILS[@]} == 0))
