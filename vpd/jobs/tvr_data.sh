#!/usr/bin/env bash
# sco payload of job vpd-tvr-data (CONTRACT_wave2 7.1; DECISIONS G3, G10(a)-(f)): TVR dev split, ProcTHOR houses, the
# frozen per-task start-pose reachable sets, the lavapipe replay of the 1,600 public action sequences, demos, tiers,
# BFS tables and blind floors.  Non-root, one card (the GPU idles: CPU rendering, O16), gate tvr, from vpd/tvr/pilot_tvr.py:
#     env VPD_JOB_DEADLINE_S=8520 timeout -k 60 8700 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/tvr_data.sh RUN
# (vpd.sco prepare rewrites the worktree prefix to RUN/code: the job runs from a code snapshot, DECISIONS G1.)
# Steps (vpd_step -> RUN/steps.jsonl; every block through `part`, so a failure never ends the payload under set -e):
#   0 node          vpd_node_facts -> RUN/node.txt
#   1 guard         pilot_tvr guard tvr-data (missing(tvr-data) == [], REPO file shas, badge font sha, probe verdict)
#   2 dev_select    tasks dev-select -> OUT/tvr/split/dev200/selection.json
#   3 houses        reach houses (sft1600, eval500, dev200)
#   4 reach_dump    reach dump, E_rep x L THOR workers (render.replay_envs x render.lp_threads), --deadline-s left-240
#     reap_reach_dump
#   5 replay        run_tvr replay sft1600 -> OUT/tvr/replay/sft1600_lvp, PNGs -> OUT/tvr/demos/sft1600_lvp/images
#     reap_replay   (6)
#   7 demos_build   8 tiers   9 bfs   10 floors
#  11 expect        pilot_tvr expect tvr-data --run RUN -> RUN/expect.json; RUN/DONE only if every step passed
# Interpreters: PY_Q3 for the tools (THOR workers: the TVR env python, started by vpd.tvr.thor_worker), PY_EVAL for
# pilot_tvr.  Threads: VPD_THREADS=2 for CPU steps; THOR workers 1 OMP thread + L lavapipe threads (lvp.worker_env).
# Idempotent: dev-select keeps an identical file, houses rewrite identical bytes, reach dump skips complete scenes,
# replay --resume reports a complete replay as complete (rc 0), demos / tiers are no-ops on an identical DONE output,
# so a rerun after a cut (or after success) reaches DONE.  A guard failure skips every later step (expect still runs).
# Deadlines: soft deadline VPD_JOB_DEADLINE_S (default 8520); every command under `timeout -k 30 <seconds left>`;
# THOR commands also get --deadline-s <seconds left - 240>; a step with < VPD_MIN_STEP_S (60) left is NAME:no_time.
# Env: VPD_DRYRUN=1 prints the commands (writes only RUN/{steps.jsonl,node.txt,df_end.txt}); test hooks honoured ONLY
#      with VPD_DRYRUN=1: VPD_TVR_OUT (instead of OUT/tvr).  VPD_TVR_PY: the TVR interpreter (tests).

RUN=${1:-}
if [[ -z $RUN || $# -ne 1 ]]; then
  echo "usage: tvr_data.sh RUN" >&2
  exit 2
fi
export RUN
export VPD_TVR_ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"  # before common_env.sh (lvp profile "envlib+orig")
export VPD_THREADS=${VPD_THREADS:-2}
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
unset ACTIVESPATIAL_USE_X11 ACTIVESPATIAL_X11_FORCE_VULKAN DISPLAY LP_NUM_THREADS
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

JOB=tvr-data
DRY=${VPD_DRYRUN:-0}
T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-8520}
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
HOUSES=$TVR/houses
REACH=$TVR/reach/v1
REPLAY=$TVR/replay/sft1600_lvp
DEMOS=$TVR/demos/sft1600_lvp
FAILS=()

mkdir -p "$RUN/logs"
cd "$VPD_WORKTREE"
if [[ $DRY != 1 ]]; then
  rm -f -- "$RUN/DONE"  # DONE describes this run only
  vpd_chown_trap "$RUN"   # no-op: this job runs non-root
fi

x() {  # run, or print under VPD_DRYRUN=1
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*"
  else
    "$@"
  fi
}

left() { echo $((DEADLINE - $(date +%s))); }

# part FUNC ARGS...: one block; its failure is recorded (a vpd_step line rc != 0 and a FAILS entry) and never ends the
# payload under set -e (it runs in an || context).  As wb_variants.sh.
part() {
  local n=${#FAILS[@]} rc=0
  "$@" || rc=$?
  if ((rc != 0 && ${#FAILS[@]} == n)); then
    vpd_step "$1:failed" false || true
    FAILS+=("$1")
  fi
  return 0
}

# timed NAME CMD...: vpd_step under the soft deadline; NAME:no_time when fewer than MIN_STEP_S seconds are left
timed() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t < MIN_STEP_S)); then
    echo "[tvr_data] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@"
}

# timed_thor NAME CMD...: as timed, plus --deadline-s <seconds left - 240> (the tool's own fast shutdown, 5.2.2)
timed_thor() {
  local name=$1; shift
  local t
  t=$(left)
  if ((t - THOR_RESERVE_S < MIN_STEP_S)); then
    echo "[tvr_data] $name skipped: ${t}s left before the soft deadline" >&2
    vpd_step "$name:no_time" false || true
    return 1
  fi
  vpd_step "$name" x timeout -k 30 "$t" "$@" --deadline-s $((t - THOR_RESERVE_S))
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
  if [[ $DRY == 1 ]]; then
    echo "[dry] reap_unity $1"
    return 0
  fi
  local pids k pat="/.ai2thor/releases/$BUILD/$BUILD"
  if ! command -v pgrep > /dev/null || ! command -v pkill > /dev/null; then
    echo "[tvr_data] $1: pgrep/pkill missing, cannot reap Unity players" | tee -a "$RUN/logs/reap.log"
    return 0
  fi
  pids=$(pgrep -f -- "$pat" | tr '\n' ' ' || true)
  [[ -n ${pids// /} ]] || return 0
  echo "[tvr_data] $(date -u +%TZ) $1: TERM lingering Unity player(s): $pids" | tee -a "$RUN/logs/reap.log"
  pkill -TERM -f -- "$pat" || true
  for ((k = 0; k < 50; k++)); do
    pgrep -f -- "$pat" > /dev/null || return 0
    sleep 0.1
  done
  echo "[tvr_data] $(date -u +%TZ) $1: KILL Unity player(s) still alive" | tee -a "$RUN/logs/reap.log"
  pkill -KILL -f -- "$pat" || true
  return 0
}

reap() { vpd_step "reap_$1" reap_unity "$1"; }

GUARD_OK=0
guard() {
  if vpd_step guard x "$PY_EVAL" -m vpd.tvr.pilot_tvr guard "$JOB" --run "$RUN"; then
    GUARD_OK=1
  else
    FAILS+=(guard)
    return 1
  fi
}

step() {  # NAME KIND CMD...: KIND plain|thor; skipped (NAME:no_guard) when the guard failed
  local name=$1 kind=$2; shift 2
  if ((GUARD_OK != 1)); then
    vpd_step "$name:no_guard" false || true
    FAILS+=("$name")
    return 1
  fi
  if [[ $kind == thor ]]; then
    timed_thor "$name" "$@" || { FAILS+=("$name"); return 1; }
  else
    timed "$name" "$@" || { FAILS+=("$name"); return 1; }
  fi
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

if [[ $DRY != 1 ]]; then
  trap 'echo "[tvr_data] TERM: reaping"; reap_unity term; exit 143' TERM
  vpd_on_exit "reap_unity exit"
fi

echo "[tvr_data] start $(date -u +%FT%TZ) RUN=$RUN TVR=$TVR uid=$(id -u) dry=$DRY deadline=${DEADLINE_S}s"
vpd_step node vpd_node_facts "$RUN/node.txt" || true
part guard
E_REP=? LP=? PROFILE=? BUILD=?
read -r E_REP LP PROFILE BUILD < <(cfg render.replay_envs render.lp_threads render.lvp_profile render.build) || true
echo "[tvr_data] replay/reach workers E=$E_REP L=$LP profile=$PROFILE"
part step dev_select plain "$PY_Q3" -B -m vpd.tvr.tasks dev-select --out "$TVR/split/dev200"
part step houses plain "$PY_Q3" -B -m vpd.tvr.reach houses --task-sets sft1600,eval500,dev200 --out "$HOUSES"
part step reach_dump thor "$PY_Q3" -B -m vpd.tvr.reach dump --task-sets sft1600,eval500,dev200 --out "$REACH" \
  --envs "$E_REP" --lp "$LP" --profile "$PROFILE" --houses "$HOUSES"
part reap reach_dump
part step replay thor "$PY_Q3" -B -m vpd.tvr.run_tvr replay --task-set sft1600 --public "$PUBLIC" --run-dir "$REPLAY" \
  --images-dir "$DEMOS/images" --envs "$E_REP" --lp "$LP" --profile "$PROFILE" --resume
part reap replay
part step demos_build plain "$PY_Q3" -B -m vpd.tvr.demos build --replay "$REPLAY" --images "$DEMOS/images" --out "$DEMOS"
part step tiers plain "$PY_Q3" -B -m vpd.tvr.tasks tiers --demos "$DEMOS/demos_all.jsonl" --out "$TVR/tiers/v1"
part step bfs plain "$PY_Q3" -B -m vpd.tvr.bfs table --task-sets eval500,dev200,sft1600 --reach "$REACH" --out "$TVR/bfs"
part step floors plain "$PY_Q3" -B -m vpd.tvr.floors --task-sets eval500,dev200 --reach "$REACH" --out "$TVR/floors"
part expect_list

df -h / /tmp /dev/shm > "$RUN/df_end.txt" 2>&1 || true
if ((${#FAILS[@]} == 0 && EXPECT_OK == 1)); then
  if [[ $DRY == 1 ]]; then
    echo "[dry] date -u > $RUN/DONE"
  else
    date -u +%FT%TZ > "$RUN/DONE"
  fi
fi
echo "[tvr_data] done $(date -u +%FT%TZ); failed steps: ${FAILS[*]:-none}"
((${#FAILS[@]} == 0))
