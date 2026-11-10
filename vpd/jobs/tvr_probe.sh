#!/usr/bin/env bash
# sco payload of job vpd-tvr-probe: the TVR lavapipe probe and the public-image comparison (DECISIONS G3).
# EXPLORATORY measurement for the TVR pilot design, never a gate.  Gate tvr; 1 x H800 (14 vCPU), non-root.
# NO GPU rendering: every AI2-THOR worker renders on lavapipe (vpd/tvr/lvp.py).  NO vLLM, no model, no download.
# Runs vpd/tvr/probe_lvp.py under the TVR env python (3.10, ai2thor 5.0.0); read its module doc for the phases.
#
#   RUN=$VPD_OUT/jobs/tvr-probe-$(date -u +%Y%m%d%H%M)
#   $PY_EVAL -m vpd.sco prepare --run $RUN --name vpd-tvr-probe --gate tvr --est-gph 0.6 -- \
#       env VPD_JOB_DEADLINE_S=1920 timeout -k 60 2040 \
#       bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/tvr_probe.sh $RUN
#   $PY_EVAL -m vpd.sco submit --run $RUN
# Wall clock: the outer timeout 2040 s + its 60 s kill window = 35 min, inside est-gph 0.6 (36 min); the soft job
# deadline VPD_JOB_DEADLINE_S (default 1920 s after start) ends the probe RESERVE_S (90 s) early so reap and
# expect still run.  probe_lvp guards every phase / grid cell / reach load by that deadline and skips what does
# not fit (recorded as skipped: no_time), so the job always writes its summary.
#
# Steps (vpd_step -> RUN/steps.jsonl; a failed step never stops the later ones):
#   node    vpd_node_facts -> RUN/node.txt
#   probe   probe_lvp run --out RUN/out --deadline-s <left>  (log RUN/out/logs/probe.log)
#   reap    kill lingering Unity players of this build (pattern UNITY_PAT); RUN/out/logs/reap.log
#   expect  probe_lvp expect --out RUN/out -> RUN/out/expect.json; rc 0 only if every required item holds
# EXPECT (what the orchestrator verifies before deleting the job record):
#   RUN/steps.jsonl       lines node, probe, reap, expect, each rc 0
#   RUN/DONE              written only when expect passed
#   RUN/out/expect.json   ok true (re-check on the login node: $PY_EVAL -m vpd.tvr.probe_lvp expect --out RUN/out)
#   RUN/out/summary.json  schema vpd.tvr.probe_lvp/1, label EXPLORATORY: grid_cells, grid_table, recommended
#                         {E, L}, determinism, load_times, reach, font, compare {verdict, frames, controls, turns}
#   RUN/out/reach/<scene>.json  one per SFT + eval scene (120), each loaded twice
#   RUN/out/compare/<task>/     our PNGs of the 2 public sample tasks (31 files)
# Exit: 0 when summary.json exists (whatever expect said), 2 on a bad call / reused RUN, 5 without a summary,
#   124 after a SIGTERM (the summary is still written by probe_lvp's own TERM handler).
# Env: VPD_TVR_GRID (probe_lvp --grid, e.g. '2:1,2,3,4,5;3:1,2,3'; default GRID_DEFAULT), VPD_TVR_STEPS (steps per
#   env and cell, default 100); pass them like VPD_JOB_DEADLINE_S, after `env` in the prepare command.
# Test hook: VPD_TVR_PY (the probe interpreter; default $VPD_OUT/envs/tvr/bin/python).

RUN=${1:-}
if [[ -z $RUN || $# -ne 1 ]]; then
  echo "usage: tvr_probe.sh RUN" >&2
  exit 2
fi
export RUN
# shellcheck source=common_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
cd "$VPD_WORKTREE" || exit 2
set +e   # common_env sets -euo pipefail; keep -u and pipefail, but run every step

OUT=$RUN/out
if [[ -e $OUT/summary.json || -e $RUN/DONE ]]; then
  echo "refusing: $RUN already holds a probe run; prepare a new RUN" >&2
  exit 2
fi
mkdir -p "$OUT/logs" || exit 2
vpd_chown_trap "$RUN"   # no-op: this job runs non-root

T0=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-1920}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
RESERVE_S=90
MIN_PROBE_S=300
PY=${VPD_TVR_PY:-$VPD_OUT/envs/tvr/bin/python}
BUILD_NAME=thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# lvp.worker_env builds every THOR worker's environment from scratch; nothing here may leak a renderer switch
unset ACTIVESPATIAL_USE_X11 ACTIVESPATIAL_X11_FORCE_VULKAN DISPLAY LP_NUM_THREADS

say() { echo "[tvr-probe $(date -u +%H:%M:%SZ)] $*"; }

# Kill Unity players of this build left behind (the probe kills its workers' sessions; this is the backstop):
# TERM, up to 5 s, then KILL.  Logged to OUT/logs/reap.log.
reap_unity() {
  local pids k
  if ! command -v pgrep > /dev/null || ! command -v pkill > /dev/null; then
    echo "[tvr-probe] $1: pgrep/pkill missing, cannot reap Unity players" | tee -a "$OUT/logs/reap.log"
    return 0
  fi
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  [[ -n ${pids// /} ]] || return 0
  echo "[tvr-probe] $(date -u +%TZ) $1: TERM lingering Unity player(s): $pids" | tee -a "$OUT/logs/reap.log"
  pkill -TERM -f -- "$UNITY_PAT"
  for ((k = 0; k < 50; k++)); do
    pgrep -f -- "$UNITY_PAT" > /dev/null || return 0
    sleep 0.1
  done
  echo "[tvr-probe] $(date -u +%TZ) $1: KILL Unity player(s) still alive" | tee -a "$OUT/logs/reap.log"
  pkill -KILL -f -- "$UNITY_PAT"
  return 0
}

probe() {
  local left
  left=$(( DEADLINE_S - ($(date +%s) - T0) - RESERVE_S ))
  if (( left < MIN_PROBE_S )); then
    echo "only ${left}s left before the reserve; the probe needs >= ${MIN_PROBE_S}s"
    return 3
  fi
  [[ -x $PY ]] || { echo "no TVR interpreter $PY"; return 4; }
  local extra=()
  if [[ -n ${VPD_TVR_GRID:-} ]]; then extra+=(--grid "$VPD_TVR_GRID"); fi
  if [[ -n ${VPD_TVR_STEPS:-} ]]; then
    [[ $VPD_TVR_STEPS =~ ^[1-9][0-9]*$ ]] || { echo "VPD_TVR_STEPS must be a positive integer"; return 2; }
    extra+=(--steps "$VPD_TVR_STEPS")
  fi
  timeout -k 30 $(( left + 30 )) "$PY" -s -u -m vpd.tvr.probe_lvp run --out "$OUT" --deadline-s "$left" \
    ${extra[@]+"${extra[@]}"} > "$OUT/logs/probe.log" 2>&1
  local rc=$?
  tail -40 "$OUT/logs/probe.log"
  return $rc
}

expect_check() {
  "$PY" -s -m vpd.tvr.probe_lvp expect --out "$OUT"
}

trap 'say "TERM: reaping and checking the outputs"; reap_unity term; expect_check; exit 124' TERM

say "start RUN=$RUN uid=$(id -u) deadline=${DEADLINE_S}s worktree=$VPD_WORKTREE py=$PY"
vpd_step node vpd_node_facts "$RUN/node.txt"
vpd_step probe probe
vpd_step reap reap_unity probe
vpd_step expect expect_check
EXP=$?
if [[ $EXP -eq 0 ]]; then
  date -u +%FT%TZ > "$RUN/DONE"
fi
[[ -f $OUT/summary.json ]] || { say "summary.json was not written"; exit 5; }
say "end after $(( $(date +%s) - T0 )) s; expect rc=$EXP"
exit 0
