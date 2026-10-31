#!/usr/bin/env bash
# RLinf's own evaluation of E3's start, without any update (the user's decision 2 of 2026-10-31), one one-card sco job:
#   1. fixed    RLinf's evaluations/eval_embodied_agent.py on OUT/cfg/eval.yaml (downstream/rlinf/rlinf_eval.py make)
#               with our hooks on, the image-size fix among them (StarVLA's obs_image_size, 224 x 224);
#   2. control  the same without the image-size fix (DOWNSTREAM_RL_IMAGE_SIZE_FIX=0: the 256 x 256 frames as in E3);
#   3. report   OUT/report.json, report.md.
# Each run has at most TIMEOUT seconds (RLinf's evaluation hung once on 2026-10-27).  Writes only under OUT.
# 2026-10-31: the first job stopped right after ray.init (RLinf never logged its cluster line, which E3's training
# logged 22 s after it).  So a run that has not logged that line after STALL seconds is diagnosed (OUT/diag_<run>/:
# the process tree, py-spy stacks of every Python process, Ray's logs) and stopped, and each run ends with
# `ray stop --force` so the next starts its own Ray.  Python output is unbuffered.
# Usage: run_rlinf_eval.sh OUT [TIMEOUT] [RUNS]   (RUNS default "fixed control"; ENV, RLINF_ROOT, STALL, RAY_TMP
#        for the tests)
set -uo pipefail
OUT="${1:?Usage: run_rlinf_eval.sh OUT [TIMEOUT] [RUNS]}"; TIMEOUT="${2:-3600}"; RUNS="${3:-fixed control}"
STALL="${STALL:-900}"; RAY_TMP="${RAY_TMP:-/tmp/ray}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${ENV:-/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365}"
RLINF_ROOT="${RLINF_ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/RLinf}"
[[ -f $WT/downstream/envs/activate_rlinf.sh && -d $ENV ]] && source "$WT/downstream/envs/activate_rlinf.sh" "$ENV"
RLINF_PY="${RLINF_PY:-$ENV/bin/python}"
export NO_PROXY='*' no_proxy='*' HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface ROBOT_PLATFORM=ROBOCASA365
export NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID" HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1 PATH="/mnt/umm/users/yinbaiqiao/downstream_runs/tools/bin:$PATH"
mkdir -p "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR" "$OUT"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONPATH="$WT/downstream/rlinf/site:$RLINF_ROOT:$WT:$WT/third_party/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
log() { echo "[rlinf-eval] $* $(date -u +%FT%TZ)" | tee -a "$OUT/rlinf_eval.log"; }

diagnose() {  # diagnose NAME: the process tree, py-spy stacks of every Python process, Ray's logs -> OUT/diag_NAME/
  local d="$OUT/diag_$1" p
  mkdir -p "$d"
  ps -eo pid,ppid,pgid,etime,stat,pcpu,rss,args --forest > "$d/ps.txt" 2>&1
  if [[ -x $ENV/bin/py-spy ]]; then
    for p in $(ps -eo pid,args | awk '/python|ray::|raylet|gcs_server/ && !/awk/ {print $1}'); do
      timeout 30 "$ENV/bin/py-spy" dump --pid "$p" > "$d/pyspy_$p.txt" 2>&1
    done
  fi
  [[ -d $RAY_TMP/session_latest/logs ]] && cp -r "$RAY_TMP/session_latest/logs" "$d/ray_logs" 2>/dev/null
  log "$1: diagnosis in $d"
}

cd "$WT"
for NAME in $RUNS; do
  if [[ -s $OUT/rc_$NAME.txt && $(cat "$OUT/rc_$NAME.txt") == 0 ]]; then
    log "$NAME done already"; continue
  fi
  log "$NAME: start (at most $TIMEOUT s; diagnosed and stopped if RLinf has no cluster after $STALL s)"
  setsid bash -c '
    export DOWNSTREAM_RL_HOOKS=1 DOWNSTREAM_RL_HOOK_LOG="$1/hooks_$2.jsonl"
    [[ $2 == control ]] && export DOWNSTREAM_RL_IMAGE_SIZE_FIX=0
    exec timeout -s KILL "$3" "$4" "$5/evaluations/eval_embodied_agent.py" \
      --config-path "$1/cfg" --config-name eval "hydra.run.dir=$1/hydra_$2" > "$1/rlinf_$2.log" 2>&1
  ' bash "$OUT" "$NAME" "$TIMEOUT" "$RLINF_PY" "$RLINF_ROOT" &
  pid=$!
  waited=0; stalled=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2; waited=$((waited + 2))
    if (( waited >= STALL )) && ! grep -q "is running on a cluster" "$OUT/rlinf_$NAME.log" 2>/dev/null; then
      diagnose "$NAME"
      kill -KILL -- "-$pid" 2>/dev/null
      stalled=1
      break
    fi
  done
  wait "$pid"
  rc=$?
  (( stalled )) && rc="stalled"
  [[ -x $ENV/bin/ray ]] && timeout 120 "$ENV/bin/ray" stop --force > "$OUT/ray_stop_$NAME.log" 2>&1
  echo "$rc" > "$OUT/rc_$NAME.txt"
  log "$NAME: rc=$rc; image-size patches applied in $(grep -c obs_image_size "$OUT/hooks_$NAME.jsonl" 2>/dev/null) processes"
done
"$RLINF_PY" "$WT/downstream/rlinf/rlinf_eval.py" report --out "$OUT" >> "$OUT/rlinf_eval.log" 2>&1
log "report rc=$?"
