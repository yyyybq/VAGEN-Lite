#!/usr/bin/env bash
# RLinf's own evaluation of E3's start, without any update (the user's decision 2 of 2026-10-31), one one-card sco job:
#   1. fixed    RLinf's evaluations/eval_embodied_agent.py on OUT/cfg/eval.yaml (downstream/rlinf/rlinf_eval.py make)
#               with our hooks on, the image-size fix among them (StarVLA's obs_image_size, 224 x 224);
#   2. control  the same without the image-size fix (DOWNSTREAM_RL_IMAGE_SIZE_FIX=0: the 256 x 256 frames as in E3);
#   3. report   OUT/report.json, report.md.
# Each run has at most TIMEOUT seconds (RLinf's evaluation hung once on 2026-10-27).  Writes only under OUT.
# Usage: run_rlinf_eval.sh OUT [TIMEOUT] [RUNS]   (RUNS default "fixed control"; ENV and RLINF_ROOT for the tests)
set -uo pipefail
OUT="${1:?Usage: run_rlinf_eval.sh OUT [TIMEOUT] [RUNS]}"; TIMEOUT="${2:-3600}"; RUNS="${3:-fixed control}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV="${ENV:-/mnt/umm/users/yinbaiqiao/envs/rlinf-starvla-rc365}"
RLINF_ROOT="${RLINF_ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/RLinf}"
[[ -f $WT/downstream/envs/activate_rlinf.sh && -d $ENV ]] && source "$WT/downstream/envs/activate_rlinf.sh" "$ENV"
RLINF_PY="${RLINF_PY:-$ENV/bin/python}"
export NO_PROXY='*' no_proxy='*' HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface ROBOT_PLATFORM=ROBOCASA365
export NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID" HYDRA_FULL_ERROR=1
mkdir -p "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR" "$OUT"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
export PYTHONPATH="$WT/downstream/rlinf/site:$RLINF_ROOT:$WT:$WT/third_party/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
log() { echo "[rlinf-eval] $* $(date -u +%FT%TZ)" | tee -a "$OUT/rlinf_eval.log"; }
cd "$WT"
for NAME in $RUNS; do
  if [[ -s $OUT/rc_$NAME.txt && $(cat "$OUT/rc_$NAME.txt") == 0 ]]; then
    log "$NAME done already"; continue
  fi
  log "$NAME: start"
  (
    export DOWNSTREAM_RL_HOOKS=1 DOWNSTREAM_RL_HOOK_LOG="$OUT/hooks_$NAME.jsonl"
    [[ $NAME == control ]] && export DOWNSTREAM_RL_IMAGE_SIZE_FIX=0
    timeout -s KILL "$TIMEOUT" "$RLINF_PY" "$RLINF_ROOT/evaluations/eval_embodied_agent.py" \
      --config-path "$OUT/cfg" --config-name eval "hydra.run.dir=$OUT/hydra_$NAME" > "$OUT/rlinf_$NAME.log" 2>&1
  )
  rc=$?
  echo "$rc" > "$OUT/rc_$NAME.txt"
  log "$NAME: rc=$rc; image-size patches applied in $(grep -c obs_image_size "$OUT/hooks_$NAME.jsonl" 2>/dev/null) processes"
done
"$RLINF_PY" "$WT/downstream/rlinf/rlinf_eval.py" report --out "$OUT" >> "$OUT/rlinf_eval.log" 2>&1
log "report rc=$?"
