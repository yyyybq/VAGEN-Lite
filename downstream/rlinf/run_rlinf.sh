#!/usr/bin/env bash
# One RLinf run of the downstream line (step 4; the 1-hour measurement and the PPO runs), one 8-card job, in the
# StarVLA + RoboCasa365 RLinf environment.  As run_rlinf_smoke.sh (handoff check, then RLinf's
# train_embodied_agent.py on RUN/cfg/train.yaml, made by scripts/prepare_rlinf_robocasa.py prepare), plus:
#   * the 7B FSDP settings of downstream/rlinf/fsdp_7b.overrides, before the overrides given after RUN;
#   * downstream/rlinf/site on PYTHONPATH with DOWNSTREAM_RL_HOOKS=1 (sitecustomize.py): the fixed counter region
#     order in every environment process, and the actor's and the value head's gradient norms, before clipping, in
#     RUN/grad_norms_<TAG>.jsonl; RUN/hooks_<TAG>.jsonl lists every process where a patch took effect;
#   * GPU memory and utilisation every 2 s (RUN/gpu_<TAG>.csv) and CPU load every 10 s (RUN/cpu_<TAG>.csv).
# Usage: run_rlinf.sh ENV RUN [HYDRA_OVERRIDE ...].  Writes only under RUN.
set -uo pipefail
ENV="${1:?Usage: run_rlinf.sh ENV RUN [OVERRIDE ...]}"
RUN="${2:?}"
shift 2
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mapfile -t FIXED < <(grep -v '^\s*#' "$WT/downstream/rlinf/fsdp_7b.overrides" | grep -v '^\s*$')
OVERRIDES=("${FIXED[@]}" "$@")
source "$WT/downstream/envs/activate_rlinf.sh" "$ENV"
export PATH="/mnt/umm/users/yinbaiqiao/downstream_runs/tools/bin:$PATH" NO_PROXY='*' no_proxy='*'
export RLINF_PY="$ENV/bin/python" HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID"
mkdir -p "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
CONFIG="$RUN/cfg/train.yaml"
TAG="${RLINF_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="$RUN/rlinf_$TAG.log"
export DOWNSTREAM_RL_HOOKS=1 DOWNSTREAM_RL_HOOK_LOG="$RUN/hooks_$TAG.jsonl" DOWNSTREAM_GRAD_LOG="$RUN/grad_norms_$TAG.jsonl"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 2000 \
  > "$RUN/gpu_$TAG.csv" &
GPU_PID=$!
( while sleep 10; do echo "$(date -u +%FT%TZ),$(cut -d' ' -f1-3 /proc/loadavg | tr ' ' ',')"; done ) > "$RUN/cpu_$TAG.csv" &
CPU_PID=$!
trap 'kill $GPU_PID $CPU_PID 2>/dev/null; rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
echo "[rlinf] python=$RLINF_PY overrides=${OVERRIDES[*]} log=$LOG $(date -u +%FT%TZ)"
cd "$WT"
export PYTHONPATH="$WT/downstream/rlinf/site:$WT:$WT/third_party/RLinf:$WT/third_party/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export ROBOT_PLATFORM=ROBOCASA365
"$RLINF_PY" "$WT/scripts/prepare_rlinf_robocasa.py" check --config "$CONFIG" > "$RUN/handoff_check_$TAG.json" \
  || { echo "[rlinf] handoff check failed"; cat "$RUN/handoff_check_$TAG.json"; exit 3; }
RLINF_ROOT=$("$RLINF_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rlinf_root"])' "$RUN/cfg/handoff.json")
export PYTHONPATH="$WT/downstream/rlinf/site:$RLINF_ROOT:$PYTHONPATH" EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
t0=$(date +%s)
"$RLINF_PY" "$EMBODIED_PATH/train_embodied_agent.py" --config-path "$RUN/cfg" --config-name train "${OVERRIDES[@]}" \
  > "$LOG" 2>&1
rc=$?
echo "[rlinf] rc=$rc $(( $(date +%s) - t0 ))s; hooks: $(grep -c . "$RUN/hooks_$TAG.jsonl" 2>/dev/null) lines," \
     "grad norms: $(grep -c . "$RUN/grad_norms_$TAG.jsonl" 2>/dev/null) lines"
grep -a -E "global_step|Traceback|Error|env/|train/|time/" "$LOG" | grep -v -i warn | tail -40
exit $rc
