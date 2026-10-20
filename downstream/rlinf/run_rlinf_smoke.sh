#!/usr/bin/env bash
# RLinf smoke of the downstream line (decision D7): 2 PPO updates of a 7B StarVLA policy on RoboCasa365, one 8-card
# job, in the StarVLA + RoboCasa365 environment.  The config is prepared beforehand on the login node with
#   scripts/prepare_rlinf_robocasa.py prepare --ckpt CKPT --task ... --steps 2 --num-envs 8 --episode-steps 128 --out RUN/cfg
# and run through examples/train/robocasa/run_rlinf_starvla.sh (it re-checks the handoff, then RLinf's
# train_embodied_agent.py).  The image has no git: a copy of the login node's git is put on PATH for the check.
# Usage: run_rlinf_smoke.sh ENV RUN.  Writes only under RUN.
set -uo pipefail
ENV="${1:?Usage: run_rlinf_smoke.sh ENV RUN}"
RUN="${2:?}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$WT/downstream/envs/activate_rlinf.sh" "$ENV"
export PATH="/mnt/umm/users/yinbaiqiao/downstream_runs/tools/bin:$PATH" NO_PROXY='*' no_proxy='*'
export RLINF_PY="$ENV/bin/python" HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID"
mkdir -p "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR"
trap 'rm -rf "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"' EXIT
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
echo "[rlinf] git=$(command -v git) python=$RLINF_PY $(date -u +%FT%TZ)"
cd "$WT"
t0=$(date +%s)
bash "$WT/examples/train/robocasa/run_rlinf_starvla.sh" "$RUN/cfg/train.yaml" > "$RUN/rlinf.log" 2>&1
rc=$?
echo "[rlinf] rc=$rc $(( $(date +%s) - t0 ))s"
grep -a -E "global_step|Traceback|Error|success|env/" "$RUN/rlinf.log" | tail -40
exit $rc
