#!/usr/bin/env bash
# q38-s4 (C line): a 1-card sco job's entry for the VSI-590K media download (the user's yes of 2026-11-23).
#   sco acp jobs create ... --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 \
#       --command="bash JOB/code/vagen/q38s4/tools/fetch_vsi_job.sh JOB DATA [CMD] [CONNS]"
# JOB/code is a git archive of a committed head (JOB/code/COMMIT), so later commits do not reach the job.
# CMD (default all) and CONNS (default 8) go to vagen.q38s4.tools.fetch_vsi; "all" starts with the probe, so a node
# that cannot reach hf-mirror stops at once (exit code of the probe, the log says "probe failed").
# All proxy variables are unset (hf-mirror is reached directly).  Log: JOB/fetch.log.
set -uo pipefail
JOB=${1:?job dir}
DATA=${2:?data dir (holds vsi_590k.jsonl)}
CMD=${3:-all}
CONNS=${4:-8}
PY=${Q38S4_PYTHON:-/mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python}
LOG=$JOB/fetch.log
{
  echo "[q38s4 fetch] $(date -u +%FT%TZ) host $(hostname) commit $(cat "$JOB/code/COMMIT") cmd ${CMD} conns ${CONNS}"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
  export PYTHONPATH=$JOB/code
  cd "$JOB/code"
  "$PY" -I -c 'import sys; print(sys.version)' 2>&1
  "$PY" -m vagen.q38s4.tools.fetch_vsi "$CMD" --out "$DATA" --conns "$CONNS"
  rc=$?
  echo "[q38s4 fetch] $(date -u +%FT%TZ) exit ${rc}"
  exit $rc
} >>"$LOG" 2>&1
