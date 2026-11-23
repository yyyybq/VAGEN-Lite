#!/usr/bin/env bash
# q38-s4 (C line): a one-card sco job's entry (vagen.q38s4.tools.job): runs JOB/job.json's module from JOB/code.
#   sco acp jobs create ... --worker-nodes=1 --worker-spec=N4lS.Iq.I80.1 --command="bash JOB/code/vagen/q38s4/tools/job_entry.sh JOB"
# The CUDA 13 forward-compatibility libraries go first on LD_LIBRARY_PATH when the driver is older than 580 (as in
# run_train.sh).  Log: JOB/job.log; the exit code also goes to JOB/exit_code.
set -uo pipefail
JOB=${1:?job dir}
PY=${Q38S4_PYTHON:-/mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python}
LOG=$JOB/job.log
{
  echo "[q38s4 job] $(date -u +%FT%TZ) host $(hostname) commit $(cat "$JOB/code/COMMIT")"
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
  echo "[driver] ${driver}"
  compat=/mnt/umm/users/yinbaiqiao/q38/envs/cuda-compat/13-0/usr/local/cuda-13.0/compat
  if [[ ${driver%%.*} =~ ^[0-9]+$ ]] && (( ${driver%%.*} < 580 )); then
    export LD_LIBRARY_PATH=${compat}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    echo "[compat] on"
  fi
  cd "$JOB/code"
  PYTHONPATH=$JOB/code "$PY" -m vagen.q38s4.tools.job run --job "$JOB"
  rc=$?
  echo "[q38s4 job] $(date -u +%FT%TZ) exit ${rc}"
  echo "${rc}" > "$JOB/exit_code"
  exit ${rc}
} >> "$LOG" 2>&1
