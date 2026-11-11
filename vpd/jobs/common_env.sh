# Common prelude of every VPD sco payload.  Source it, do not run it; a payload in vpd/jobs/ sources the copy next
# to itself (vpd/tvr/*.sh: "$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh"):
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
#     cd "$VPD_WORKTREE"
#
# Sets strict mode, per-process thread caps (VPD_THREADS, default 2 = DECISIONS B9; vLLM processes call
# `vpd_threads 4`), offline HF, no bytecode, no vLLM usage stats, VPD_WORKTREE and PYTHONPATH = that tree
# only, and defines:
#   vpd_threads N            re-export OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=N (and VPD_THREADS)
#   vpd_on_exit CMD          add CMD to the EXIT handler (handlers run in order; the exit code is kept)
#   vpd_chown_trap DIR...    as root only: chown -R 20325:20325 DIR... at EXIT
#   vpd_node_facts OUTFILE   id, nproc, cgroup cpu.max, df, free, nvidia-smi, /dev/nvidia* and /dev/dri
#   vpd_step NAME CMD...     run CMD, append {"name","start","end","secs","rc"} to $RUN/steps.jsonl, return rc
#   vpd_snapshot_file REL    in a code snapshot ($VPD_WORKTREE/SNAPSHOT.json): exit 0 and print the commit that last
#                            changed REL only if the job's copy is the committed file (python -m vpd.snapshot file)
# VPD_WORKTREE: kept when set (launch.sh of a snapshot job exports RUN/code, DECISIONS G1), else the tree this file
# lives in (the git worktree, or a job's code snapshot RUN/code); it must contain vpd/.
# Nothing here writes outside the directories the payload names.

set -euo pipefail

export VPD_THREADS="${VPD_THREADS:-2}"
export OMP_NUM_THREADS="$VPD_THREADS" MKL_NUM_THREADS="$VPD_THREADS" \
       OPENBLAS_NUM_THREADS="$VPD_THREADS" NUMEXPR_NUM_THREADS="$VPD_THREADS"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# vLLM reports usage stats to stats.vllm.ai by default; VLLMServer passes os.environ on (DECISIONS G14a)
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
VPD_WORKTREE="${VPD_WORKTREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[[ $VPD_WORKTREE == /* && -d $VPD_WORKTREE/vpd ]] || {
  echo "[vpd] VPD_WORKTREE='$VPD_WORKTREE' is not an absolute path of a code tree with vpd/" >&2
  exit 2
}
export VPD_WORKTREE
export PYTHONPATH="$VPD_WORKTREE"
export VPD_OUT=/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd
export PY_EVAL=/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin/python
export PY_Q3=/mnt/umm/users/yinbaiqiao/eval_external_assets/envs/eval-external-q3/bin/python
export PY_TRAIN=/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python
export VPD_JOB_UID=20325

vpd_threads() {
  [[ $# -eq 1 && $1 =~ ^[1-9][0-9]*$ ]] || { echo "vpd_threads: give one positive integer" >&2; return 2; }
  export VPD_THREADS="$1" OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1" \
         NUMEXPR_NUM_THREADS="$1"
}

# EXIT handlers: one trap, many commands, so a payload's own handler and the chown both run.
if [[ -z ${VPD__EXIT_CMDS+x} ]]; then
  VPD__EXIT_CMDS=()
fi
vpd__run_exit() {
  local rc=$? c
  set +e
  for c in "${VPD__EXIT_CMDS[@]}"; do
    eval "$c"
  done
  exit "$rc"
}
vpd_on_exit() {
  [[ $# -ge 1 ]] || { echo "vpd_on_exit: give a command" >&2; return 2; }
  VPD__EXIT_CMDS+=("$*")
  trap vpd__run_exit EXIT
}

vpd_chown_trap() {
  [[ $# -ge 1 ]] || { echo "vpd_chown_trap: give at least one directory" >&2; return 2; }
  if [[ $(id -u) -eq 0 ]]; then
    local q
    q=$(printf '%q ' "$@")
    vpd_on_exit "chown -R ${VPD_JOB_UID}:${VPD_JOB_UID} $q || echo '[vpd] chown failed' >&2"
  fi
  return 0
}

vpd_node_facts() {
  [[ $# -eq 1 ]] || { echo "vpd_node_facts: give OUTFILE" >&2; return 2; }
  local out=$1
  mkdir -p "$(dirname "$out")"
  (
    set +e
    echo "## date";      date -u +%FT%TZ
    echo "## host";      hostname
    echo "## id";        id
    echo "## nproc";     nproc
    echo "## cpu.max";   cat /sys/fs/cgroup/cpu.max 2>&1
    echo "## df";        df -h / /tmp /dev/shm 2>&1
    echo "## free";      free -g 2>&1
    echo "## nvidia-smi"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
    echo "## devices";   ls -la /dev/nvidia* /dev/dri 2>&1
    true
  ) > "$out" 2>&1
  return 0
}

vpd_step() {
  [[ $# -ge 2 ]] || { echo "vpd_step: give NAME CMD..." >&2; return 2; }
  [[ -n ${RUN:-} ]] || { echo "vpd_step: RUN is not set" >&2; return 2; }
  local name=$1; shift
  [[ $name =~ ^[A-Za-z0-9._:@+-]+$ ]] || { echo "vpd_step: bad step name '$name'" >&2; return 2; }
  local start end t0 t1 rc had_e=0
  [[ $- == *e* ]] && had_e=1
  start=$(date -u +%FT%TZ); t0=$(date +%s)
  echo "[vpd_step] $name start $start"
  set +e
  "$@"
  rc=$?
  ((had_e)) && set -e
  end=$(date -u +%FT%TZ); t1=$(date +%s)
  mkdir -p "$RUN"
  printf '{"name":"%s","start":"%s","end":"%s","secs":%d,"rc":%d}\n' "$name" "$start" "$end" $((t1 - t0)) "$rc" \
    >> "$RUN/steps.jsonl"
  echo "[vpd_step] $name end $end rc=$rc"
  return "$rc"
}

vpd_snapshot_file() {  # runs from $VPD_WORKTREE, so `-m vpd.snapshot` is that tree's copy whatever the cwd
  [[ $# -eq 1 ]] || { echo "vpd_snapshot_file: give one path relative to \$VPD_WORKTREE" >&2; return 2; }
  (cd "$VPD_WORKTREE" && "$PY_Q3" -B -m vpd.snapshot file "$1")
}
