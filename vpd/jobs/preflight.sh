#!/usr/bin/env bash
# VPD login-node preflight (read-only): cores in use, disk flags, live vpd sco jobs, ledger budget.
#     bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/preflight.sh
# Prints one final line "PREFLIGHT GO" (exit 0) or "PREFLIGHT NO-GO: <reasons>" (exit 1).
# Runs one python process at a time with 1 thread, under nice 19.  Writes nothing.
set -uo pipefail

WORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd
PY=${VPD_PREFLIGHT_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin/python}
MAX_CORES=${VPD_PREFLIGHT_MAX_CORES:-8}   # DECISIONS A6: wait when more than 8 of 14 cores are in use
CPU_STAT=/sys/fs/cgroup/cpu.stat

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WORKTREE"

reasons=()
echo "[preflight] $(date -u +%FT%TZ) host=$(hostname)"

# 1. cores in use (CLAUDE.md check, 5 s window)
if [[ -r $CPU_STAT ]]; then
  a=$(awk '/^usage_usec/{print $2}' "$CPU_STAT"); sleep 5; b=$(awk '/^usage_usec/{print $2}' "$CPU_STAT")
  cores=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.1f", (b-a)/5e6}')
  echo "[preflight] cores in use $cores of 14 (limit for new work: $MAX_CORES)"
  if awk -v c="$cores" -v m="$MAX_CORES" 'BEGIN{exit !(c > m)}'; then
    reasons+=("cores in use $cores > $MAX_CORES")
  fi
else
  echo "[preflight] $CPU_STAT unreadable"
  reasons+=("cpu.stat unreadable")
fi

cd "$WORKTREE" || { echo "PREFLIGHT NO-GO: no worktree $WORKTREE"; exit 1; }

# 2. disk flags and remaining space (same rule as vpd.sco submit)
echo "[preflight] disk:"
if ! nice -n 19 "$PY" -m vpd.sco disk | sed 's/^/  /'; then
  reasons+=("disk check refused")
fi

# 3. live downstream-vpd-* jobs of yinbaiqiao (one page of `sco acp jobs list`, newest 500)
echo "[preflight] queue:"
qout=$(nice -n 19 "$PY" -m vpd.sco queue 2>&1); qrc=$?
printf '%s\n' "$qout" | sed 's/^/  /'
if [[ $qrc -ne 0 ]]; then
  reasons+=("job list unreadable")
else
  live=$(printf '%s\n' "$qout" | sed -n 's/^QUEUE live_vpd=\([0-9]*\) max=\([0-9]*\)$/\1 \2/p')
  read -r n_live n_max <<<"${live:-x x}"
  if [[ $n_live =~ ^[0-9]+$ && $n_max =~ ^[0-9]+$ ]]; then
    ((n_live >= n_max)) && reasons+=("$n_live live vpd jobs (max $n_max)")
  else
    reasons+=("queue output not understood")
  fi
fi

# 4. ledger budget (informational; the per-job check runs at submit)
echo "[preflight] budget (vpd.sco budget):"
if ! nice -n 19 "$PY" -m vpd.sco budget 2>&1 | sed 's/^/  /'; then
  reasons+=("ledger unreadable")
fi

if ((${#reasons[@]} == 0)); then
  echo "PREFLIGHT GO"
  exit 0
fi
msg=$(printf '%s; ' "${reasons[@]}")
echo "PREFLIGHT NO-GO: ${msg%; }"
exit 1
