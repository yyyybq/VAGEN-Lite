#!/usr/bin/env bash
# sco payload of job vpd-wb-demos (non-root; the GPU stays idle, rendering is CPU):
#     timeout -k 60 <outer> bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_demos.sh RUN
# outer = floor(est_high*3600) - 300 (pilot.py job_timeouts; plan est_high 0.3 -> 780 s; WP-WBDEMO measured a few
# minutes), so the job never holds the card past the GPU-h charged for it (DECISIONS B2).  A cut job leaves no DONE.
# RUN = the job's run dir (OUT/jobs/<job>), also taken from $RUN.  Steps (each logged by vpd_step to RUN/steps.jsonl,
# the first failure ends the job):
#   split_check     python -m vpd.walkerbench.split --check   (the frozen S2 split is intact)
#   demos_shards    8 x python -m vpd.walkerbench.demos --shard i/8 in the background, then wait (1 thread each)
#   demos_merge     python -m vpd.walkerbench.demos --merge   -> OUT/walkerbench/demos/s2train_640x400/demos_all.jsonl
#   warmup          python -m vpd.walkerbench.warmup          -> OUT/walkerbench/warmup/warmup20.jsonl
#   tiers_<S>       python -m vpd.walkerbench.tiers materialize --seed S (S = 1001, 1, 2, 3)
#   DONE            touch OUT/walkerbench/demos/s2train_640x400/DONE and RUN/DONE
# Env: VPD_DEMO_SHARDS (default 8), VPD_DEMO_WORK (sandbox parent, default RUN/work; removed by the python steps),
#      VPD_DRYRUN=1 prints the commands instead of running them (no step is executed, nothing is written but RUN).
# Threads: VPD_THREADS=1 per process, 8 renderers + the shell = at most 9 of the job's 14 vCPU.

RUN=${1:-${RUN:-}}
if [[ -z $RUN ]]; then
  echo "usage: wb_demos.sh RUN" >&2
  exit 2
fi
export RUN
export VPD_THREADS=1
# shellcheck source=common_env.sh
source /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/common_env.sh

NSHARD=${VPD_DEMO_SHARDS:-8}
[[ $NSHARD =~ ^[1-9][0-9]*$ ]] || { echo "VPD_DEMO_SHARDS must be a positive integer" >&2; exit 2; }
WORK=${VPD_DEMO_WORK:-$RUN/work}
DEMOS_OUT=$VPD_OUT/walkerbench/demos/s2train_640x400
SEEDS=(1001 1 2 3)
PY=$PY_EVAL
DRY=${VPD_DRYRUN:-0}

mkdir -p "$RUN"
cd "$VPD_WORKTREE"

x() {  # run, or print under VPD_DRYRUN=1
  if [[ $DRY == 1 ]]; then
    echo "[dry] $*"
  else
    "$@"
  fi
}

run_shards() {
  local i rc=0 pids=()
  mkdir -p "$RUN/logs"
  for ((i = 0; i < NSHARD; i++)); do
    if [[ $DRY == 1 ]]; then
      echo "[dry] nice -n 10 $PY -m vpd.walkerbench.demos --shard $i/$NSHARD --work $WORK > $RUN/logs/demos_shard$i.log 2>&1 &"
      continue
    fi
    nice -n 10 "$PY" -m vpd.walkerbench.demos --shard "$i/$NSHARD" --work "$WORK" \
      > "$RUN/logs/demos_shard$i.log" 2>&1 &
    pids+=($!)
  done
  [[ $DRY == 1 ]] && return 0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "[wb_demos] shard $i/$NSHARD failed; tail of its log:" >&2
      tail -n 20 "$RUN/logs/demos_shard$i.log" >&2 || true
      rc=1
    fi
  done
  return "$rc"
}

echo "[wb_demos] start $(date -u +%FT%TZ) RUN=$RUN shards=$NSHARD work=$WORK dry=$DRY"
vpd_node_facts "$RUN/node.txt"
vpd_step split_check x "$PY" -m vpd.walkerbench.split --check
vpd_step demos_shards run_shards
vpd_step demos_merge x "$PY" -m vpd.walkerbench.demos --merge
vpd_step warmup x "$PY" -m vpd.walkerbench.warmup --work "$WORK"
for S in "${SEEDS[@]}"; do
  vpd_step "tiers_$S" x "$PY" -m vpd.walkerbench.tiers materialize --seed "$S" --tiers 25,50,100,200,all \
    --demos "$DEMOS_OUT/demos_all.jsonl"
done
if [[ $DRY == 1 ]]; then
  echo "[dry] touch $DEMOS_OUT/DONE $RUN/DONE"
else
  date -u +%FT%TZ > "$DEMOS_OUT/DONE"
  date -u +%FT%TZ > "$RUN/DONE"
fi
echo "[wb_demos] done $(date -u +%FT%TZ)"
