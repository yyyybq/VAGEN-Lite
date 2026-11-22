#!/usr/bin/env bash
# q38-s4 (C line, stage 0 item 4): a multi-node sco job's entry for the all-reduce bandwidth test.
#   sco acp jobs create ... --worker-nodes=N --worker-spec=N4lS.Iq.I80.8 --command="bash <this> OUT_DIR"
# Every node runs it.  The node rank and the master's address come from the PyTorch job's variables (PET_NODE_RANK,
# else RANK as the replica index; MASTER_ADDR / MASTER_PORT; PET_NNODES, else WORLD_SIZE as the node count); all of
# them are printed first.  The CUDA 13 forward-compatibility libraries go first on LD_LIBRARY_PATH when the driver is
# older than 580 (as the q38 task script does).  The run is capped at 8 minutes; NCCL's INFO log goes to
# OUT_DIR/nccl_node<R>.log (transport lines: NET/IB vs NET/Socket).
set -uo pipefail
OUT=${1:?out dir}
mkdir -p "$OUT"
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${Q38S4_PYTHON:-/mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python}
NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
NNODES=${PET_NNODES:-${WORLD_SIZE:-1}}
LOG=$OUT/node${NODE_RANK}.log
{
  echo "[q38s4 allreduce] $(date -u +%FT%TZ) host $(hostname) node_rank ${NODE_RANK} nnodes ${NNODES}"
  env | grep -E '^(RANK|WORLD_SIZE|LOCAL_RANK|MASTER_ADDR|MASTER_PORT|PET_[A-Z_]+|NODE_RANK|VC_[A-Z_]+|NCCL_[A-Z_]+)=' | sort
  ip -4 -o addr 2>/dev/null | awk '{print "[ip]", $2, $4}'
  ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ' | sed 's/^/[ib devices] /'; echo
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
  echo "[driver] ${driver}"
  compat=/mnt/umm/users/yinbaiqiao/q38/envs/cuda-compat/13-0/usr/local/cuda-13.0/compat
  if [[ ${driver%%.*} =~ ^[0-9]+$ ]] && (( ${driver%%.*} < 580 )); then
    export LD_LIBRARY_PATH=${compat}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    echo "[compat] on"
  fi
  export OMP_NUM_THREADS=4 NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$OUT/nccl_node${NODE_RANK}.%h.%p.log
  cd "$WT"
  timeout 480 "$PY" -m torch.distributed.run --nnodes "${NNODES}" --nproc-per-node 8 --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT:-29500}" \
    vagen/q38s4/tools/allreduce_bench.py --out "$OUT"
  rc=$?
  echo "[q38s4 allreduce] $(date -u +%FT%TZ) exit ${rc}"
  exit ${rc}
} >> "$LOG" 2>&1
