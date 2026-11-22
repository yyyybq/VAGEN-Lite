#!/usr/bin/env bash
# q38-s4 (C line): a sco job's entry for the stage-4 SFT, written into RUN/code by vagen.q38s4.launch prepare.
#   sco acp jobs create ... --worker-nodes=N --worker-spec=N4lS.Iq.I80.G --command="bash RUN/code/vagen/q38s4/run_train.sh RUN"
# Every node runs it.  RUN/plan.json gives the mode (train | tokenize), the node count and the GPUs per node; the
# node rank and the master's address come from the PyTorch job's variables (PET_NODE_RANK, else RANK as the replica
# index; MASTER_ADDR / MASTER_PORT; PET_NNODES, else WORLD_SIZE), all printed first, and a node count other than the
# plan's stops the node.  The CUDA 13 forward-compatibility libraries go first on LD_LIBRARY_PATH when the driver is
# older than 580.  The code is RUN/code (a git archive), never the worktree, so later commits do not reach a running
# job.  Log: RUN/logs/node<R>.log.
#   train:    accelerate launch (FSDP1 config of the q38 27B SFT, machine / process counts from the plan)
#             -m vagen.q38s4.lf_nfp RUN/train.yaml
#   tokenize: one process, Q38S4_TOKENIZE_ONLY=1 python -m vagen.q38s4.lf_nfp RUN/train.yaml
set -uo pipefail
RUN=${1:?run dir}
CODE=$RUN/code
PY=${Q38S4_PYTHON:-/mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python}
NODE_RANK=${PET_NODE_RANK:-${RANK:-0}}
NNODES=${PET_NNODES:-${WORLD_SIZE:-1}}
mkdir -p "$RUN/logs"
LOG=$RUN/logs/node${NODE_RANK}.log
plan() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$RUN/plan.json" "$1"; }
{
  echo "[q38s4 train] $(date -u +%FT%TZ) host $(hostname) node_rank ${NODE_RANK} nnodes ${NNODES} commit $(cat "$CODE/COMMIT")"
  env | grep -E '^(RANK|WORLD_SIZE|LOCAL_RANK|MASTER_ADDR|MASTER_PORT|PET_[A-Z_]+|NODE_RANK|VC_[A-Z_]+|NCCL_[A-Z_]+)=' | sort
  ip -4 -o addr 2>/dev/null | awk '{print "[ip]", $2, $4}'
  ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ' | sed 's/^/[ib devices] /'; echo
  MODE=$(plan mode); PLAN_NODES=$(plan nodes); GPUS=$(plan gpus_per_node)
  echo "[plan] mode ${MODE} nodes ${PLAN_NODES} gpus_per_node ${GPUS} grad_accum $(plan grad_accum)"
  if [[ "${NNODES}" != "${PLAN_NODES}" ]]; then
    echo "[q38s4 train] the job has ${NNODES} nodes, the plan ${PLAN_NODES}: stop"; exit 2
  fi
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
  echo "[driver] ${driver}"
  compat=/mnt/umm/users/yinbaiqiao/q38/envs/cuda-compat/13-0/usr/local/cuda-13.0/compat
  if [[ ${driver%%.*} =~ ^[0-9]+$ ]] && (( ${driver%%.*} < 580 )); then
    export LD_LIBRARY_PATH=${compat}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    echo "[compat] on"
  fi
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
  export DISABLE_VERSION_CHECK=1 PYTHONPATH=$CODE VAGEN_Q38_SFT_METRICS_DIR=$RUN/sft_metrics
  cd "$CODE"
  if [[ "${MODE}" == "tokenize" ]]; then
    Q38S4_TOKENIZE_ONLY=1 "$PY" -m vagen.q38s4.lf_nfp "$RUN/train.yaml"
  else
    "$PY" -m accelerate.commands.launch --config_file "$CODE/vagen/q38/configs/sft/accelerate_fsdp_qwen35.yaml" \
      --num_machines "${NNODES}" --machine_rank "${NODE_RANK}" --num_processes $((NNODES * GPUS)) \
      --main_process_ip "${MASTER_ADDR:-127.0.0.1}" --main_process_port "${MASTER_PORT:-29500}" \
      -m vagen.q38s4.lf_nfp "$RUN/train.yaml"
  fi
  rc=$?
  echo "[q38s4 train] $(date -u +%FT%TZ) exit ${rc}"
  exit ${rc}
} >> "$LOG" 2>&1
