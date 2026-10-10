#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=${1:?usage: sco_r5_production_backward.sh RUN_ROOT}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
SNAPSHOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5/snapshot/global_step_000001_pre_update
CRITIC=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5/critic_initial_pre_update
OUTPUT=${RUN_ROOT}/r5_production_backward.json

mkdir -p "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/r5_production_backward.log") 2>&1

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_ROOT}/bin:${PATH}"
export CUDA_HOME="${ENV_ROOT}"
export CURAND_INCLUDE_DIR="${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand/include"
export CURAND_LIBRARY_DIR="${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand/lib"
export LD_LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29617}

test -f "${SNAPSHOT}/manifest.json"
test -f "${SNAPSHOT}/payload.pt"
test -f "${CRITIC}/fsdp_config.json"
[[ $("${ENV_ROOT}/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["world_size"])' "${CRITIC}/fsdp_config.json") == 4 ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -ge 4 ]]

"${ENV_ROOT}/bin/torchrun" --standalone --nnodes=1 --nproc-per-node=4 \
  "${REPO_ROOT}/scripts/active_spatial_r5_production_backward.py" \
  --snapshot "${SNAPSHOT}" \
  --critic-checkpoint "${CRITIC}" \
  --output "${OUTPUT}" \
  --expected-world-size 4

"${ENV_ROOT}/bin/python" - "${OUTPUT}" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "PASS":
    raise SystemExit("BLOCKED: production backward report did not pass")
print(json.dumps({
    "status": report["status"],
    "actor_parity": report["actor"]["old_logprob_parity"],
    "critic_parity": report["critic"]["saved_value_parity"],
    "actor_s1_vs_s0": report["actor"]["branches"]["S1_vs_S0"],
    "critic_s1_vs_s0": report["critic"]["branches"]["S1_vs_S0"],
}, sort_keys=True))
PY
