#!/usr/bin/env bash
# Run one fixed-manifest milestone evaluation on the authorized 217 debug node.
set -euo pipefail

STEP="${1:?usage: $0 STEP}"
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
EXP="${ROOT}/exps/vagen_active_spatial/protocol_only_prepost"
CKPT="${EXP}/checkpoints/global_step_${STEP}"
MODEL="${CKPT}/actor/huggingface"
OUT="${EXP}/eval_step_${STEP}"
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python

cd "${ROOT}"
test -f "${CKPT}/COMPLETE"
test -f "${CKPT}/checkpoint_metadata.json"
test -f "${CKPT}/resolved_config.yaml"
test -f "${MODEL}/config.json"
test -f "${EXP}/protocol_eval_manifest.jsonl"

health=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 http://127.0.0.1:8768/health || true)
test "${health}" = 200

# Refuse to compete with another evaluator/training process. The renderer is
# deliberately isolated on physical GPU 7 and is not part of this check.
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null | sort -u | wc -l)
if [ "${busy}" -gt 1 ]; then
  echo "ERROR: debug node GPUs are occupied by another workload; defer milestone eval" >&2
  exit 3
fi

mkdir -p "${OUT}/logs"
export SENSENOVA_U1_SRC=/mnt/umm/users/yinbaiqiao/SenseNova-U1/src
export PYTHONPATH="${SENSENOVA_U1_SRC}:${ROOT}:${ROOT}/verl"
export TRANSFORMERS_ATTN_IMPLEMENTATION=eager

pids=()
status=0
for shard in 0 1 2 3 4 5 6; do
  CUDA_VISIBLE_DEVICES="${shard}" "${PY}" tools/u1_protocol_prepost_eval.py \
    --model "${MODEL}" \
    --out "${OUT}" \
    --checkpoint-label "step_${STEP}" \
    --num-shards 7 \
    --shard-index "${shard}" \
    --overwrite > "${OUT}/logs/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
test "${status}" = 0

"${PY}" tools/u1_protocol_prepost_eval.py \
  --model "${MODEL}" \
  --out "${OUT}" \
  --checkpoint-label "step_${STEP}" \
  --num-shards 7 \
  --aggregate-only
