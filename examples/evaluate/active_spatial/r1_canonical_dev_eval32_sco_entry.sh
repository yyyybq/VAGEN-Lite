#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
RUN_ROOT=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915
FROZEN=${RUN_ROOT}/frozen_input
MODELS=${RUN_ROOT}/model_restore
OUTPUT=${RUN_ROOT}/policy_eval
ARCHIVE=${R1_EVAL_CODE_ARCHIVE:?R1_EVAL_CODE_ARCHIVE is required}
EXPECTED_ARCHIVE_SHA=${R1_EVAL_CODE_ARCHIVE_SHA256:?R1_EVAL_CODE_ARCHIVE_SHA256 is required}
EXPECTED_COMMIT=${R1_EVAL_COMMIT:?R1_EVAL_COMMIT is required}
PORT=${R1_EVAL_RENDER_PORT:-8898}
WORK=/tmp/r1_canonical_dev_eval32_${EXPECTED_COMMIT:0:12}_$(hostname)

mkdir -p "${OUTPUT}" "${WORK}"
actual_archive_sha=$(sha256sum "${ARCHIVE}" | awk '{print $1}')
[[ "${actual_archive_sha}" == "${EXPECTED_ARCHIVE_SHA}" ]] || {
  echo "archive SHA mismatch" >&2
  exit 2
}
tar -xzf "${ARCHIVE}" -C "${WORK}"
cd "${WORK}"
actual_commit=$(git rev-parse HEAD 2>/dev/null || true)
# git archive has no .git; the commit is independently injected and recorded.

cleanup() {
  status=$?
  if [[ -n "${renderer_pid:-}" ]]; then
    kill "${renderer_pid}" 2>/dev/null || true
    wait "${renderer_pid}" 2>/dev/null || true
  fi
  printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/worker_exit.json"
  exit "${status}"
}
trap cleanup EXIT INT TERM

export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${WORK}
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export NO_PROXY='*'
export no_proxy='*'
export TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
export CUDA_VISIBLE_DEVICES=0

{
  echo "hostname=$(hostname)"
  echo "expected_commit=${EXPECTED_COMMIT}"
  echo "archive=${ARCHIVE}"
  echo "archive_sha256=${actual_archive_sha}"
  echo "python=${PY}"
  "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo "gs_root=/mnt/umm/users/yinbaiqiao/InteriorGS"
  echo "render_port=${PORT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUTPUT}/worker_environment.txt"

bash examples/train/active_spatial/start_gs_render_http_service.sh \
  --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS \
  --host 127.0.0.1 --port "${PORT}" --gpus 0 \
  --max-workers 1 --max-inflight 1 --admit-timeout 300 \
  --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite \
  > "${OUTPUT}/renderer.log" 2>&1 &
renderer_pid=$!
echo "${renderer_pid}" > "${OUTPUT}/renderer.pid"
for _ in $(seq 1 180); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" > "${OUTPUT}/renderer_health.json"; then
    break
  fi
  kill -0 "${renderer_pid}" 2>/dev/null || { echo "renderer exited" >&2; exit 3; }
  sleep 2
done
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

common=(
  --protocol "${FROZEN}/eval_protocol.json"
  --renderer-url "http://127.0.0.1:${PORT}/render"
  --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.60
  --max-model-len 32768
)

"${PY}" scripts/r1_run_canonical_dev_eval32.py "${common[@]}" \
  --model-key qwen25vl7b_pretrained_cc594898 \
  --model-path "${MODELS}/qwen25vl7b_pretrained_cc594898" \
  --model-sha256s "${MODELS}/qwen25vl7b_pretrained_cc594898_SHA256SUMS" \
  --output-dir "${OUTPUT}/pretrained" \
  > "${OUTPUT}/pretrained.log" 2>&1

"${PY}" scripts/r1_run_canonical_dev_eval32.py "${common[@]}" \
  --model-key v46_baseline_qwen25vl_7b_step250 \
  --model-path "${MODELS}/v46_step250_hf" \
  --model-sha256s "${MODELS}/v46_step250_hf_SHA256SUMS" \
  --output-dir "${OUTPUT}/v46_step250" \
  > "${OUTPUT}/v46_step250.log" 2>&1

"${PY}" scripts/r1_summarize_canonical_dev_eval32.py \
  --manifest "${FROZEN}/development_regression_manifest.jsonl" \
  --pretrained-ledger "${OUTPUT}/pretrained/episode_ledger.json" \
  --v46-ledger "${OUTPUT}/v46_step250/episode_ledger.json" \
  --output-dir "${OUTPUT}/paired" \
  > "${OUTPUT}/paired_summary.log" 2>&1

find "${OUTPUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUTPUT}/SHA256SUMS"
