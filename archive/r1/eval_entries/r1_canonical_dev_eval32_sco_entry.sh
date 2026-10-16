#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
RUN_ROOT=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915
FROZEN=${RUN_ROOT}/frozen_input
MODELS=${RUN_ROOT}/model_restore
OUTPUT=${RUN_ROOT}/policy_eval_h1_render_v2
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
export CUDA_HOME=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
export TORCH_CUDA_ARCH_LIST=9.0
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_COMPILE_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

# Triton compiles a tiny CUDA driver helper during first vLLM initialization.
# The SCO base image used by this evaluation may not include a host C compiler.
# Keep this an explicit infrastructure preflight rather than allowing model
# loading to fail after the full snapshot hash pass.
HISTORICAL_CC=${CUDA_HOME}/bin/x86_64-conda-linux-gnu-gcc
HISTORICAL_CXX=${CUDA_HOME}/bin/x86_64-conda-linux-gnu-g++
if [[ -x "${HISTORICAL_CC}" && -x "${HISTORICAL_CXX}" ]]; then
  export CC=${HISTORICAL_CC}
  export CXX=${HISTORICAL_CXX}
else
  if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y gcc g++
  fi
  if command -v gcc >/dev/null 2>&1; then
    export CC
    CC=$(command -v gcc)
  elif command -v clang >/dev/null 2>&1; then
    export CC
    CC=$(command -v clang)
  else
    export CC
    CC=$(command -v cc)
  fi
  if command -v g++ >/dev/null 2>&1; then
    export CXX
    CXX=$(command -v g++)
  elif command -v clang++ >/dev/null 2>&1; then
    export CXX
    CXX=$(command -v clang++)
  fi
fi

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
  echo "cc=${CC}"
  "${CC}" --version | head -1
  if [[ -n "${CXX:-}" ]]; then
    echo "cxx=${CXX}"
    "${CXX}" --version | head -1
  fi
  echo "cuda_home=${CUDA_HOME}"
  echo "vllm_use_flashinfer_sampler=${VLLM_USE_FLASHINFER_SAMPLER}"
  echo "vllm_attention_backend=${VLLM_ATTENTION_BACKEND}"
  echo "vllm_torch_compile_level=${VLLM_TORCH_COMPILE_LEVEL}"
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
  --max-model-len 4480
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

# Stop the official renderer before freezing output hashes so its final log
# flush cannot invalidate the manifest after it has been written.
kill "${renderer_pid}" 2>/dev/null || true
wait "${renderer_pid}" 2>/dev/null || true
renderer_pid=
printf '{"exit_status":0,"stopped_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/worker_exit.json"
trap - EXIT INT TERM
find "${OUTPUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUTPUT}/SHA256SUMS"
