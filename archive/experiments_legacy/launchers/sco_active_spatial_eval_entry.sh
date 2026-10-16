#!/usr/bin/env bash
# Run one named Active Spatial checkpoint evaluation inside an SCO worker.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
TARGET="${EVAL_TARGET:?set EVAL_TARGET to qwen or cambrian}"

case "${TARGET}" in
  qwen)
    EXPERIMENT="${EVAL_EXPERIMENT:-qwen_v46_clean_d0pass_20260822_sco_renderer_r1_full}"
    STEP="${EVAL_STEP:-450}"
    SWEEP_NAME="${EVAL_SWEEP_NAME:-qwen_v46_clean_step450_eval_20260823}"
    ;;
  cambrian)
    EXPERIMENT="${EVAL_EXPERIMENT:-cambrian_c4_clean_d0pass_20260822_h800_r5_full}"
    STEP="${EVAL_STEP:-640}"
    SWEEP_NAME="${EVAL_SWEEP_NAME:-cambrian_c4_clean_step640_eval_20260823}"
    ;;
  *)
    echo "[fatal] unsupported EVAL_TARGET=${TARGET}" >&2
    exit 2
    ;;
esac

OUT_ROOT="${EVAL_OUT_ROOT:-${ROOT}/evaluation/sweeps/active_spatial}"
OUT_DIR="${OUT_ROOT}/${SWEEP_NAME}"
SCO_DIR="${OUT_DIR}/sco"
STEPS="${EVAL_STEPS:-${STEP}}"
MODE="${EVAL_MODE:-full}"
case "${MODE}" in
  full|nav|qa) ;;
  *) echo "[fatal] EVAL_MODE must be full, nav, or qa; got ${MODE}" >&2; exit 2 ;;
esac

cd "${ROOT}"
export PATH="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}"
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets_easi_shared}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
# The navigation vLLM wrapper and the EASI ``cambrians`` backend both need
# the native Cambrian-S source tree.  Keep this explicit in the SCO entry so
# evaluation does not silently fall back to a Qwen-only import path.
export CAMBRIAN_SRC="${CAMBRIAN_SRC:-/mnt/umm/users/yinbaiqiao/cambrian-s}"
if [[ -d "${CAMBRIAN_SRC}" ]]; then
  export PYTHONPATH="${CAMBRIAN_SRC}:${ROOT}:${PYTHONPATH:-}"
fi
# Match the proven H800 training launcher: vLLM/Triton must use the shared
# conda compiler and avoid its runtime torch.compile path in SCO pods.
export VAGEN_ENV_ROOT="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
export CC="${VAGEN_ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${VAGEN_ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME="${VAGEN_ENV_ROOT}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CPATH="${VAGEN_ENV_ROOT}/targets/x86_64-linux/include:${CPATH:-}"
export CPLUS_INCLUDE_PATH="${VAGEN_ENV_ROOT}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="${VAGEN_ENV_ROOT}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
export VLLM_TORCH_COMPILE_LEVEL=0
export TORCH_COMPILE_DISABLE=1
# Keep evaluation on the same proven vLLM startup path as training.  Without
# this explicit opt-out, vLLM 0.11 selects the FlashInfer sampler and tries to
# JIT-compile it; the SCO image does not ship the curand development headers.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1

# Isolate runtime compiler caches on node-local storage.  This avoids stale
# root-owned FlashInfer artifacts and NFS locking across evaluation jobs.
JIT_CACHE_ROOT="/tmp/vagen_eval_jit_${USER:-root}/${SWEEP_NAME}_$$"
export FLASHINFER_WORKSPACE_BASE="${JIT_CACHE_ROOT}/flashinfer"
export TRITON_CACHE_DIR="${JIT_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JIT_CACHE_ROOT}/torchinductor"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer}"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${SCO_DIR}"

GSPLAT_EXTENSION="${TORCH_EXTENSIONS_DIR}/gsplat_cuda/gsplat_cuda.so"
if [[ ! -s "${GSPLAT_EXTENSION}" ]]; then
  echo "[fatal] precompiled gsplat extension is missing: ${GSPLAT_EXTENSION}" >&2
  echo "[fatal] refusing to let parallel evaluation workers race during JIT compilation" >&2
  exit 4
fi
export VAGEN_GSPLAT_PREBUILT=1

exec > >(tee -a "${SCO_DIR}/full_eval.log") 2>&1
trap 'status=$?; printf "status=%s\\ndate_utc=%s\\n" "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SCO_DIR}/exit_status.txt"; exit "${status}"' EXIT

{
  echo "target=${TARGET}"
  echo "experiment=${EXPERIMENT}"
  echo "steps=${STEPS}"
  echo "sweep_name=${SWEEP_NAME}"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "parallel_gpus=${EVAL_PARALLEL_GPUS:-disabled}"
  echo "mode=${MODE}"
  echo "qa_benchmarks=${EVAL_QA_BENCHMARKS:-easi_8}"
  echo "hf_home=${HF_HOME}"
  echo "hf_datasets_cache=${HF_DATASETS_CACHE}"
  echo "resume_from_sweep=${EVAL_RESUME_FROM_SWEEP:-none}"
  echo "git_commit=$(git -c safe.directory="${ROOT}" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE_ON_WORKER)"
} > "${SCO_DIR}/frozen_state.txt"

if [[ -n "${EVAL_PARALLEL_GPUS:-}" ]]; then
  if [[ "${MODE}" != qa ]]; then
    parallel_args=(
      --suite-config examples/evaluate/active_spatial/test_suites_h800_7b_unified_v1.yaml
      --exp-root exps/vagen_active_spatial
      --out-root evaluation/sweeps/active_spatial
      --sweep-name "${SWEEP_NAME}"
      --exps "${EXPERIMENT}"
      --steps "${STEPS}"
      --suites "${EVAL_NAV_SUITES:-all}"
      --agents model
      --gpus "${EVAL_PARALLEL_GPUS}"
      --max-attempts "${EVAL_MAX_ATTEMPTS:-2}"
    )
    if [[ -n "${EVAL_RESUME_FROM_SWEEP:-}" ]]; then
      parallel_args+=(--resume-from "${EVAL_RESUME_FROM_SWEEP}")
    fi
    if [[ -n "${EVAL_GPU_MEMORY_UTILIZATION:-}" ]]; then
      parallel_args+=(--gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}")
    fi
    "${PYTHON}" scripts/active_spatial_eval_parallel.py "${parallel_args[@]}"
  fi

  if [[ "${MODE}" != nav && "${EVAL_INCLUDE_EASI:-1}" == 1 ]]; then
    qa_args=(
      --exp-root exps/vagen_active_spatial
      --out-root evaluation/sweeps/active_spatial
      --sweep-name "${SWEEP_NAME}"
      --exps "${EXPERIMENT}"
      --steps "${STEPS}"
      --benchmarks "${EVAL_QA_BENCHMARKS:-easi_8}"
      --gpus "${EVAL_PARALLEL_GPUS}"
      --nproc "${EVAL_QA_NPROC:-1}"
      --max-attempts "${EVAL_QA_MAX_ATTEMPTS:-2}"
    )
    if [[ "${EVAL_RERUN_QA:-0}" == 1 ]]; then
      qa_args+=(--rerun)
    fi
    "${PYTHON}" scripts/active_spatial_qa_parallel.py "${qa_args[@]}"
  fi

  if [[ "${MODE}" == full ]]; then
    "${PYTHON}" scripts/active_spatial_eval_report.py \
      --nav-summary "${OUT_DIR}/summary.csv" \
      --easi-summary "${OUT_DIR}/easi_results/easi_summary.json" \
      --out "${OUT_DIR}/analysis_report.md"
  fi
  exit 0
fi

eval_args=(
  --exp-root exps/vagen_active_spatial
  --exps "${EXPERIMENT}"
  --steps "${STEPS}"
  --suite-config examples/evaluate/active_spatial/test_suites_h800_7b_unified_v1.yaml
  --sweep-name "${SWEEP_NAME}"
  --out-root evaluation/sweeps/active_spatial
  --nav-suites "${EVAL_NAV_SUITES:-all}"
  --nav-agents model
)
if [[ "${EVAL_INCLUDE_EASI:-1}" == 1 ]]; then
  eval_args+=(--easi-benchmarks easi_8 --no-easi-base --easi-gpu 0 --easi-nproc 1)
else
  eval_args+=(--no-easi)
fi

"${PYTHON}" scripts/active_spatial_full_eval.py "${eval_args[@]}" \
  --run
