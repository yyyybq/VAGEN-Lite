#!/usr/bin/env bash
# Single-GPU SCO worker: real Act->QA canary on the frozen 30-state bank.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
PY="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
CONDA_ENV="${CONDA_ENV:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite}"
OUT="${ACT2QA_OUT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_qa/artifacts/act2qa_real_canary_20260919T0618}"
BANK_SRC="${BANK_SRC:-${ROOT}/data_gen/active_spatial_qa/artifacts/qa_v2_runtime_canary_20260914/oracle_bank/manifest.jsonl}"
SCENE_ID="${SCENE_ID:-0003_839989}"
LOCAL_GS="${LOCAL_GS:-/mnt/umm/users/yinbaiqiao/InteriorGS}"
BASE_MODEL="${BASE_MODEL:-/mnt/umm/shared_model/Qwen2.5-VL-7B-Instruct}"
ACT_SRC_DEFAULT="${ACT_SRC_DEFAULT:-${ROOT}/exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/checkpoints/global_step_50/actor/huggingface}"
RENDER_PORT="${RENDER_PORT:-8768}"
AOSS_DIR="${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}"
ADS_CLI="${ADS_CLI:-${AOSS_DIR}/ads-cli}"
CONF="${CONF:-${AOSS_DIR}/petreloss.conf}"
HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
BUCKET="${BUCKET:-baiqiao}"

mkdir -p "${OUT}"/{sco,rgb,render_probe,models,logs,eval,scene_cache}
exec > >(tee -a "${OUT}/sco/worker.log") 2>&1
cd "${ROOT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export NO_PROXY="${NO_PROXY:-*}"
export no_proxy="${no_proxy:-*}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/mnt/umm/users/yinbaiqiao/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export VAGEN_ENV_ROOT="${CONDA_ENV}"
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
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_SKIP_DEEP_GEMM_WARMUP=1
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer}"
export VAGEN_GSPLAT_PREBUILT=1
export AOSS_PLY_CACHE_DIR="${OUT}/scene_cache/ply"
JIT_CACHE_ROOT="/tmp/act2qa_jit_${USER:-root}_$$"
export FLASHINFER_WORKSPACE_BASE="${JIT_CACHE_ROOT}/flashinfer"
export TRITON_CACHE_DIR="${JIT_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JIT_CACHE_ROOT}/torchinductor"
mkdir -p "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${AOSS_PLY_CACHE_DIR}"

renderer_pid=""
cleanup() {
  status=$?
  if [[ -n "${renderer_pid}" ]]; then
    kill "${renderer_pid}" 2>/dev/null || true
    wait "${renderer_pid}" 2>/dev/null || true
  fi
  printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUT}/sco/worker_exit.json"
  exit "${status}"
}
trap cleanup EXIT INT TERM

{
  echo "# RESOURCE_STATUS"
  echo
  echo "- hostname: $(hostname)"
  echo "- utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- user: $(id -un) uid=$(id -u)"
  echo "- cuda: ${CUDA_VISIBLE_DEVICES}"
  echo "- python: ${PY}"
  echo "- gs_root_candidate: ${LOCAL_GS}"
  echo "- scene_id: ${SCENE_ID}"
  echo "- bank: ${BANK_SRC}"
  echo "- out: ${OUT}"
  echo
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || true
} > "${OUT}/RESOURCE_STATUS.md"

{
  echo "hostname=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$(git -c safe.directory=${ROOT} rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)"
  echo "worktree=${ROOT}"
  echo "python=$("${PY}" --version 2>&1)"
  echo "whoami=$(id -un) $(id -u)"
} > "${OUT}/sco/frozen_state.txt"
nvidia-smi > "${OUT}/sco/nvidia_smi.txt" || true

echo "[stage] resolve scene"
GS_ROOT=""
if [[ -r "${LOCAL_GS}/${SCENE_ID}/3dgs_compressed.ply" || -r "${LOCAL_GS}/${SCENE_ID}/gaussian.ply" ]]; then
  GS_ROOT="${LOCAL_GS}"
  echo "[scene] using local InteriorGS ${GS_ROOT} (readable on worker)"
else
  echo "[scene] local InteriorGS not readable; restoring ${SCENE_ID} via ads-cli"
  AK="$(awk '/^\[fj\]/{i=1;next} /^\[/{i=0} i && /^access_key[[:space:]]*=/{sub(/^access_key[[:space:]]*=[[:space:]]*/,""); print; exit}' "${CONF}")"
  SK="$(awk '/^\[fj\]/{i=1;next} /^\[/{i=0} i && /^secret_key[[:space:]]*=/{sub(/^secret_key[[:space:]]*=[[:space:]]*/,""); print; exit}' "${CONF}")"
  SCENE_DEST="${OUT}/scene_cache/InteriorGS/${SCENE_ID}"
  mkdir -p "${SCENE_DEST}"
  set +e
  "${ADS_CLI}" -p 16 -l 4 --conntimeout 120 --timeout 600 \
    sync "s3://${AK}:${SK}@${BUCKET}.${HOST}/InteriorGS/${SCENE_ID}/" "${SCENE_DEST}/" \
    > "${OUT}/logs/scene_restore.log" 2>&1
  scene_rc=$?
  set -e
  echo "scene_restore_rc=${scene_rc}" >> "${OUT}/logs/scene_restore.log"
  if [[ -r "${SCENE_DEST}/3dgs_compressed.ply" || -r "${SCENE_DEST}/gaussian.ply" ]]; then
    GS_ROOT="${OUT}/scene_cache/InteriorGS"
    echo "[scene] restored to ${GS_ROOT}"
  else
    echo "[scene] ads-cli restore incomplete; will try UnifiedRenderGS remote gs_root"
    GS_ROOT="fj:s3://baiqiao/InteriorGS"
  fi
fi
printf '%s\n' "${GS_ROOT}" > "${OUT}/sco/gs_root.txt"

echo "[stage] start localhost HTTP renderer"
set +e
bash "${ROOT}/examples/train/active_spatial/start_gs_render_http_service.sh" \
  --gs-root "${GS_ROOT}" \
  --port "${RENDER_PORT}" \
  --gpus 0 \
  --max-workers 1 \
  --max-inflight 2 \
  --admit-timeout 300 \
  --conda-env "${CONDA_ENV}" \
  > "${OUT}/logs/renderer_service.log" 2>&1 &
renderer_pid=$!
set -e
echo "renderer_pid=${renderer_pid}" > "${OUT}/sco/renderer_pid.txt"

health_ok=0
for _ in $(seq 1 240); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${RENDER_PORT}/health" > "${OUT}/logs/renderer_health.json"; then
    health_ok=1
    break
  fi
  if ! kill -0 "${renderer_pid}" 2>/dev/null; then
    echo "[warn] HTTP renderer exited; falling back to UnifiedRenderGS local backend"
    renderer_pid=""
    break
  fi
  sleep 2
done

RENDER_BACKEND="http"
RENDER_URL="http://127.0.0.1:${RENDER_PORT}/render"
if [[ "${health_ok}" -ne 1 ]]; then
  if [[ -n "${renderer_pid}" ]]; then
    kill "${renderer_pid}" 2>/dev/null || true
    wait "${renderer_pid}" 2>/dev/null || true
    renderer_pid=""
  fi
  RENDER_BACKEND="local"
  RENDER_URL=""
  echo "[renderer] using local backend"
fi
printf '{"backend":"%s","url":"%s","gs_root":"%s"}\n' "${RENDER_BACKEND}" "${RENDER_URL}" "${GS_ROOT}" > "${OUT}/sco/renderer_choice.json"

echo "[stage] render probe n=1"
set +e
"${PY}" "${ROOT}/data_gen/active_spatial_qa/render_paired_bank.py" \
  --bank "${BANK_SRC}" \
  --output-dir "${OUT}/render_probe" \
  --backend "${RENDER_BACKEND}" \
  --renderer-url "${RENDER_URL}" \
  --gs-root "${GS_ROOT}" \
  --gpu-device 0 \
  --width 512 \
  --height 512 \
  --limit 1
probe_rc=$?
set -e
echo "probe_rc=${probe_rc}" | tee "${OUT}/logs/render_probe.rc"
if [[ "${probe_rc}" -ne 0 ]]; then
  echo "[fatal] probe render command failed" >&2
  exit 10
fi
"${PY}" "${ROOT}/data_gen/active_spatial_qa/check_render_probe.py" "${OUT}/render_probe"

echo "[stage] render full bank"
set +e
"${PY}" "${ROOT}/data_gen/active_spatial_qa/render_paired_bank.py" \
  --bank "${BANK_SRC}" \
  --output-dir "${OUT}/rgb" \
  --backend "${RENDER_BACKEND}" \
  --renderer-url "${RENDER_URL}" \
  --gs-root "${GS_ROOT}" \
  --gpu-device 0 \
  --width 512 \
  --height 512
full_rc=$?
set -e
echo "full_rc=${full_rc}" | tee "${OUT}/logs/render_full.rc"

echo "[stage] freeze eligible bank"
"${PY}" "${ROOT}/data_gen/active_spatial_qa/finalize_act2qa.py" freeze \
  --rendered-bank "${OUT}/rgb/manifest.jsonl" \
  --errors "${OUT}/rgb/errors.jsonl" \
  --output-dir "${OUT}"

echo "[stage] stop renderer before VLM load"
if [[ -n "${renderer_pid}" ]]; then
  kill "${renderer_pid}" 2>/dev/null || true
  wait "${renderer_pid}" 2>/dev/null || true
  renderer_pid=""
  echo "[renderer] stopped"
fi
sleep 3
nvidia-smi > "${OUT}/sco/nvidia_smi_after_renderer.txt" || true

echo "[stage] resolve Act checkpoint"
ACT_DIR="${OUT}/models/act_qwen_v50_step50"
mkdir -p "${ACT_DIR}"
act_ready=0
if [[ -r "${ACT_SRC_DEFAULT}/config.json" && -r "${ACT_SRC_DEFAULT}/model.safetensors.index.json" ]]; then
  echo "[act] original v50 step50 is readable on worker; copying into owned dir"
  set +e
  cp -a "${ACT_SRC_DEFAULT}/." "${ACT_DIR}/"
  cp_rc=$?
  set -e
  echo "copy_rc=${cp_rc}" > "${OUT}/logs/act_copy.rc"
  if [[ -r "${ACT_DIR}/config.json" ]]; then act_ready=1; fi
fi
if [[ "${act_ready}" -ne 1 ]]; then
  echo "[act] restoring v50 step50 via ads-cli into ${ACT_DIR}"
  AK="$(awk '/^\[fj\]/{i=1;next} /^\[/{i=0} i && /^access_key[[:space:]]*=/{sub(/^access_key[[:space:]]*=[[:space:]]*/,""); print; exit}' "${CONF}")"
  SK="$(awk '/^\[fj\]/{i=1;next} /^\[/{i=0} i && /^secret_key[[:space:]]*=/{sub(/^secret_key[[:space:]]*=[[:space:]]*/,""); print; exit}' "${CONF}")"
  set +e
  "${ADS_CLI}" -p 32 -l 8 --conntimeout 120 --timeout 600 \
    sync "s3://${AK}:${SK}@${BUCKET}.${HOST}/VAGEN-Lite/exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/checkpoints/global_step_50/actor/huggingface/" \
    "${ACT_DIR}/" > "${OUT}/logs/act_restore.log" 2>&1
  act_rc=$?
  set -e
  echo "act_restore_rc=${act_rc}" >> "${OUT}/logs/act_restore.log"
  if [[ -r "${ACT_DIR}/config.json" && -e "${ACT_DIR}/model.safetensors.index.json" ]]; then
    act_ready=1
  fi
fi
if [[ "${act_ready}" -ne 1 ]]; then
  echo "[act] v50 step50 unrestored; leaving identity as UNRESOLVED"
  ACT_DIR=""
fi

"${PY}" "${ROOT}/data_gen/active_spatial_qa/finalize_act2qa.py" identities \
  --output-dir "${OUT}" \
  --base "${BASE_MODEL}" \
  --act "${ACT_DIR}" \
  --act-source "${ACT_SRC_DEFAULT}" \
  --train-config "${ROOT}/exps/vagen_active_spatial/qwen_v50_clean_d0pass_20260824_full/hydra_run/.hydra/config.yaml"

FROZEN_BANK="${OUT}/frozen_visual_manifest.jsonl"
if [[ ! -s "${FROZEN_BANK}" ]]; then
  echo "[fatal] no frozen eligible bank" >&2
  exit 12
fi

echo "[stage] Base QA inference"
set +e
"${PY}" "${ROOT}/data_gen/active_spatial_qa/model_qa_eval.py" \
  --bank "${FROZEN_BANK}" \
  --checkpoint "${BASE_MODEL}" \
  --output "${OUT}/eval/base_predictions.jsonl" \
  --tp 1 \
  --gpu-memory-utilization 0.85
base_rc=$?
set -e
echo "base_rc=${base_rc}" | tee "${OUT}/logs/base_infer.rc"

echo "[stage] Act QA inference"
if [[ -n "${ACT_DIR}" && -r "${ACT_DIR}/config.json" ]]; then
  set +e
  "${PY}" "${ROOT}/data_gen/active_spatial_qa/model_qa_eval.py" \
    --bank "${FROZEN_BANK}" \
    --checkpoint "${ACT_DIR}" \
    --output "${OUT}/eval/act_predictions.jsonl" \
    --tp 1 \
    --gpu-memory-utilization 0.85
  act_inf_rc=$?
  set -e
  echo "act_infer_rc=${act_inf_rc}" | tee "${OUT}/logs/act_infer.rc"
else
  echo "[act] skipped inference; checkpoint unresolved" | tee "${OUT}/logs/act_infer.rc"
fi

echo "[stage] evaluate and write table"
"${PY}" "${ROOT}/data_gen/active_spatial_qa/qa_eval.py" \
  --bank "${FROZEN_BANK}" \
  --predictions "${OUT}/eval/base_predictions.jsonl" \
  --output "${OUT}/eval/base_qa_eval.json" \
  --require-observable || true
if [[ -s "${OUT}/eval/act_predictions.jsonl" ]]; then
  "${PY}" "${ROOT}/data_gen/active_spatial_qa/qa_eval.py" \
    --bank "${FROZEN_BANK}" \
    --predictions "${OUT}/eval/act_predictions.jsonl" \
    --output "${OUT}/eval/act_qa_eval.json" \
    --require-observable || true
fi
"${PY}" "${ROOT}/data_gen/active_spatial_qa/finalize_act2qa.py" table \
  --output-dir "${OUT}" \
  --bank "${FROZEN_BANK}" \
  --base-pred "${OUT}/eval/base_predictions.jsonl" \
  --act-pred "${OUT}/eval/act_predictions.jsonl"

echo "[done] artifacts at ${OUT}"
