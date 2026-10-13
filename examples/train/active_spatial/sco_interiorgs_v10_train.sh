#!/usr/bin/env bash
# InteriorGS v10 training jobs (2026-10-14), run from the separate work tree VAGEN-Lite-r4.
#
#   sft-trial  RUN LF_DIR [MAX_STEPS]   1 card : LLaMA-Factory through accelerate + FSDP (the formal path, 1 process)
#   sft        RUN LF_DIR               8 cards: the formal SFT (FSDP, full fine-tune, vision tower frozen)
#   rl-trial   RUN RL_DIR [ENDPOINT]    1 card : the 3DGS render service and PPO on the same card
#   renderer   RUN                      1 card : the render service for a formal RL run (writes RUN/renderer_endpoint.txt)
#   rl         RUN RL_DIR ENDPOINT      8 cards: PPO, waits for RUN/renderer_endpoint.txt
#
# Environment setup as in sco_dense_score_reward_only_parallel.sh. sco jobs run as root: outputs are handed
# back to the owner of RUN on exit.
set -euo pipefail
MODE=${1:?mode}
RUN=${2:?run directory}
WORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-r4
ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV_ROOT}/bin/python
GS_ROOT=/mnt/umm/users/yinbaiqiao/InteriorGS_ply
VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/torch_extensions/gsplat_cuda
mkdir -p "${RUN}/logs"
OWNER=$(stat -c %u:%g "${RUN}")
trap 'status=$?; [[ -n ${renderer_pid:-} ]] && kill "${renderer_pid}" 2>/dev/null; chown -R "${OWNER}" "${RUN}" 2>/dev/null || true; printf "{\"mode\":\"%s\",\"exit_status\":%s,\"ended_utc\":\"%s\"}\n" "${MODE}" "${status}" "$(date -u +%FT%TZ)" > "${RUN}/logs/${MODE}_exit.json"; exit ${status}' EXIT
exec > >(tee -a "${RUN}/logs/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "${WORKTREE}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export PYTHONPATH="${WORKTREE}/verl:${WORKTREE}/scripts:${WORKTREE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=${RUN}/torch_extensions_${MODE}
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2 INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10
WHEEL_ROOT=${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=${ENV_ROOT} CURAND_INCLUDE_DIR=${WHEEL_ROOT}/include CURAND_LIBRARY_DIR=${WHEEL_ROOT}/lib
export PATH="${CUDA_HOME}/bin:${ENV_ROOT}/bin:${PATH}"
export CPATH="${CURAND_INCLUDE_DIR}:${ENV_ROOT}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=1
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
git -C "${WORKTREE}" rev-parse HEAD 2>/dev/null || cat "${WORKTREE}/.synced_commit" 2>/dev/null || true

start_renderer() {  # $1 = gpu index; leaves renderer_pid and R1_RENDER_URL
  mkdir -p "${TORCH_EXTENSIONS_DIR}"
  [[ -d ${TORCH_EXTENSIONS_DIR}/gsplat_cuda ]] || cp -a "${VALIDATED_GSPLAT_CACHE}" "${TORCH_EXTENSIONS_DIR}/gsplat_cuda"
  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${GS_ROOT}" --host 0.0.0.0 --port 8915 --gpus "$1" \
    --max-workers 12 --max-inflight 12 --admit-timeout 900 --conda-env "${ENV_ROOT}" \
    > "${RUN}/logs/renderer_service.log" 2>&1 &
  renderer_pid=$!
  for _ in $(seq 1 180); do
    "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)' 2>/dev/null && break
    kill -0 "${renderer_pid}"
    sleep 2
  done
  "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)'
  export R1_RENDER_URL=http://127.0.0.1:8915/render
}

lf_train() {  # $1 = LLaMA-Factory recipe, $2 = number of processes
  accelerate launch --config_file examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml \
    --num_processes "$2" -m llamafactory.launcher "$1"
}

case ${MODE} in
  sft-trial)
    LF=${3:?LLaMA-Factory directory}
    STEPS=${4:-30}
    export CUDA_VISIBLE_DEVICES=0
    "${PY}" - "${LF}/train.yaml" "${RUN}/sft_trial.yaml" "${RUN}/model" "${STEPS}" <<'PY'
import sys, yaml
src, dst, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
c = yaml.safe_load(open(src))
c.update(output_dir=out, max_steps=steps, save_steps=steps, eval_steps=steps, logging_steps=1)
open(dst, "w").write(yaml.safe_dump(c, sort_keys=False))
PY
    lf_train "${RUN}/sft_trial.yaml" 1
    ;;
  sft)
    LF=${3:?LLaMA-Factory directory}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    lf_train "${LF}/train.yaml" 8
    ;;
  rl-trial)
    RL=${3:?RL config directory}
    ENDPOINT=${4:-2}
    export CUDA_VISIBLE_DEVICES=0
    start_renderer 0
    export V10_RAY_TEMP_DIR=/tmp/v10_ray_$$
    "${PY}" -m vagen.interiorgs_v10_ppo --config "${RL}/ppo.yaml" --endpoint "${ENDPOINT}"
    ;;
  renderer)
    export CUDA_VISIBLE_DEVICES=0
    start_renderer 0
    printf 'http://%s:8915/render\n' "$(hostname -I | awk '{print $1}')" > "${RUN}/renderer_endpoint.tmp"
    mv "${RUN}/renderer_endpoint.tmp" "${RUN}/renderer_endpoint.txt"
    while [[ ! -e ${RUN}/STOP_RENDERER ]]; do kill -0 "${renderer_pid}"; sleep 10; done
    ;;
  rl)
    RL=${3:?RL config directory}
    ENDPOINT=${4:?endpoint}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    for _ in $(seq 1 720); do [[ -s ${RUN}/renderer_endpoint.txt ]] && break; sleep 5; done
    export R1_RENDER_URL
    R1_RENDER_URL=$(<"${RUN}/renderer_endpoint.txt")
    export V10_RAY_TEMP_DIR=/tmp/v10_ray_$$
    "${PY}" -m vagen.interiorgs_v10_ppo --config "${RL}/ppo.yaml" --endpoint "${ENDPOINT}"
    touch "${RUN}/STOP_RENDERER"
    ;;
  *)
    echo "unknown mode ${MODE}" >&2
    exit 2
    ;;
esac
