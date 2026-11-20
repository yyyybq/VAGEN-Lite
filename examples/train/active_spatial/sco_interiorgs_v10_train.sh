#!/usr/bin/env bash
# InteriorGS v10 training jobs (2026-10-14), run from the cluster checkout /mnt/umm/users/yinbaiqiao/VAGEN-Lite
# (since round 6 the only code directory; it is checked against main by git hash before a launch).
#
#   sft-trial  RUN LF_DIR [MAX_STEPS]   1 card : LLaMA-Factory through accelerate + FSDP (the formal path, 1 process)
#   sft        RUN LF_DIR               8 cards: the formal SFT (FSDP, full fine-tune, vision tower frozen)
#   rl-trial   RUN RL_DIR [ENDPOINT]    1 card : the 3DGS render service and PPO on the same card
#   renderer   RUN [WORKERS [ENDPOINT_NAME]]  1 card : the render service for a formal RL run (writes
#                                        RUN/renderer_endpoint.txt, or RUN/ENDPOINT_NAME); WORKERS worker processes and
#                                        requests at a time (default 12); RENDERER_GPU / RENDERER_PORT (default 0 /
#                                        8915): the card and port, for several services on one machine
#   rl         RUN RL_DIR ENDPOINT [WAIT_S]  8 cards: PPO, waits up to WAIT_S (default 3600) s for RUN/renderer_endpoint.txt
#   eval       RUN EVAL NAME MODEL [PROMPT_FORMAT|- [OPTION ...]]  1 card: the render service and vLLM; one greedy
#                                        episode per task of EVAL/tasks.jsonl (data_gen/interiorgs_v10/evaluation.py),
#                                        results in EVAL/NAME; OPTIONs go to "evaluation.py run" (round 9 switches,
#                                        e.g. --check-each-action --max-actions 20 --max-turns 20); "-": the default
#                                        prompt format.  VAGEN_LITE_EVAL_CARDS (q38; default 1): the job's cards; with
#                                        2 the render service keeps card 0 and vLLM gets card 1 to itself
#   rl-resume-check RUN RL_DIR          1 card : a trial config (rl-config --trial) run to update 2, then a second
#                                        process that must resume from update 2 and stop after 3; checks that and
#                                        the retention rule (kept: updates 2 and 3) in RUN/logs/resume_check.json
#   distance-tables RUN ARG ...         1 card (its 14 vCPUs; the GPU is not used): round 13's per-task distance
#                                        tables, "python -m data_gen.interiorgs_v10.distance_table build --gs-root
#                                        <GS_ROOT> ARG ..." (e.g. --rows R --out D --shard 0/8 --workers 13)
#
# Environment setup as in sco_dense_score_reward_only_parallel.sh. sco jobs run as root: outputs are handed
# back to the owner of RUN on exit (and the SFT model / evaluation results, written outside RUN).
#
# q38 (the Qwen3.8-27B rerun line, DESIGN_ASYNC 2.2 and question 6; every variable below is passed on by job_sco
# PASS_ENV, and unset leaves the job as before):
#   TRAIN_ENV_ROOT   the Python environment of the trainer, the evaluation and the QA evaluation (q38:
#                    /mnt/umm/users/yinbaiqiao/q38/envs/verl, torch cu130 + vLLM 0.28); the render service and the rest
#                    of this script's environment stay in RENDER_ENV_ROOT (vagen-lite).  Unset: vagen-lite for both
#   V10_PPO_MODULE   the trainer's entry module (q38: vagen.q38.ppo_v1); unset: vagen.interiorgs_v10_ppo
#   Q38_CUDA_COMPAT  only an override: with a separate TRAIN_ENV_ROOT the job puts NVIDIA's CUDA 13.0 forward-
#                    compatibility libraries first on the trainer's library path by itself when the node's driver
#                    (nvidia-smi) is older than 580, as zoetrope's 550.90.07 (unset or auto); 1: always; 0: never
set -euo pipefail
MODE=${1:?mode}
RUN=${2:?run directory}
# round 12 freeze: VAGEN_LITE_WORKTREE (set by the launch script of a chain run from another copy, job_sco.WORKTREE)
WORKTREE=${VAGEN_LITE_WORKTREE:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}
RENDER_ENV_ROOT=${RENDER_ENV_ROOT:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite}
TRAIN_ENV_ROOT=${TRAIN_ENV_ROOT:-${RENDER_ENV_ROOT}}
ENV_ROOT=${RENDER_ENV_ROOT}  # the render service's env, and the CUDA toolkit / compilers of the exports below
RENDER_PY=${RENDER_ENV_ROOT}/bin/python
# the model line (2026-10-24): VAGEN_LITE_PYTHON is the model type's Python (vagen/models/model_types.py, e.g. the
# Qwen3-VL runtime venv), passed on by job_sco; unset, TRAIN_ENV_ROOT's (vagen-lite unless q38 sets it).  The render
# service keeps RENDER_ENV_ROOT
PY=${VAGEN_LITE_PYTHON:-${TRAIN_ENV_ROOT}/bin/python}
PPO_MODULE=${V10_PPO_MODULE:-vagen.interiorgs_v10_ppo}
# the container's own search paths, before the exports below put the render env's CUDA 12 toolkit on them
# (use_train_env starts the trainer's paths from these)
BASE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} BASE_LIBRARY_PATH=${LIBRARY_PATH:-} BASE_CPATH=${CPATH:-}
GS_ROOT=/mnt/umm/users/yinbaiqiao/InteriorGS_ply
# the gsplat build validated on 2026-10-07 (dense-score pilot), kept outside the code directory since round 11: the
# checkout's experiment folders get moved into archive/ by the directory clean-up
VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/gsplat_cuda_validated_20261007
# q38: logs, configs and checkpoints readable from the login node while the job runs (the chain reads the trainer's
# log every minute); pear's own sco entry set umask 077, so its train.log and hydra/ were root-only (PEAR m9)
umask 022
mkdir -p "${RUN}/logs"
OWNER=$(stat -c %u:%g "${RUN}")
# the exit record is written before ownership is handed back, so it is not left owned by root
trap 'status=$?; [[ -n ${renderer_pid:-} ]] && kill "${renderer_pid}" 2>/dev/null; [[ -n ${gpu_log_pid:-} ]] && kill "${gpu_log_pid}" 2>/dev/null; printf "{\"mode\":\"%s\",\"exit_status\":%s,\"ended_utc\":\"%s\"}\n" "${MODE}" "${status}" "$(date -u +%FT%TZ)" > "${RUN}/logs/${MODE}_exit.json"; chown -R "${OWNER}" "${RUN}" ${ALSO_OWNED:+"${ALSO_OWNED}"} 2>/dev/null || true; exit ${status}' EXIT
exec > >(tee -a "${RUN}/logs/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "${WORKTREE}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
# q38: the offline wandb files under RUN, which is handed back on exit (unset, wandb writes WORKTREE/wandb as root,
# never handed back: /mnt/umm/users/yinbaiqiao/VAGEN-Lite/wandb is root's); and USER, which sco's root container does
# not set (pear's r5 ended on os.environ['USER'], PEAR m9)
export WANDB_DIR=${WANDB_DIR:-${RUN}}
export USER=${USER:-$(id -un 2>/dev/null || echo root)}
export PYTHONPATH="${WORKTREE}/verl:${WORKTREE}/scripts:${WORKTREE}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Ray answers the first driver's registration only once every worker it prestarts for it (one a CPU, 112 on 8
# cards, all importing from /mnt/umm at once) has registered: the QA RL smoke's trainer hung in ray.init twice on
# 2026-11-04 (the driver in RegisterClient, 15 workers up); without the prestart, workers start on demand
export RAY_prestart_worker_first_driver=0
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
# the model line (2026-10-25): with VAGEN_LITE_GPU_LOG set (plan "gpu_log"), each card's memory every 5 s in
# RUN/logs/gpu_mem_<mode>.csv, stopped on exit (the smoke reports' per-card peaks); unset: as before
if [[ -n ${VAGEN_LITE_GPU_LOG:-} ]]; then
  nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 5000 \
    > "${RUN}/logs/gpu_mem_${MODE}.csv" 2>/dev/null &
  gpu_log_pid=$!
fi

start_renderer() {  # $1 = gpu index, $2 = worker processes = requests at a time (default 12); leaves renderer_pid and R1_RENDER_URL
  local workers=${2:-12} port=${RENDERER_PORT:-8915}  # RENDERER_PORT: several services on one machine
  mkdir -p "${TORCH_EXTENSIONS_DIR}"
  [[ -d ${TORCH_EXTENSIONS_DIR}/gsplat_cuda ]] || cp -a "${VALIDATED_GSPLAT_CACHE}" "${TORCH_EXTENSIONS_DIR}/gsplat_cuda"
  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${GS_ROOT}" --host 0.0.0.0 --port "${port}" --gpus "$1" \
    --max-workers "${workers}" --max-inflight "${workers}" --admit-timeout 900 --conda-env "${RENDER_ENV_ROOT}" \
    > "${RENDERER_SERVICE_LOG:-${RUN}/logs/renderer_service.log}" 2>&1 &
  renderer_pid=$!
  for _ in $(seq 1 180); do  # q38: the render env's Python asks, whatever PY is
    "${RENDER_PY}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/health', timeout=2)" 2>/dev/null && break
    kill -0 "${renderer_pid}"
    sleep 2
  done
  "${RENDER_PY}" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${port}/health', timeout=2)"
  export R1_RENDER_URL=http://127.0.0.1:${port}/render
}

eval_policy_cards() {
  # q38: the cards of an eval job's vLLM.  One card (VAGEN_LITE_EVAL_CARDS unset or 1): card 0, shared with the render
  # service, as before.  N > 1 (the 27B model, plan "eval_cards" 2): cards 1..N-1, card 0 left to the render service
  # (SMOKE_PLAN 4: GPU0 the render service, GPU1 vLLM with TP 1; 55.6 GB of bf16 weights do not fit beside it)
  local n=${VAGEN_LITE_EVAL_CARDS:-1}
  if [[ ! ${n} =~ ^[1-9][0-9]*$ ]]; then
    echo "[sco_interiorgs_v10_train] VAGEN_LITE_EVAL_CARDS must be a number of cards, not '${n}'" >&2
    return 1
  fi
  if (( n == 1 )); then echo 0; else seq -s, 1 $(( n - 1 )); fi
}

use_train_env() {
  # q38: from here on this job's commands run in TRAIN_ENV_ROOT's environment (the render service started before
  # keeps the render env's).  Without a separate TRAIN_ENV_ROOT nothing changes.  The trainer's environment:
  # - CUDA_HOME: TRAIN_ENV_ROOT's own CUDA 13 toolkit (the pip nvidia/cu13 wheels: nvcc for FlashInfer's JIT; the
  #   q38 flash-attn build, q38/jobs/q38-fa2-build2/build.sh, used the same); TRAIN_CUDA_HOME overrides it;
  # - LD_LIBRARY_PATH / LIBRARY_PATH / CPATH: the container's own, none of the render env's CUDA 12 directories
  #   (libcurand.so.10, libcusparse.so.12, ... have the same names in both CUDA versions and would come first);
  # - an unversioned libcuda.so for "-lcuda" (FlashInfer's and Triton's JIT link with the conda gcc, whose ld does not
  #   search the host's driver directory; pear's fix, embodied/pear_sco/stage1_4x8_entry.sh:32-59; pear's r4 ended
  #   there), on LIBRARY_PATH and as TRITON_LIBCUDA_PATH;
  # - NVIDIA's CUDA 13.0 forward-compatibility libraries (cuda-compat-13-0 580.196.07) first on LD_LIBRARY_PATH when
  #   the node's driver is older than 580 (nvidia-smi; Q38_CUDA_COMPAT at the top overrides).  On a zoetrope H800
  #   (driver 550.90.07) torch cu130 sees no GPU without them ("The NVIDIA driver on your system is too old (found
  #   version 12040)") and runs a matmul with them (the probe job, q38/jobs/q38-fa2-build2/build.log, 2026-11-18); a
  #   driver of 580 or newer must not get them (forward compatibility only serves older drivers);
  # - CC / CXX: conda-forge gcc 12.4 of the probe-spatial env (only read; TRAIN_CC / TRAIN_CXX override).  The job
  #   image has no gcc, and the trainer's Triton kernels (fla's gated delta rule) need a C compiler: the probe job's
  #   fla kernel ended on "Failed to find C compiler. Please specify via CC environment variable" (the same log).  The
  #   q38 flash-attn 2.8.3.post1 was built with this gcc and nvcc 13.0 (q38/jobs/q38-fa2-build2/build.sh).
  [[ ${TRAIN_ENV_ROOT} == "${RENDER_ENV_ROOT}" ]] && return 0
  local cuda_home=${TRAIN_CUDA_HOME:-}
  if [[ -z ${cuda_home} ]]; then
    cuda_home=$(ls -d "${TRAIN_ENV_ROOT}"/lib/python3*/site-packages/nvidia/cu13 2>/dev/null | head -n 1 || true)
  fi
  [[ -x ${cuda_home}/bin/nvcc ]] || echo "[sco_interiorgs_v10_train] warning: no nvcc in '${cuda_home}' (TRAIN_CUDA_HOME); a JIT build will fail"
  local cbin=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin
  local cc=${TRAIN_CC:-${cbin}/x86_64-conda-linux-gnu-gcc} cxx=${TRAIN_CXX:-${cbin}/x86_64-conda-linux-gnu-g++}
  [[ -x ${cc} && -x ${cxx} ]] || echo "[sco_interiorgs_v10_train] warning: no C / C++ compiler at '${cc}' / '${cxx}' (TRAIN_CC / TRAIN_CXX); Triton's kernels will fail"
  local compat=${Q38_CUDA_COMPAT:-auto} compat_below=580  # 580: the compatibility libraries' driver branch
  local compat_dir=${Q38_CUDA_COMPAT_DIR:-/mnt/umm/users/yinbaiqiao/q38/envs/cuda-compat/13-0/usr/local/cuda-13.0/compat}
  if [[ ${compat} == auto ]]; then
    local driver
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)
    compat=0
    if [[ ${driver%%.*} =~ ^[0-9]+$ ]]; then
      if (( ${driver%%.*} < compat_below )); then compat=1; fi
    else
      echo "[sco_interiorgs_v10_train] warning: no driver version from nvidia-smi; the CUDA 13 compatibility libraries" \
        "are not added (Q38_CUDA_COMPAT=1 adds them)"
    fi
    echo "[sco_interiorgs_v10_train] driver ${driver:-unknown} (compatibility libraries below ${compat_below}):" \
      "Q38_CUDA_COMPAT=auto -> ${compat}"
  fi
  local ld=${BASE_LD_LIBRARY_PATH} link_dir=
  if [[ ${compat} == 1 ]]; then
    if [[ ! -e ${compat_dir}/libcuda.so.1 ]]; then  # torch cu130 would see no GPU on such a node
      echo "[sco_interiorgs_v10_train] the CUDA 13 compatibility libraries are needed but ${compat_dir} has no libcuda.so.1" >&2
      exit 1
    fi
    ld=${compat_dir}${ld:+:${ld}}
    link_dir=${compat_dir}  # libcuda.so -> libcuda.so.1 there
  else
    link_dir=${TMPDIR:-/tmp}/q38_cuda_link_$$
    mkdir -p "${link_dir}"
    local cand
    for cand in /usr/lib/x86_64-linux-gnu/libcuda.so /lib/x86_64-linux-gnu/libcuda.so /usr/lib64/libcuda.so \
        /usr/local/nvidia/lib64/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so.1 \
        /usr/lib64/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so.1; do
      if [[ -e ${cand} ]]; then
        ln -sfn "${cand}" "${link_dir}/libcuda.so"
        ln -sfn "${cand}" "${link_dir}/libcuda.so.1"  # Triton links -l:libcuda.so.1 (triton/backends/nvidia/driver.py:16)
        break
      fi
    done
    if [[ ! -e ${link_dir}/libcuda.so ]]; then
      echo "[sco_interiorgs_v10_train] warning: no libcuda found; a JIT build linking -lcuda will fail"
      link_dir=
    fi
  fi
  export CUDA_HOME=${cuda_home} CC=${cc} CXX=${cxx}
  export PATH="${cuda_home:+${cuda_home}/bin:}${TRAIN_ENV_ROOT}/bin:${PATH}"
  if [[ -n ${ld} ]]; then export LD_LIBRARY_PATH=${ld}; else unset LD_LIBRARY_PATH; fi
  if [[ -n ${link_dir} ]]; then
    export LIBRARY_PATH="${link_dir}${BASE_LIBRARY_PATH:+:${BASE_LIBRARY_PATH}}" TRITON_LIBCUDA_PATH=${link_dir}
  elif [[ -n ${BASE_LIBRARY_PATH} ]]; then
    export LIBRARY_PATH=${BASE_LIBRARY_PATH}
  else
    unset LIBRARY_PATH
  fi
  if [[ -n ${BASE_CPATH} ]]; then export CPATH=${BASE_CPATH}; else unset CPATH; fi
  unset CURAND_INCLUDE_DIR CURAND_LIBRARY_DIR
  echo "[sco_interiorgs_v10_train] train env ${TRAIN_ENV_ROOT}: CUDA_HOME=${CUDA_HOME} CC=${CC}" \
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} LIBRARY_PATH=${LIBRARY_PATH:-} PY=${PY} PPO_MODULE=${PPO_MODULE}"
}

lf_train() {  # $1 = LLaMA-Factory recipe, $2 = number of processes
  # LLaMA-Factory 0.9.3 pins transformers <= 4.52.4; vagen-lite has 4.56.1 (needed by the RL stack). The
  # trial checks that SFT runs anyway; the pin is LLaMA-Factory's own check, not a known incompatibility.
  export DISABLE_VERSION_CHECK=1
  if [[ -n ${VAGEN_LITE_MODEL_TYPE:-} && ${VAGEN_LITE_MODEL_TYPE} != qwen2_5_vl ]]; then
    # the model line: the model type's FSDP config and LLaMA-Factory launcher (its registrations first), in its Python
    local cfg launcher
    read -r cfg launcher < <("${PY}" -m vagen.models.model_types sft-launch "${VAGEN_LITE_MODEL_TYPE}")
    "${PY}" -m accelerate.commands.launch --config_file "${cfg}" --num_processes "$2" -m "${launcher}" "$1"
    return
  fi
  accelerate launch --config_file examples/train/active_spatial/accelerate_fsdp_qwen25vl.yaml \
    --num_processes "$2" -m llamafactory.launcher "$1"
}

case ${MODE} in
  sft-trial)
    LF=${3:?LLaMA-Factory directory}
    ALSO_OWNED=${LF}  # the model is written to LF/model, outside RUN
    STEPS=${4:-30}
    export CUDA_VISIBLE_DEVICES=0
    use_train_env  # q38: the SFT in TRAIN_ENV_ROOT's environment (nothing changes without a separate one)
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
    ALSO_OWNED=${LF}  # the model is written to LF/model, outside RUN
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    # q38 (2026-11-20): Qwen3.8-27B's SFT runs in the q38 env (torch cu130): without use_train_env a zoetrope node
    # (driver 550) has no CUDA 13 compatibility libraries (torch sees no GPU), and fla's Triton kernels of the gated
    # delta net layers have no C compiler / libcuda link (the same reasons as the RL modes)
    use_train_env
    lf_train "${LF}/train.yaml" 8
    ;;
  rl-trial)
    RL=${3:?RL config directory}
    ENDPOINT=${4:-2}
    export CUDA_VISIBLE_DEVICES=0
    start_renderer 0
    use_train_env
    export V10_RAY_TEMP_DIR=/tmp/v10_ray_$$
    "${PY}" -m "${PPO_MODULE}" --config "${RL}/ppo.yaml" --endpoint "${ENDPOINT}"
    ;;
  rl-resume-check)
    RL=${3:?RL config directory}
    # q38: the check below reads the 0.6.1 trainer's resume line ("Setting global step to 2", verl's legacy
    # ray_trainer.py); the V1 trainer (V10_PPO_MODULE vagen.q38.ppo_v1) prints none, so the check would fail after
    # two trainer runs.  The q38 resume check is the smoke S3 of DESIGN_ASYNC 5.2
    if [[ ${PPO_MODULE} != vagen.interiorgs_v10_ppo ]]; then
      echo "rl-resume-check checks the 0.6.1 trainer only, not ${PPO_MODULE}" >&2
      exit 2
    fi
    export CUDA_VISIBLE_DEVICES=0
    start_renderer 0
    use_train_env
    export V10_RAY_TEMP_DIR=/tmp/v10_ray_$$
    "${PY}" -m "${PPO_MODULE}" --config "${RL}/ppo.yaml" --endpoint 2
    "${PY}" -m "${PPO_MODULE}" --config "${RL}/ppo.yaml" --endpoint 3 2>&1 | tee "${RUN}/logs/resume_part2.log"
    "${PY}" - "${RL}/ppo.yaml" "${RUN}/logs" <<'PY'
import json, pathlib, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
root = pathlib.Path(cfg["trainer"]["default_local_dir"])
steps = sorted(int(p.name.split("global_step_")[-1]) for p in root.glob("global_step_*"))
log = (pathlib.Path(sys.argv[2]) / "resume_part2.log").read_text(errors="ignore")
result = {"checkpoints": steps, "resumed_from_update_2": "Setting global step to 2" in log,
          "expected_checkpoints": [2, 3]}
result["ok"] = result["resumed_from_update_2"] and steps == [2, 3]
(pathlib.Path(sys.argv[2]) / "resume_check.json").write_text(json.dumps(result, indent=1))
print(json.dumps(result))
assert result["ok"], result
PY
    ;;
  eval)
    EVAL=${3:?evaluation folder}
    NAME=${4:?model name}
    ALSO_OWNED=${EVAL}/${NAME}  # the results folder, outside RUN
    MODEL=${5:?Hugging Face model folder}
    PROMPT_FORMAT=${6:-}
    [[ ${PROMPT_FORMAT} == - ]] && PROMPT_FORMAT=
    EVAL_OPTIONS=("${@:7}")
    export CUDA_VISIBLE_DEVICES=0
    start_renderer 0
    use_train_env
    CUDA_VISIBLE_DEVICES=$(eval_policy_cards)  # q38: card 1 with two cards; card 0 (with the render service) with one
    "${PY}" -m data_gen.interiorgs_v10.evaluation run --eval "${EVAL}" --name "${NAME}" --model "${MODEL}" \
      ${PROMPT_FORMAT:+--prompt-format "${PROMPT_FORMAT}"} ${EVAL_OPTIONS[@]+"${EVAL_OPTIONS[@]}"}
    ;;
  qa)
    # Qwen3 main line (decision 2 of 2026-11-01): a QA evaluation of one model (plan "qa_eval", train_chain.qa_jobs),
    # data_gen.interiorgs_v10.qa_eval in the model type's Python; $6 its command (easi), then its options
    OUT=${3:?benchmark output folder}
    NAME=${4:?model name}
    ALSO_OWNED=${OUT}/${NAME}  # the results folder, outside RUN
    MODEL=${5:?Hugging Face model folder}
    export CUDA_VISIBLE_DEVICES=0  # q38: one card still (qa_eval hands easi_eval the first visible card only)
    use_train_env
    "${PY}" -m data_gen.interiorgs_v10.qa_eval "${6:?qa_eval command}" --name "${NAME}" --model "${MODEL}" \
      --out "${OUT}" "${@:7}"
    ;;
  renderer)
    # round 12 freeze (the user's decision of 2026-10-24): RENDERER_GPU / RENDERER_PORT (default 0 / 8915) let
    # several services share a machine outside sco (train_chain's renderer_external.json)
    export CUDA_VISIBLE_DEVICES=${RENDERER_GPU:-0}
    # round 12 batch: the chain's plan rl "renderer_workers" passes more worker processes (one scene each), so the
    # scenes of an update stay loaded; default 12.  ENDPOINT_NAME (the chain's renderer_repack): a second service,
    # started before the first one is deleted, writes its own address file (the chain renames it once the move is
    # decided) and its own service log
    ENDPOINT_NAME=${4:-renderer_endpoint.txt}
    [[ ${ENDPOINT_NAME} == renderer_endpoint.txt ]] ||
      RENDERER_SERVICE_LOG=${RUN}/logs/renderer_service-$(date -u +%Y%m%dT%H%M%SZ).log
    start_renderer "${RENDERER_GPU:-0}" "${3:-12}"
    printf 'http://%s:%s/render\n' "$(hostname -I | awk '{print $1}')" "${RENDERER_PORT:-8915}" \
      > "${RUN}/${ENDPOINT_NAME%.txt}.tmp"
    mv "${RUN}/${ENDPOINT_NAME%.txt}.tmp" "${RUN}/${ENDPOINT_NAME}"
    while [[ ! -e ${RUN}/STOP_RENDERER ]]; do kill -0 "${renderer_pid}"; sleep 10; done
    ;;
  rl)
    RL=${3:?RL config directory}
    ENDPOINT=${4:?endpoint}
    # round 12 batch: the chain's renderer_when_trainer_runs switch submits the render service once this job has
    # its cards and passes a longer wait; default an hour
    RENDERER_WAIT_S=${5:-3600}
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    for _ in $(seq 1 $((RENDERER_WAIT_S / 5))); do [[ -s ${RUN}/renderer_endpoint.txt ]] && break; sleep 5; done
    export R1_RENDER_URL
    R1_RENDER_URL=$(<"${RUN}/renderer_endpoint.txt")
    use_train_env
    # Ray's raylet has 30 s (hard-coded in this Ray) to register with the GCS; on a slow shared disk it may miss
    # them and the trainer ends at once ("timed out during startup": the QA RL smoke, 2026-11-04 11:12, after five
    # hours in the queue).  Such a start is tried again in this job, up to RAY_START_TRIES (default 3) times, instead
    # of the chain resubmitting it to the back of the queue; a trainer that reached an update, or ended otherwise,
    # ends the job as before
    tries=${RAY_START_TRIES:-3}
    for try in $(seq 1 "${tries}"); do
      export V10_RAY_TEMP_DIR=/tmp/v10_ray_$$_${try}
      try_log=/tmp/v10_rl_try_$$_${try}.log
      set +e
      "${PY}" -m "${PPO_MODULE}" --config "${RL}/ppo.yaml" --endpoint "${ENDPOINT}" 2>&1 | tee "${try_log}"
      status=${PIPESTATUS[0]}
      set -e
      [[ ${status} -eq 0 ]] && break
      # q38 (2026-11-18): the three markers are still Ray 2.58's (TRAIN_ENV_ROOT; ray/_private/node.py:441 "timed
      # out during startup ... some of the raylet failed to startup", ray/_private/state.py:161 "Failed to connect
      # to GCS"), and verl's V1 console lines are still "step:N - " (verl/utils/logger/aggregate_logger.py:26-31)
      if (( try < tries )) && ! grep -q -E 'step:[0-9]+ - ' "${try_log}" &&
        grep -q -E 'timed out during startup|Failed to connect to GCS|raylet failed to start' "${try_log}"; then
        echo "[sco_interiorgs_v10_train] Ray start-up failed (try ${try} of ${tries}); trying again in this job"
        pkill -9 -f "${V10_RAY_TEMP_DIR}" || true  # what is left of this try's Ray (its command lines name the dir)
        sleep 30
        continue
      fi
      exit "${status}"
    done
    touch "${RUN}/STOP_RENDERER"
    ;;
  distance-tables)
    # round 13 (the user's 12th supplement, sections 2.2 and 3): CPU work on a one-card job's 14 vCPUs
    "${PY}" -m data_gen.interiorgs_v10.distance_table build --gs-root "${GS_ROOT}" "${@:3}"
    ;;
  *)
    echo "unknown mode ${MODE}" >&2
    exit 2
    ;;
esac
