#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?renderer or training}
RUN=${2:?pilot run root}
PACKAGE_ROOT=${PWD}
R1_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0
R1_FROZEN=${R1_ROOT}/frozen_v1
ASSETS=${R1_ROOT}/assets/ready
ENV_ROOT=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV_ROOT}/bin/python
DIAGNOSIS=${RUN}/diagnostic/r5_production_backward.json
CONFIG_REPORT=${RUN}/config_diff_report.json
PREFLIGHT_REPORT=${RUN}/frozen/preflight_report.json
FINAL_PREFLIGHT=${RUN}/final_static_preflight.json

[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
[[ ${MODE} == renderer || ${MODE} == training ]] || exit 2

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export NO_PROXY='*' no_proxy='*'
export WANDB_MODE=offline
export PYTHONPATH="${PACKAGE_ROOT}/verl:${PACKAGE_ROOT}/scripts:${PACKAGE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=${RUN}/torch_extensions
export INTERIORGS_HTTP_TIMEOUT=900 INTERIORGS_HTTP_RETRIES=2
export INTERIORGS_HTTP_BACKOFF=1 INTERIORGS_HTTP_MAX_BACKOFF=10

WHEEL_ROOT=${ENV_ROOT}/lib/python3.12/site-packages/nvidia/curand
export CUDA_HOME=${ENV_ROOT}
export CURAND_INCLUDE_DIR=${WHEEL_ROOT}/include
export CURAND_LIBRARY_DIR=${WHEEL_ROOT}/lib
[[ -x ${CUDA_HOME}/bin/nvcc && -f ${CURAND_INCLUDE_DIR}/curand.h ]]
compgen -G "${CURAND_LIBRARY_DIR}/libcurand.so*" >/dev/null
export PATH="${CUDA_HOME}/bin:${ENV_ROOT}/bin:${PATH}"
export CPATH="${CURAND_INCLUDE_DIR}:${ENV_ROOT}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CURAND_LIBRARY_DIR}:${ENV_ROOT}/lib:${ENV_ROOT}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER=1

ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1

renderer_pid=
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == training ]]; then touch "${RUN}/STOP_TRAIN_RENDERER"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

{
  hostname
  id
  date -u +%FT%TZ
  "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
} > "${OUT}/environment.txt"

(cd "${RUN}/frozen" && sha256sum -c SHA256SUMS)
"${PY}" - "${DIAGNOSIS}" "${CONFIG_REPORT}" "${PREFLIGHT_REPORT}" "${FINAL_PREFLIGHT}" <<'PY'
import json, sys
diagnosis, configs, preflight, final_preflight = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:])
assert diagnosis["status"] == "PASS"
assert diagnosis["safety"]["optimizer_step_called"] is False
assert diagnosis["actor"]["parameters_unchanged"] and diagnosis["critic"]["parameters_unchanged"]
assert configs["status"] == "PASS" and all(configs["checks"].values())
assert preflight["status"] == "PASS"
assert final_preflight["status"] == "PASS" and final_preflight["submission_allowed"] is True
PY

if [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${ASSETS}" --host 0.0.0.0 --port 8915 --gpus 0 \
    --max-workers 1 --max-inflight 1 --admit-timeout 900 --conda-env "${ENV_ROOT}" \
    > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!
  ready=false
  for _ in $(seq 1 180); do
    if "${PY}" -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8915/health", timeout=2)' 2>/dev/null; then
      ready=true
      break
    fi
    kill -0 "${renderer_pid}"
    sleep 2
  done
  [[ ${ready} == true ]]
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8915/render\n' "${worker_ip}" > "${RUN}/renderer_endpoint.tmp"
  mv "${RUN}/renderer_endpoint.tmp" "${RUN}/renderer_endpoint.txt"
  printf '{"status":"PASS","health":"http://127.0.0.1:8915/health","utc":"%s"}\n' "$(date -u +%FT%TZ)" > "${RUN}/renderer_health.json"
  while [[ ! -e ${RUN}/STOP_TRAIN_RENDERER ]]; do
    kill -0 "${renderer_pid}"
    sleep 10
  done
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for _ in $(seq 1 360); do
  [[ -s ${RUN}/renderer_endpoint.txt && -f ${RUN}/renderer_health.json ]] && break
  sleep 5
done
[[ -s ${RUN}/renderer_endpoint.txt && -f ${RUN}/renderer_health.json ]]
export R1_RENDER_URL
R1_RENDER_URL=$(<"${RUN}/renderer_endpoint.txt")
MODEL=$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["path"])' "${R1_FROZEN}/data_gate.json")
(cd "${MODEL}" && sha256sum -c "${MODEL}_SHA256SUMS") > "${OUT}/model_hash_check.txt"

for branch in S0 S1 S5; do
  config=${RUN}/frozen/configs/${branch}.yaml
  branch_root=${RUN}/branches/${branch}
  [[ ! -e ${branch_root}/checkpoints/global_step_1 ]]
  mkdir -p "${branch_root}"
  "${PY}" - "${branch}" "${MODEL}" "${branch_root}/step0_identity.json" <<'PY'
import hashlib, json, pathlib, sys
branch, model, output = sys.argv[1:]
manifest = pathlib.Path(model + "_SHA256SUMS")
payload = {
    "status": "PASS",
    "branch": branch,
    "actor_initialization": model,
    "actor_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "critic_initialization": "deterministic fresh value head under common training seed",
    "training_update_applied": False,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  timeout --signal=TERM --kill-after=180s 30h \
    "${PY}" -m vagen.dense_score_reward_only_pilot --config "${config}" \
    2>&1 | tee "${branch_root}/train.log"
  for step in 50 100 150; do
    checkpoint=${branch_root}/checkpoints/global_step_${step}
    [[ -f ${checkpoint}/COMPLETE ]]
    [[ -f ${checkpoint}/actor/huggingface/model.safetensors.index.json ]]
  done
  "${PY}" - "${branch}" "${branch_root}" <<'PY'
import json, pathlib, sys
branch, root_text = sys.argv[1:]
root = pathlib.Path(root_text)
payload = {
    "status": "PASS",
    "branch": branch,
    "updates": 150,
    "evaluation_steps": [0, 50, 100, 150],
    "saved_steps": [50, 100, 150],
    "step0_identity": str(root / "step0_identity.json"),
    "checkpoints": {str(step): str(root / "checkpoints" / f"global_step_{step}") for step in (50, 100, 150)},
}
(root / "branch_training_gate.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
done

"${PY}" - "${RUN}" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
branches = {name: json.load(open(root / "branches" / name / "branch_training_gate.json", encoding="utf-8")) for name in ("S0", "S1", "S5")}
assert all(item["status"] == "PASS" for item in branches.values())
(root / "pilot_training_gate.json").write_text(json.dumps({"status": "PASS", "branches": branches}, indent=2) + "\n", encoding="utf-8")
PY
