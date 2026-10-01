#!/usr/bin/env bash
# The control-side launch wrapper demotes root BEFORE this script writes files.
set -euo pipefail
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || { echo incorrect_artifact_owner; exit 3; }
MODE=${1:?renderer or training}
RUN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0
FROZEN=${R1_FROZEN_DIR:?required}
PACKAGE=${R1_PACKAGE_DIR:?required}
STOP_RENDERER_FILE=${R1_STOP_RENDERER_FILE:-${RUN}/STOP_RENDERER}
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
WORK=$(mktemp -d /tmp/r1_clean_projective.XXXXXXXX)
cd "${PACKAGE}"
sha256sum -c SHA256SUMS
tar -xzf source.tar.gz -C "${WORK}"
cd "${WORK}"
export PATH=${ENV}/bin:${PATH} PYTHONPATH=${WORK}/verl:${WORK}/scripts:${WORK}
export PYTHONDONTWRITEBYTECODE=1 NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export CUDA_HOME=${ENV} TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++
export CPATH=${ENV}/targets/x86_64-linux/include:${CPATH:-}
export LIBRARY_PATH=${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=${RUN}/torch_extensions
# A scene load can exceed five minutes on the shared asset mount.  PPO has 48
# concurrent rollouts, so admission and client timeouts must cover both a slow
# load and a bounded queue.  This changes only transport/lifecycle behavior;
# the renderer implementation, camera, observations, and task semantics remain
# frozen.
export INTERIORGS_HTTP_TIMEOUT=1800
export INTERIORGS_HTTP_RETRIES=7
export INTERIORGS_HTTP_BACKOFF=1
export INTERIORGS_HTTP_MAX_BACKOFF=30
ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
OUT=${RUN}/${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1
renderer_pid=
health_check() {
  "${PY}" - "$1" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
    assert response.status == 200
    body = response.read()
    if body:
        json.loads(body)
PY
}
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; wait "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == training ]]; then touch "${STOP_RENDERER_FILE}"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT
{
  hostname; id; date -u +%FT%TZ; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo cluster=h800; echo pool=h800; echo "code_dir=${WORK}"
  sha256sum "${PACKAGE}/source.tar.gz" "${FROZEN}/SHA256SUMS"
} > "${OUT}/environment.txt"
(cd "${FROZEN}" && sha256sum -c SHA256SUMS)
"${PY}" -m pip freeze > "${OUT}/pip_freeze.txt"
if [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 8 ]]
  LEDGER=${RUN}/assets/ledger.json
  mkdir -p "${RUN}/assets"
  [[ -f ${LEDGER} ]] || "${PY}" scripts/r1_aoss_scene_pipeline.py build-ledger --sources "${FROZEN}/asset_sources.json" --output "${LEDGER}"
  mapfile -t scenes < <("${PY}" -c 'import json,sys;print("\n".join(sorted({json.loads(l)["scene_id"] for f in sys.argv[1:] for l in open(f) if l.strip()})))' "${FROZEN}/train.jsonl" "${FROZEN}/eval_policy_rows.jsonl")
  for scene in "${scenes[@]}"; do
    "${PY}" scripts/r1_aoss_scene_pipeline.py stage-one --ledger "${LEDGER}" --scene-id "${scene}" --cache-root "${RUN}/assets" --log-dir "${RUN}/assets/logs"
  done
  # One isolated renderer process per H800.  Each process still serializes its
  # own scene switch + render operations; the eight independent GPUs provide
  # enough admission capacity for the frozen 12x4 PPO rollout batch.
  bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "${RUN}/assets/ready" --host 0.0.0.0 --port 8914 --gpus 0,1,2,3,4,5,6,7 --max-workers 8 --max-inflight 8 --admit-timeout 1800 --conda-env "${ENV}" > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!; echo "${renderer_pid}" > "${OUT}/renderer.pid"
  for _ in $(seq 1 180); do
    health_check http://127.0.0.1:8914/health && break
    kill -0 "${renderer_pid}"; sleep 2
  done
  health_check http://127.0.0.1:8914/health
  "${PY}" -c 'import json,sys,urllib.request; json.dump(json.load(urllib.request.urlopen(sys.argv[1],timeout=10)),open(sys.argv[2],"w"),indent=2)' http://127.0.0.1:8914/health "${OUT}/health.json"
  "${PY}" scripts/r1_clean_projective_runtime_preflight.py --frozen "${FROZEN}" --renderer-url http://127.0.0.1:8914/render --gs-root "${RUN}/assets/ready" --output "${RUN}/runtime_preflight.json"
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8914/render\n' "${worker_ip}" > "${RUN}/renderer_endpoint.tmp"
  mv "${RUN}/renderer_endpoint.tmp" "${RUN}/renderer_endpoint.txt"
  # Bounded lease; another task's services are never stopped.
  for _ in $(seq 1 17280); do
    [[ -e ${STOP_RENDERER_FILE} ]] && exit 0
    kill -0 "${renderer_pid}"; sleep 10
  done
  echo renderer_48h_lease_expired; exit 6
elif [[ ${MODE} == training ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  [[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 8 ]]
  # Training is submitted only after control-side runtime preflight PASS.
  "${PY}" -c 'import json,sys; r=json.load(open(sys.argv[1]));assert r["status"]=="PASS" and r["completed"]==210' "${RUN}/runtime_preflight.json"
  export R1_RENDER_URL
  R1_RENDER_URL=$(<"${RUN}/renderer_endpoint.txt")
  health_check "${R1_RENDER_URL%/render}/health"
  "${PY}" -c 'import json,sys,urllib.request; json.dump(json.load(urllib.request.urlopen(sys.argv[1],timeout=10)),open(sys.argv[2],"w"),indent=2)' "${R1_RENDER_URL%/render}/health" "${OUT}/health.json"
  MODEL=$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["path"])' "${FROZEN}/data_gate.json")
  (cd "${MODEL}" && sha256sum -c "${MODEL}_SHA256SUMS") > "${OUT}/model_hash_check.txt"
  mkdir -p "${RUN}/smoke" "${RUN}/formal"
  "${PY}" -m vagen.r1_clean_projective_ppo --config "${FROZEN}/smoke.yaml" --endpoint 1 2>&1 | tee "${RUN}/smoke/train.log"
  "${PY}" scripts/r1_clean_projective_smoke_gate.py --run "${RUN}" --frozen "${FROZEN}"
  # Entirely new process; never load smoke weights or optimizer state.
  "${PY}" -m vagen.r1_clean_projective_ppo --config "${FROZEN}/formal.yaml" --endpoint 250 2>&1 | tee "${RUN}/formal/train.log"
else
  echo unknown_mode; exit 2
fi
