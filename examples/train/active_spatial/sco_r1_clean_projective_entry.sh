#!/usr/bin/env bash
# The control-side launch wrapper demotes root BEFORE this script writes files.
set -euo pipefail
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || { echo incorrect_artifact_owner; exit 3; }
MODE=${1:?renderer or training}
RUN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-clean-Projective-v0
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
WORK=$(mktemp -d /tmp/r1_clean_projective.XXXXXXXX)
cd "${RUN}/package"
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
OUT=${RUN}/${MODE}_worker
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1
renderer_pid=
cleanup() {
  status=$?
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; wait "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == training ]]; then touch "${RUN}/STOP_RENDERER"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
}
trap cleanup EXIT
{
  hostname; id; date -u +%FT%TZ; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo cluster=zoetrope; echo pool=zoetrope; echo "code_dir=${WORK}"
  sha256sum "${RUN}/package/source.tar.gz" "${RUN}/frozen/SHA256SUMS"
} > "${OUT}/environment.txt"
(cd "${RUN}/frozen" && sha256sum -c SHA256SUMS)
"${PY}" -m pip freeze > "${OUT}/pip_freeze.txt"
if [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  LEDGER=${RUN}/assets/ledger.json
  mkdir -p "${RUN}/assets"
  [[ -f ${LEDGER} ]] || "${PY}" scripts/r1_aoss_scene_pipeline.py build-ledger --sources "${RUN}/frozen/asset_sources.json" --output "${LEDGER}"
  mapfile -t scenes < <("${PY}" -c 'import json,sys;print("\n".join(sorted({json.loads(l)["scene_id"] for f in sys.argv[1:] for l in open(f) if l.strip()})))' "${RUN}/frozen/train.jsonl" "${RUN}/frozen/eval_policy_rows.jsonl")
  for scene in "${scenes[@]}"; do
    "${PY}" scripts/r1_aoss_scene_pipeline.py stage-one --ledger "${LEDGER}" --scene-id "${scene}" --cache-root "${RUN}/assets" --log-dir "${RUN}/assets/logs"
  done
  bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "${RUN}/assets/ready" --host 0.0.0.0 --port 8914 --gpus 0 --max-workers 1 --max-inflight 1 --admit-timeout 300 --conda-env "${ENV}" > "${OUT}/renderer.log" 2>&1 &
  renderer_pid=$!; echo "${renderer_pid}" > "${OUT}/renderer.pid"
  for _ in $(seq 1 180); do
    curl --noproxy '*' -fsS http://127.0.0.1:8914/health > "${OUT}/health.json" && break
    kill -0 "${renderer_pid}"; sleep 2
  done
  curl --noproxy '*' -fsS http://127.0.0.1:8914/health > "${OUT}/health.json"
  "${PY}" scripts/r1_clean_projective_runtime_preflight.py --frozen "${RUN}/frozen" --renderer-url http://127.0.0.1:8914/render --gs-root "${RUN}/assets/ready" --output "${RUN}/runtime_preflight.json"
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8914/render\n' "${worker_ip}" > "${RUN}/renderer_endpoint.tmp"
  mv "${RUN}/renderer_endpoint.tmp" "${RUN}/renderer_endpoint.txt"
  # Bounded lease; another task's services are never stopped.
  for _ in $(seq 1 17280); do
    [[ -e ${RUN}/STOP_RENDERER ]] && exit 0
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
  curl --noproxy '*' -fsS "${R1_RENDER_URL%/render}/health" > "${OUT}/health.json"
  MODEL=$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["path"])' "${RUN}/frozen/data_gate.json")
  (cd "${MODEL}" && sha256sum -c "${MODEL}_SHA256SUMS") > "${OUT}/model_hash_check.txt"
  mkdir -p "${RUN}/smoke" "${RUN}/formal"
  "${PY}" -m vagen.r1_clean_projective_ppo --config "${RUN}/frozen/smoke.yaml" --endpoint 1 2>&1 | tee "${RUN}/smoke/train.log"
  "${PY}" scripts/r1_clean_projective_smoke_gate.py --run "${RUN}"
  # Entirely new process; never load smoke weights or optimizer state.
  "${PY}" -m vagen.r1_clean_projective_ppo --config "${RUN}/frozen/formal.yaml" --endpoint 250 2>&1 | tee "${RUN}/formal/train.log"
else
  echo unknown_mode; exit 2
fi
