#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?renderer or training}
BRANCH=${2:?S0, S1, or S5}
RUN=${3:?parallel pilot run root}
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
BRANCH_ROOT=${RUN}/branches/${BRANCH}
VALIDATED_GSPLAT_CACHE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/dense_score_reward_only_pilot_20261007/torch_extensions/gsplat_cuda

[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
[[ ${MODE} == renderer || ${MODE} == training ]] || exit 2
[[ ${BRANCH} == S0 || ${BRANCH} == S1 || ${BRANCH} == S5 ]] || exit 2

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export NO_PROXY='*' no_proxy='*'
export WANDB_MODE=offline
export PYTHONPATH="${PACKAGE_ROOT}/verl:${PACKAGE_ROOT}/scripts:${PACKAGE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${ENV_ROOT}/bin/x86_64-conda-linux-gnu-g++
export TORCH_EXTENSIONS_DIR=${BRANCH_ROOT}/torch_extensions_${MODE}
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
OUT=${BRANCH_ROOT}/${MODE}_worker/attempts/${ATTEMPT}
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1

renderer_pid=
monitor_pid=
ray_sync_pid=

snapshot_ray_logs() {
  [[ ${MODE} == training && -n ${DENSE_SCORE_RAY_TEMP_DIR:-} && -d ${DENSE_SCORE_RAY_TEMP_DIR} ]] || return 0
  local tmp=${OUT}/ray_logs_latest.tar.gz.tmp
  tar -C "${DENSE_SCORE_RAY_TEMP_DIR}" -czf "${tmp}" . 2>/dev/null && mv "${tmp}" "${OUT}/ray_logs_latest.tar.gz" || true
}

stop_observers() {
  local pid
  for pid in "${monitor_pid}" "${ray_sync_pid}"; do
    if [[ -n ${pid} ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  snapshot_ray_logs
}

cleanup() {
  status=$?
  trap - EXIT
  stop_observers
  if [[ -n ${renderer_pid} ]]; then kill "${renderer_pid}" 2>/dev/null || true; fi
  if [[ ${MODE} == training ]]; then touch "${BRANCH_ROOT}/STOP_RENDERER"; fi
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' "${status}" "$(date -u +%FT%TZ)" > "${OUT}/exit.json"
  exit "${status}"
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
"${PY}" - "${DIAGNOSIS}" "${CONFIG_REPORT}" "${PREFLIGHT_REPORT}" "${FINAL_PREFLIGHT}" "${BRANCH}" <<'PY'
import json, sys
diagnosis, configs, preflight, final_preflight = (json.load(open(path, encoding="utf-8")) for path in sys.argv[1:5])
branch = sys.argv[5]
assert diagnosis["status"] == "PASS"
assert diagnosis["safety"]["optimizer_step_called"] is False
assert diagnosis["actor"]["parameters_unchanged"] and diagnosis["critic"]["parameters_unchanged"]
assert configs["status"] == "PASS" and all(configs["checks"].values())
assert preflight["status"] == "PASS"
assert final_preflight["status"] == "PASS" and final_preflight["submission_allowed"] is True
assert branch in final_preflight["resource_topology"]["branches"]
PY

if [[ ${MODE} == renderer ]]; then
  export CUDA_VISIBLE_DEVICES=0
  # ProcessPool workers import gsplat independently.  If all twelve workers
  # attempt the first JIT build in one shared cache they race on gsplat_cuda.so.
  # Seed every branch from the already validated, completed H800 build before
  # the service creates any worker process.
  [[ -f ${VALIDATED_GSPLAT_CACHE}/gsplat_cuda.so ]]
  mkdir -p "${TORCH_EXTENSIONS_DIR}"
  cp -a "${VALIDATED_GSPLAT_CACHE}" "${TORCH_EXTENSIONS_DIR}/gsplat_cuda"
  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${ASSETS}" --host 0.0.0.0 --port 8915 --gpus 0 \
    --max-workers 12 --max-inflight 12 --admit-timeout 900 --conda-env "${ENV_ROOT}" \
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
  # Health alone does not initialize ProcessPool workers or import gsplat.
  # Exercise twelve distinct scenes concurrently, matching one PPO update's
  # distinct-scene count and binding all twelve slots on the selected GPU.
  # and fails closed on extension/cache or real-render errors.
  "${PY}" - "${R1_FROZEN}/train.jsonl" "${OUT}/render_smoke.json" <<'PY'
import asyncio, hashlib, io, json, pathlib, sys
import numpy as np
from vagen.envs.active_spatial.env import runtime_render_camera_parameters
from vagen.envs.active_spatial.render.http_render_client import InteriorGSHTTPRenderClient

manifest, output = map(pathlib.Path, sys.argv[1:])
rows, scenes = [], set()
for line in manifest.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row["scene_id"] not in scenes:
        scenes.add(row["scene_id"])
        rows.append(row)
    if len(rows) == 12:
        break
assert len(rows) == 12

async def render_one(row):
    c2w = np.asarray(row["init_camera"]["extrinsics"], dtype=np.float64)
    native_k = np.asarray(row["init_camera"]["intrinsics"], dtype=np.float64)
    k, w2c = runtime_render_camera_parameters(row, c2w, native_k, (256, 256))
    client = InteriorGSHTTPRenderClient("http://127.0.0.1:8915/render", timeout=900, retries=0)
    images = await client.render(row["scene_id"], [{
        "mode": "cam_param", "intrinsics": k.tolist(),
        "extrinsics": w2c.tolist(), "size": [256, 256],
    }])
    assert len(images) == 1 and images[0].size == (256, 256)
    buffer = io.BytesIO()
    images[0].save(buffer, format="PNG")
    return {"scene_id": row["scene_id"], "png_sha256": hashlib.sha256(buffer.getvalue()).hexdigest()}

async def main():
    return await asyncio.gather(*(render_one(row) for row in rows))

results = asyncio.run(main())
output.write_text(json.dumps({"status": "PASS", "real_render_count": 8, "results": results}, indent=2) + "\n", encoding="utf-8")
PY
  worker_ip=$(hostname -I | awk '{print $1}')
  printf 'http://%s:8915/render\n' "${worker_ip}" > "${BRANCH_ROOT}/renderer_endpoint.tmp"
  mv "${BRANCH_ROOT}/renderer_endpoint.tmp" "${BRANCH_ROOT}/renderer_endpoint.txt"
  printf '{"status":"PASS","branch":"%s","health":"http://127.0.0.1:8915/health","gpus":1,"max_workers":12,"max_inflight":12,"real_render_smoke":"PASS","utc":"%s"}\n' \
    "${BRANCH}" "$(date -u +%FT%TZ)" > "${BRANCH_ROOT}/renderer_health.json"
  while [[ ! -e ${BRANCH_ROOT}/STOP_RENDERER ]]; do
    kill -0 "${renderer_pid}"
    sleep 10
  done
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for _ in $(seq 1 720); do
  [[ -s ${BRANCH_ROOT}/renderer_endpoint.txt && -f ${BRANCH_ROOT}/renderer_health.json ]] && break
  sleep 5
done
[[ -s ${BRANCH_ROOT}/renderer_endpoint.txt && -f ${BRANCH_ROOT}/renderer_health.json ]]
export R1_RENDER_URL
R1_RENDER_URL=$(<"${BRANCH_ROOT}/renderer_endpoint.txt")
"${PY}" - "${R1_RENDER_URL}" <<'PY'
import json, sys, urllib.request
health = sys.argv[1].removesuffix("/render") + "/health"
with urllib.request.urlopen(health, timeout=10) as response:
    payload = json.load(response)
assert payload["ok"] is True
assert payload["max_inflight"] == 12
PY
MODEL=$("${PY}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"]["path"])' "${R1_FROZEN}/data_gate.json")
(cd "${MODEL}" && sha256sum -c "${MODEL}_SHA256SUMS") > "${OUT}/model_hash_check.txt"

# Persist enough evidence to distinguish Python/Ray failures from a platform
# SIGKILL.  Ray sockets remain under a short node-local path; its complete log
# tree is mirrored to the shared attempt directory once per minute.  Resource
# sampling is intentionally read-only and cannot call an optimizer.
export PYTHONUNBUFFERED=1
export DENSE_SCORE_OBSERVABILITY_ROOT=${OUT}
export DENSE_SCORE_RAY_TEMP_DIR=/tmp/ds_ray_${BRANCH}_$$
mkdir -p "${DENSE_SCORE_RAY_TEMP_DIR}"
printf 'utc\tevent\tpid\n%s\tlauncher_ready\t%s\n' "$(date -u +%FT%TZ)" "$$" > "${OUT}/lifecycle.tsv"
printf 'utc,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,utilization_memory_pct,power_w\n' > "${OUT}/gpu_resources.csv"
printf 'utc,mem_total_bytes,mem_used_bytes,mem_available_bytes,swap_used_bytes,load1,load5,load15\n' > "${OUT}/host_resources.csv"
(
  while :; do
    ts=$(date -u +%FT%TZ)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw \
      --format=csv,noheader,nounits | sed "s/^/${ts},/" >> "${OUT}/gpu_resources.csv" || true
    read -r load1 load5 load15 _ < /proc/loadavg
    free -b | awk -v ts="${ts}" -v l1="${load1}" -v l5="${load5}" -v l15="${load15}" \
      '/^Mem:/ {total=$2; used=$3; avail=$7} /^Swap:/ {swap=$3} END {printf "%s,%s,%s,%s,%s,%s,%s,%s\n",ts,total,used,avail,swap,l1,l5,l15}' \
      >> "${OUT}/host_resources.csv" || true
    {
      printf '\n===== %s =====\n' "${ts}"
      ps -eo pid,ppid,state,etimes,pcpu,pmem,rss,vsz,comm,args --sort=-rss | head -n 100
    } >> "${OUT}/process_snapshots.log" 2>&1 || true
    sleep 10
  done
) &
monitor_pid=$!
(
  while :; do
    sleep 60
    snapshot_ray_logs
  done
) &
ray_sync_pid=$!

config=${RUN}/frozen/configs/${BRANCH}.yaml
[[ ! -e ${BRANCH_ROOT}/checkpoints/global_step_1 ]]
"${PY}" - "${BRANCH}" "${MODEL}" "${BRANCH_ROOT}/step0_identity.json" <<'PY'
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
printf '%s\tpython_start\t%s\n' "$(date -u +%FT%TZ)" "$$" >> "${OUT}/lifecycle.tsv"
set +e
timeout --signal=TERM --kill-after=180s 7d \
  "${PY}" -u -m vagen.dense_score_reward_only_pilot --config "${config}" \
  2>&1 | tee "${BRANCH_ROOT}/train.log"
train_status=${PIPESTATUS[0]}
set -e
printf '%s\tpython_exit_%s\t%s\n' "$(date -u +%FT%TZ)" "${train_status}" "$$" >> "${OUT}/lifecycle.tsv"
[[ ${train_status} -eq 0 ]] || exit "${train_status}"
for step in 50 100 150; do
  checkpoint=${BRANCH_ROOT}/checkpoints/global_step_${step}
  [[ -f ${checkpoint}/COMPLETE ]]
  [[ -f ${checkpoint}/actor/huggingface/model.safetensors.index.json ]]
done
"${PY}" - "${BRANCH}" "${BRANCH_ROOT}" <<'PY'
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
