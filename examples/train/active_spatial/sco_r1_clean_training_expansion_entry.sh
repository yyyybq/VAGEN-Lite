#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
BASE=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
SCOPE=${BASE}/r1_clean_training_expansion_scope_v1_20260918
RUN=${BASE}/r1_clean_training_expansion_v1_20260918
ARCHIVE=${R1_CLEAN_EXPANSION_CODE_ARCHIVE:?required}
EXPECTED_ARCHIVE_SHA=${R1_CLEAN_EXPANSION_CODE_ARCHIVE_SHA256:?required}
EXPECTED_COMMIT=${R1_CLEAN_EXPANSION_COMMIT:?required}
PORT=${R1_CLEAN_EXPANSION_RENDER_PORT:-8898}
WORK=$(mktemp -d /tmp/r1_clean_training_expansion_code.XXXXXXXX)
renderer_pid=

mkdir -p "${RUN}"
actual_archive_sha=$(sha256sum "${ARCHIVE}" | awk '{print $1}')
[[ "${actual_archive_sha}" == "${EXPECTED_ARCHIVE_SHA}" ]] || { echo archive_sha_mismatch >&2; exit 2; }
tar -xzf "${ARCHIVE}" -C "${WORK}"

cleanup() {
  status=$?
  if [[ -n "${renderer_pid}" ]]; then
    kill "${renderer_pid}" 2>/dev/null || true
    wait "${renderer_pid}" 2>/dev/null || true
  fi
  printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
  if [[ "${WORK}" == /tmp/r1_clean_training_expansion_code.* ]]; then rm -rf -- "${WORK}"; fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

cd "${WORK}"
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${WORK}/scripts:${WORK}
export CUDA_VISIBLE_DEVICES=0
export NO_PROXY='*'
export no_proxy='*'
GS_ROOT=/mnt/umm/users/yinbaiqiao/InteriorGS

SCENES=$(${PY} -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["scenes"]))' "${SCOPE}/scope_summary.json")
IFS=',' read -ra scene_array <<< "${SCENES}"
for scene in "${scene_array[@]}"; do
  [[ -d "${GS_ROOT}/${scene}" ]] || { echo "missing shared asset ${scene}" >&2; exit 3; }
done

{
  echo "hostname=$(hostname)"
  echo "expected_commit=${EXPECTED_COMMIT}"
  echo "archive=${ARCHIVE}"
  echo "archive_sha256=${actual_archive_sha}"
  echo "python=${PY}"
  "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo "required_cluster=H800"
  echo "required_pool=h800"
  echo "gs_root=${GS_ROOT}"
  echo "render_port=${PORT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RUN}/worker_environment.txt"

"${PY}" scripts/r1_run_medium_selector_scene_shards.py \
  --selection "${SCOPE}/projective_medium_selection.json" \
  --sources "${SCOPE}/sources_train_only.json" --gs-root "${GS_ROOT}" \
  --output-dir "${RUN}/projective_medium_v2" --selector v2 --max-workers 5 \
  --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 \
  --lower-expansions 100000 > "${RUN}/projective_medium_v2.log" 2>&1

"${PY}" scripts/r1_materialize_difficulty_conditioned.py \
  --selector "${RUN}/projective_medium_v2/selector_results.json" \
  --output-dir "${RUN}/projective_medium_v2/independent_validation" \
  > "${RUN}/projective_materialize.log" 2>&1
"${PY}" scripts/r1_replay_difficulty_candidates.py \
  --candidates "${RUN}/projective_medium_v2/independent_validation/candidate_rows.jsonl" \
  --reachability "${RUN}/projective_medium_v2/independent_validation/reachability_manifest.jsonl" \
  --gs-root "${GS_ROOT}" \
  --output "${RUN}/projective_medium_v2/independent_validation/runtime_replay.json" \
  > "${RUN}/projective_runtime.log" 2>&1

"${PY}" scripts/r1_full_regeneration.py \
  --sources "${SCOPE}/sources_train_only.json" --gs-root "${GS_ROOT}" \
  --output-dir "${RUN}/fov_min4" --scenes "${SCENES}" --splits train \
  --source-index-selection "${SCOPE}/fov_source_index_selection.json" \
  --max-steps 12 --reachability-tiers 2000,25000 --final-tier-expansions 250000 \
  --max-replacement-candidates 0 --resume > "${RUN}/fov_min4.log" 2>&1
"${PY}" scripts/r1_replay_fov_certificates.py \
  --candidates "${RUN}/fov_min4/train/trainable.jsonl" \
  --reachability "${RUN}/fov_min4/train/reachability_manifest.jsonl" \
  --gs-root "${GS_ROOT}" --output "${RUN}/fov_min4/train/runtime_replay.json" \
  > "${RUN}/fov_runtime.log" 2>&1

bash examples/train/active_spatial/start_gs_render_http_service.sh \
  --gs-root "${GS_ROOT}" --host 127.0.0.1 --port "${PORT}" --gpus 0 \
  --max-workers 1 --max-inflight 1 --admit-timeout 300 \
  --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite \
  > "${RUN}/renderer.log" 2>&1 &
renderer_pid=$!
echo "${renderer_pid}" > "${RUN}/renderer.pid"
for _ in $(seq 1 180); do
  if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" > "${RUN}/renderer_health.json"; then break; fi
  kill -0 "${renderer_pid}" 2>/dev/null || { echo renderer_exited >&2; exit 4; }
  sleep 2
done
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null

"${PY}" scripts/r1_projective_observability_audit.py \
  --repaired "${RUN}/projective_medium_v2/independent_validation/candidate_rows.jsonl" \
  --reachability "${RUN}/projective_medium_v2/independent_validation/reachability_manifest.jsonl" \
  --gs-root "${GS_ROOT}" --renderer-url "http://127.0.0.1:${PORT}/render" \
  --renderer-lock "${RUN}/renderer.lock" \
  --output-dir "${RUN}/projective_medium_v2/official_observability_v1" \
  > "${RUN}/projective_rgb.log" 2>&1
"${PY}" scripts/r1_fov_observability_audit.py \
  --repaired "${RUN}/fov_min4/train/trainable.jsonl" \
  --reachability "${RUN}/fov_min4/train/reachability_manifest.jsonl" \
  --gs-root "${GS_ROOT}" --renderer-url "http://127.0.0.1:${PORT}/render" \
  --renderer-lock "${RUN}/renderer.lock" \
  --output-dir "${RUN}/fov_min4/official_observability_v1" \
  > "${RUN}/fov_rgb.log" 2>&1

kill "${renderer_pid}" 2>/dev/null || true
wait "${renderer_pid}" 2>/dev/null || true
renderer_pid=
find "${RUN}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN}/SHA256SUMS"
printf '{"exit_status":0,"stopped_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
trap - EXIT INT TERM
if [[ "${WORK}" == /tmp/r1_clean_training_expansion_code.* ]]; then rm -rf -- "${WORK}"; fi
