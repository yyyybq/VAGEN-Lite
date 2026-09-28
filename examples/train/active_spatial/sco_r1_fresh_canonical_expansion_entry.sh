#!/usr/bin/env bash
# H800-only, scene-staged fresh canonical expansion.  It never trains and
# keeps every FOV result provisional until an independently calibrated gate.
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
BASE=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
SCOPE=${BASE}/r1_fullscale_canonical_expansion_scope_v1_20260928
RUN=${BASE}/r1_fullscale_canonical_expansion_phase_a_v1_20260928
ARCHIVE=${R1_FRESH_EXPANSION_CODE_ARCHIVE:?required}
EXPECTED_ARCHIVE_SHA=${R1_FRESH_EXPANSION_CODE_ARCHIVE_SHA256:?required}
EXPECTED_COMMIT=${R1_FRESH_EXPANSION_COMMIT:?required}
PORT=${R1_FRESH_EXPANSION_RENDER_PORT:-8904}
WORK=$(mktemp -d /tmp/r1_fresh_expansion_code.XXXXXXXX)
CACHE=${RUN}/scene_cache
LEDGER=${RUN}/asset_ledger.json
renderer_pid=

cleanup() {
  status=$?
  if [[ -n "${renderer_pid}" ]]; then kill "${renderer_pid}" 2>/dev/null || true; wait "${renderer_pid}" 2>/dev/null || true; fi
  printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
  [[ "${WORK}" == /tmp/r1_fresh_expansion_code.* ]] && rm -rf -- "${WORK}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

mkdir -p "${RUN}"
actual_archive_sha=$(sha256sum "${ARCHIVE}" | awk '{print $1}')
[[ "${actual_archive_sha}" == "${EXPECTED_ARCHIVE_SHA}" ]] || { echo archive_sha_mismatch >&2; exit 2; }
tar -xzf "${ARCHIVE}" -C "${WORK}"
cd "${WORK}"
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${WORK}/scripts:${WORK}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 NO_PROXY='*' no_proxy='*'
{
  echo "hostname=$(hostname)"; echo "expected_commit=${EXPECTED_COMMIT}"; echo "code_archive=${ARCHIVE}"; echo "code_archive_sha256=${actual_archive_sha}"; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo required_cluster=H800; echo required_pool=h800; echo "renderer_port=${PORT}"; echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RUN}/worker_environment.txt"

[[ -f "${LEDGER}" ]] || "${PY}" scripts/r1_aoss_scene_pipeline.py build-ledger --sources "${SCOPE}/sources.json" --output "${LEDGER}" --tsv-output "${RUN}/asset_ledger.tsv"
start_renderer() {
  if [[ -n "${renderer_pid}" ]] && kill -0 "${renderer_pid}" 2>/dev/null; then return; fi
  bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root "${CACHE}/ready" --host 127.0.0.1 --port "${PORT}" --gpus 0 --max-workers 1 --max-inflight 1 --admit-timeout 300 --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite > "${RUN}/renderer.log" 2>&1 &
  renderer_pid=$!; echo "${renderer_pid}" > "${RUN}/renderer.pid"
  for _ in $(seq 1 180); do
    curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" > "${RUN}/renderer_health.json" && return
    kill -0 "${renderer_pid}" 2>/dev/null || { echo renderer_exited >&2; exit 4; }; sleep 2
  done
  echo renderer_health_timeout >&2; exit 5
}

mapfile -t scene_rows < <("${PY}" - "${SCOPE}/fresh_sources.jsonl" <<'PY'
import json, sys
from collections import defaultdict
by = defaultdict(list)
for i, line in enumerate(open(sys.argv[1])):
    if line.strip(): by[json.loads(line)['scene_id']].append((i, json.loads(line)))
for scene in sorted(by):
    rows = by[scene]; p = [str(i) for i,r in rows if r['task_type'] == 'projective_relations']; f = [str(i) for i,r in rows if r['task_type'] == 'fov_inclusion']
    print(f"{scene}|{','.join(p)}|{','.join(f)}|{rows[0][0]}")
PY
)
for scene_row in "${scene_rows[@]}"; do
  IFS='|' read -r scene projective_indices fov_indices smoke_index <<< "${scene_row}"
  scene_evidence="${RUN}/scene_evidence/${scene}.json"
  [[ -f "${scene_evidence}" ]] && { echo "[resume] scene_complete=${scene}"; continue; }
  "${PY}" scripts/r1_aoss_scene_pipeline.py stage-one --ledger "${LEDGER}" --scene-id "${scene}" --cache-root "${CACHE}" --log-dir "${RUN}/asset_download_logs"
  start_renderer
  "${PY}" scripts/r1_renderer_scene_smoke.py --source "${SCOPE}/fresh_sources.jsonl" --source-row-index "${smoke_index}" --scene-id "${scene}" --renderer-url "http://127.0.0.1:${PORT}/render" --output "${RUN}/renderer_smoke/${scene}.json"
  "${PY}" scripts/r1_aoss_scene_pipeline.py mark --ledger "${LEDGER}" --scene-id "${scene}" --kind renderer-smoke --status success --evidence "${RUN}/renderer_smoke/${scene}.json"
  projective_root="${RUN}/projective_medium_v2/shards/${scene}"
  if [[ -n "${projective_indices}" && ! -f "${projective_root}/selector_results.json" ]]; then
    keys=$(sed 's/,/,train:/g; s/^/train:/' <<< "${projective_indices}")
    "${PY}" scripts/r1_projective_medium_selector_v2.py --selection "${SCOPE}/projective_medium_selection.json" --sources "${SCOPE}/sources.json" --gs-root "${CACHE}/ready" --output-dir "${projective_root}" --source-keys "${keys}" --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 --lower-expansions 100000
  fi
  if [[ -n "${projective_indices}" && ! -f "${projective_root}/independent_validation/summary.json" ]]; then
    "${PY}" scripts/r1_materialize_difficulty_conditioned.py --selector "${projective_root}/selector_results.json" --output-dir "${projective_root}/independent_validation"
    "${PY}" scripts/r1_replay_difficulty_candidates.py --candidates "${projective_root}/independent_validation/candidate_rows.jsonl" --reachability "${projective_root}/independent_validation/reachability_manifest.jsonl" --gs-root "${CACHE}/ready" --output "${projective_root}/independent_validation/runtime_replay.json"
  fi
  if [[ -n "${projective_indices}" && ! -f "${projective_root}/official_observability_v1/summary.json" ]]; then
    "${PY}" scripts/r1_projective_observability_audit.py --repaired "${projective_root}/independent_validation/candidate_rows.jsonl" --reachability "${projective_root}/independent_validation/reachability_manifest.jsonl" --gs-root "${CACHE}/ready" --renderer-url "http://127.0.0.1:${PORT}/render" --renderer-lock "${RUN}/renderer.lock" --output-dir "${projective_root}/official_observability_v1"
  fi
  fov_root="${RUN}/fov_min4/shards/${scene}"
  if [[ -n "${fov_indices}" && ! -f "${fov_root}/train/summary.json" ]]; then
    "${PY}" scripts/r1_full_regeneration.py --sources "${SCOPE}/sources.json" --gs-root "${CACHE}/ready" --output-dir "${fov_root}" --scenes "${scene}" --splits train --source-index-selection "${SCOPE}/fov_source_index_selection.json" --max-steps 12 --reachability-tiers 2000,25000 --final-tier-expansions 250000 --max-replacement-candidates 0 --resume
  fi
  if [[ -n "${fov_indices}" && ! -f "${fov_root}/train/runtime_replay.json" ]]; then
    "${PY}" scripts/r1_replay_fov_certificates.py --candidates "${fov_root}/train/trainable.jsonl" --reachability "${fov_root}/train/reachability_manifest.jsonl" --gs-root "${CACHE}/ready" --output "${fov_root}/train/runtime_replay.json"
  fi
  if [[ -n "${fov_indices}" && ! -f "${fov_root}/official_observability_v2/summary.json" ]]; then
    "${PY}" scripts/r1_fov_observability_v2_audit.py --repaired "${fov_root}/train/trainable.jsonl" --reachability "${fov_root}/train/reachability_manifest.jsonl" --gs-root "${CACHE}/ready" --renderer-url "http://127.0.0.1:${PORT}/render" --renderer-lock "${RUN}/renderer.lock" --output-dir "${fov_root}/official_observability_v2"
  fi
  "${PY}" - "${scene_evidence}" "${scene}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps({'scene_id':sys.argv[2], 'completed_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
PY
  "${PY}" scripts/r1_aoss_scene_pipeline.py mark --ledger "${LEDGER}" --scene-id "${scene}" --kind processing --status success --evidence "${scene_evidence}"
  "${PY}" scripts/r1_aoss_scene_pipeline.py cleanup-one --ledger "${LEDGER}" --scene-id "${scene}" --cache-root "${CACHE}" --evidence "${scene_evidence}"
done
"${PY}" scripts/r1_merge_fresh_canonical_expansion_shards.py --scope "${SCOPE}" --projective-shards "${RUN}/projective_medium_v2/shards" --fov-shards "${RUN}/fov_min4/shards" --output-dir "${RUN}/expansion_matrix"
find "${RUN}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN}/SHA256SUMS"
printf '{"exit_status":0,"stopped_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
trap - EXIT INT TERM
[[ "${WORK}" == /tmp/r1_fresh_expansion_code.* ]] && rm -rf -- "${WORK}"
