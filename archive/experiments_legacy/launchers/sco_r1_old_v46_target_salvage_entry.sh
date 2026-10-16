#!/usr/bin/env bash
# Resumable, scene-staged old-v46 target salvage.  This is data validation only:
# no model checkpoint, policy rollout, or training command is invoked here.
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
BASE=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
SCOPE=${BASE}/r1_old_v46_target_salvage_scope_v1_20260919
RUN=${BASE}/r1_old_v46_target_salvage_v1_20260919
ARCHIVE=${R1_OLD_V46_SALVAGE_CODE_ARCHIVE:?required}
EXPECTED_ARCHIVE_SHA=${R1_OLD_V46_SALVAGE_CODE_ARCHIVE_SHA256:?required}
EXPECTED_COMMIT=${R1_OLD_V46_SALVAGE_COMMIT:?required}
PORT=${R1_OLD_V46_SALVAGE_RENDER_PORT:-8902}
WORK=$(mktemp -d /tmp/r1_old_v46_salvage_code.XXXXXXXX)
CACHE=${RUN}/scene_cache
LEDGER=${RUN}/asset_ledger.json
renderer_pid=

cleanup() {
  status=$?
  if [[ -n "${renderer_pid}" ]]; then
    kill "${renderer_pid}" 2>/dev/null || true
    wait "${renderer_pid}" 2>/dev/null || true
  fi
  printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
  if [[ "${WORK}" == /tmp/r1_old_v46_salvage_code.* ]]; then rm -rf -- "${WORK}"; fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

mkdir -p "${RUN}"
actual_archive_sha=$(sha256sum "${ARCHIVE}" | awk '{print $1}')
[[ "${actual_archive_sha}" == "${EXPECTED_ARCHIVE_SHA}" ]] || { echo "archive_sha_mismatch" >&2; exit 2; }
tar -xzf "${ARCHIVE}" -C "${WORK}"
cd "${WORK}"
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${WORK}/scripts:${WORK}
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
export NO_PROXY='*'
export no_proxy='*'

{
  echo "hostname=$(hostname)"
  echo "expected_commit=${EXPECTED_COMMIT}"
  echo "code_archive=${ARCHIVE}"
  echo "code_archive_sha256=${actual_archive_sha}"
  echo "python=${PY}"
  "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo "required_cluster=H800"
  echo "required_pool=h800"
  echo "renderer_port=${PORT}"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RUN}/worker_environment.txt"

if [[ ! -f "${LEDGER}" ]]; then
  "${PY}" scripts/r1_aoss_scene_pipeline.py build-ledger \
    --sources "${SCOPE}/asset_ledger_sources.json" --output "${LEDGER}" \
    --tsv-output "${RUN}/asset_ledger.tsv"
fi

# The service is bound to CACHE/ready.  Only one validated scene is present at
# a time; scene stage/render/cleanup are therefore mutually exclusive.
start_renderer() {
  if [[ -n "${renderer_pid}" ]] && kill -0 "${renderer_pid}" 2>/dev/null; then return; fi
  bash examples/train/active_spatial/start_gs_render_http_service.sh \
    --gs-root "${CACHE}/ready" --host 127.0.0.1 --port "${PORT}" --gpus 0 \
    --max-workers 1 --max-inflight 1 --admit-timeout 300 \
    --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite \
    > "${RUN}/renderer.log" 2>&1 &
  renderer_pid=$!
  echo "${renderer_pid}" > "${RUN}/renderer.pid"
  for _ in $(seq 1 180); do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" > "${RUN}/renderer_health.json"; then return; fi
    kill -0 "${renderer_pid}" 2>/dev/null || { echo "renderer_exited" >&2; exit 4; }
    sleep 2
  done
  echo "renderer_health_timeout" >&2
  exit 5
}

mapfile -t scene_rows < <("${PY}" - "${SCOPE}/eligible_target_source_inventory.jsonl" <<'PY'
import json, sys
from collections import defaultdict
by_scene = defaultdict(list)
for line in open(sys.argv[1]):
    if line.strip():
        row = json.loads(line)
        by_scene[row['scene_id']].append(row)
for scene in sorted(by_scene):
    projective = [str(r['source_row_index']) for r in by_scene[scene] if r['task_type'] == 'projective_relations']
    fov_count = sum(r['task_type'] == 'fov_inclusion' for r in by_scene[scene])
    smoke_index = min(int(r['source_row_index']) for r in by_scene[scene])
    # A scene can be FOV-only.  Do not use a whitespace IFS delimiter here:
    # Bash coalesces adjacent whitespace delimiters and would shift fov_count
    # into projective_indices, leaving smoke_index empty for such a scene.
    print(f"{scene}|{','.join(projective)}|{fov_count}|{smoke_index}")
PY
)

for scene_row in "${scene_rows[@]}"; do
  IFS='|' read -r scene projective_indices fov_count smoke_index <<< "${scene_row}"
  scene_evidence="${RUN}/scene_evidence/${scene}.json"
  # Evidence is written before cleanup.  A resumed job therefore does not
  # redownload a safely completed scene merely to rediscover the same shard.
  if [[ -f "${scene_evidence}" ]]; then
    echo "[resume] scene_complete=${scene}"
    continue
  fi
  "${PY}" scripts/r1_aoss_scene_pipeline.py stage-one --ledger "${LEDGER}" --scene-id "${scene}" \
    --cache-root "${CACHE}" --log-dir "${RUN}/asset_download_logs"
  start_renderer
  "${PY}" scripts/r1_renderer_scene_smoke.py \
    --source "${ROOT}/exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl" \
    --source-row-index "${smoke_index}" --scene-id "${scene}" \
    --renderer-url "http://127.0.0.1:${PORT}/render" --output "${RUN}/renderer_smoke/${scene}.json"
  "${PY}" scripts/r1_aoss_scene_pipeline.py mark --ledger "${LEDGER}" --scene-id "${scene}" \
    --kind renderer-smoke --status success --evidence "${RUN}/renderer_smoke/${scene}.json"

  projective_root="${RUN}/projective_medium_v2/shards/${scene}"
  if [[ -n "${projective_indices}" && ! -f "${projective_root}/selector_results.json" ]]; then
    "${PY}" scripts/r1_projective_medium_selector_v2.py \
      --selection "${SCOPE}/projective_medium_selection.json" --sources "${SCOPE}/sources_train_only.json" \
      --gs-root "${CACHE}/ready" --output-dir "${projective_root}" --source-keys "$(sed 's/,/,train:/g; s/^/train:/' <<< "${projective_indices}")" \
      --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 --lower-expansions 100000
  fi
  if [[ -n "${projective_indices}" && ! -f "${projective_root}/independent_validation/summary.json" ]]; then
    "${PY}" scripts/r1_materialize_difficulty_conditioned.py \
      --selector "${projective_root}/selector_results.json" --output-dir "${projective_root}/independent_validation"
    "${PY}" scripts/r1_replay_difficulty_candidates.py \
      --candidates "${projective_root}/independent_validation/candidate_rows.jsonl" \
      --reachability "${projective_root}/independent_validation/reachability_manifest.jsonl" \
      --gs-root "${CACHE}/ready" --output "${projective_root}/independent_validation/runtime_replay.json"
  fi
  if [[ -n "${projective_indices}" && ! -f "${projective_root}/official_observability_v1/summary.json" ]]; then
    "${PY}" scripts/r1_projective_observability_audit.py \
      --repaired "${projective_root}/independent_validation/candidate_rows.jsonl" \
      --reachability "${projective_root}/independent_validation/reachability_manifest.jsonl" \
      --gs-root "${CACHE}/ready" --renderer-url "http://127.0.0.1:${PORT}/render" \
      --renderer-lock "${RUN}/renderer.lock" --output-dir "${projective_root}/official_observability_v1"
  fi

  fov_root="${RUN}/fov_min4/shards/${scene}"
  if [[ "${fov_count}" -gt 0 && ! -f "${fov_root}/train/summary.json" ]]; then
    "${PY}" scripts/r1_full_regeneration.py \
      --sources "${SCOPE}/sources_train_only.json" --gs-root "${CACHE}/ready" --output-dir "${fov_root}" \
      --scenes "${scene}" --splits train --source-index-selection "${SCOPE}/fov_source_index_selection.json" \
      --max-steps 12 --reachability-tiers 2000,25000 --final-tier-expansions 250000 \
      --max-replacement-candidates 0 --resume
  fi
  if [[ "${fov_count}" -gt 0 && ! -f "${fov_root}/train/runtime_replay.json" ]]; then
    "${PY}" scripts/r1_replay_fov_certificates.py \
      --candidates "${fov_root}/train/trainable.jsonl" --reachability "${fov_root}/train/reachability_manifest.jsonl" \
      --gs-root "${CACHE}/ready" --output "${fov_root}/train/runtime_replay.json"
  fi
  if [[ "${fov_count}" -gt 0 && ! -f "${fov_root}/official_observability_v2/summary.json" ]]; then
    "${PY}" scripts/r1_fov_observability_v2_audit.py \
      --repaired "${fov_root}/train/trainable.jsonl" --reachability "${fov_root}/train/reachability_manifest.jsonl" \
      --gs-root "${CACHE}/ready" --renderer-url "http://127.0.0.1:${PORT}/render" \
      --renderer-lock "${RUN}/renderer.lock" --output-dir "${fov_root}/official_observability_v2"
  fi

  "${PY}" - "${scene_evidence}" "${scene}" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps({"scene_id": sys.argv[2], "completed_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
PY
  "${PY}" scripts/r1_aoss_scene_pipeline.py mark --ledger "${LEDGER}" --scene-id "${scene}" \
    --kind processing --status success --evidence "${scene_evidence}"
  "${PY}" scripts/r1_aoss_scene_pipeline.py cleanup-one --ledger "${LEDGER}" --scene-id "${scene}" \
    --cache-root "${CACHE}" --evidence "${scene_evidence}"
done

"${PY}" scripts/r1_merge_old_v46_target_salvage_shards.py \
  --scope "${SCOPE}" --projective-shards "${RUN}/projective_medium_v2/shards" \
  --fov-shards "${RUN}/fov_min4/shards" \
  --fov-calibration "${BASE}/r1_fov_observability_v2_calibration_20260919_r2/calibration_report.json" \
  --output-dir "${RUN}/salvage_matrix"
find "${RUN}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN}/SHA256SUMS"
printf '{"exit_status":0,"stopped_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
if [[ -n "${renderer_pid}" ]]; then
  kill "${renderer_pid}" 2>/dev/null || true
  wait "${renderer_pid}" 2>/dev/null || true
  renderer_pid=
fi
trap - EXIT INT TERM
if [[ "${WORK}" == /tmp/r1_old_v46_salvage_code.* ]]; then rm -rf -- "${WORK}"; fi
