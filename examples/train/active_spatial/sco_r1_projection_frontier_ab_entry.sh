#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${R1_CODE_ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
DATA_ROOT="${R1_DATA_ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
GS_ROOT="${GS_ROOT:-/mnt/umm/users/yinbaiqiao/InteriorGS}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
BASE="${DATA_ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905"
PRIOR="${BASE}/projective_medium_v3_independent_reverse_diag_20260914_zoetrope_r3"
OUTPUT="${R1_OUTPUT:-${BASE}/projective_projection_frontier_ab20_20260914_zoetrope}"
SELECTION="${PRIOR}/reverse_cap16_controls4_frozen_selection.json"
SOURCES="${BASE}/path_first_canary10_v1/frozen_selection.json"
RENDER_PORT="${R1_RENDER_PORT:-8771}"
RENDER_STATE="${OUTPUT}/renderer"
ENDPOINT_FILE="${RENDER_STATE}/endpoint.txt"
LOCK_FILE="${OUTPUT}/official_rgb/renderer.lock"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${CODE_ROOT}/scripts:${CODE_ROOT}:${PYTHONPATH:-}"
export NO_PROXY='*'
export no_proxy='*'
cd "${CODE_ROOT}"
mkdir -p "${OUTPUT}"
exec > >(tee -a "${OUTPUT}/bootstrap.log") 2>&1

mark() {
  "${PYTHON}" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; p=Path(sys.argv[1]); row={"phase":sys.argv[2],"event":sys.argv[3],"utc":datetime.now(timezone.utc).isoformat()}; f=p.open("a"); f.write(json.dumps(row,sort_keys=True)+"\n"); f.close()' "${OUTPUT}/phase_timestamps.jsonl" "$1" "$2"
}

echo "[bootstrap] utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) code=${CODE_ROOT} data=${DATA_ROOT} output=${OUTPUT}"
echo "[bootstrap] code_commit=${R1_CODE_COMMIT:-unknown_not_injected} id=$(id) host=$(hostname)"
"${PYTHON}" --version
nvidia-smi > "${OUTPUT}/nvidia_smi.txt"

"${PYTHON}" -c 'import hashlib,json,os,platform,sys; from pathlib import Path; out=Path(sys.argv[1]); root=Path(sys.argv[2]); selection=Path(sys.argv[3]); files=["scripts/r1_projective_medium_selector_v4_projection_frontier.py","scripts/r1_projective_medium_selector_v2.py","scripts/r1_action_graph.py","vagen/envs/active_spatial/canonical_camera.py","vagen/envs/active_spatial/canonical_task_metrics.py","vagen/envs/active_spatial/collision_detector.py","scripts/r1_projective_observability.py"]; sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); payload={"version":"r1_projection_frontier_ab_run_config_v1","hostname":platform.node(),"code_commit":os.environ.get("R1_CODE_COMMIT","unknown_not_injected"),"python":sys.executable,"python_version":sys.version,"selection":{"path":str(selection),"sha256":sha(selection)},"budgets":{"success_seed_cap":12,"per_seed_expansions":512,"candidate_cap_per_seed":48,"lower_bound_expansions":100000},"frontier_order":["depth_ascending","object_count_descending","in_front_count_descending","visible_count_descending","min_inside_frame_fraction_descending","min_clipped_bbox_area_ratio_descending","pair_alignment_cosine_descending","state_key_ascending"],"frozen_semantics":{"canonical_metric":"canonical_spatial_task_h1_v1","projective_margin_px":12,"projective_inside_frame_fraction":0.50,"fov_inside_frame_fraction":0.95,"medium":"complete no-success through depth 3 and first success by depth 6","observability":"projective_observability_v1_initial_keyframe","same_pair_only":True},"component_sha256":{name:sha(root/name) for name in files},"resource":{"worker_spec":"N4lS.Iq.I80.1","allocated_gpu":1,"renderer_gpu":0}}; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' "${OUTPUT}/run_config.json" "${CODE_ROOT}" "${SELECTION}"

mark baseline_diagnostic start
"${PYTHON}" scripts/r1_reverse_cap_structure_diagnostic.py run --selection "${SELECTION}" --source-inventory "${SOURCES}" --gs-root "${GS_ROOT}" --output-dir "${OUTPUT}/baseline_diagnostic"
mark baseline_diagnostic end

mark baseline_selector start
"${PYTHON}" scripts/r1_run_medium_selector_scene_shards.py --selection "${SELECTION}" --sources "${SOURCES}" --gs-root "${GS_ROOT}" --output-dir "${OUTPUT}/baseline_v2" --selector v2 --max-workers 5 --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 --lower-expansions 100000
mark baseline_selector end

mark frontier_selector start
"${PYTHON}" scripts/r1_run_medium_selector_scene_shards.py --selection "${SELECTION}" --sources "${SOURCES}" --gs-root "${GS_ROOT}" --output-dir "${OUTPUT}/projection_frontier_v4" --selector v4_projection_frontier --max-workers 5 --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 --lower-expansions 100000
mark frontier_selector end

for variant in baseline_v2 projection_frontier_v4; do
  mark "${variant}_materialize_runtime" start
  validation="${OUTPUT}/${variant}/independent_validation"
  "${PYTHON}" scripts/r1_materialize_difficulty_conditioned.py --selector "${OUTPUT}/${variant}/selector_results.json" --output-dir "${validation}"
  "${PYTHON}" scripts/r1_replay_projective_certificates.py --rows "${validation}/candidate_rows.jsonl" --reachability "${validation}/reachability_manifest.jsonl" --gs-root "${GS_ROOT}" --output "${validation}/runtime_replay.json"
  mark "${variant}_materialize_runtime" end
done

renderer_pid=''
cleanup() {
  status=$?
  if [[ -n "${renderer_pid}" ]]; then
    kill "${renderer_pid}" 2>/dev/null || true
    wait "${renderer_pid}" 2>/dev/null || true
  fi
  "${PYTHON}" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"exit_status":int(sys.argv[2]),"finished_utc":datetime.now(timezone.utc).isoformat()},indent=2)+"\n")' "${OUTPUT}/exit.json" "${status}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

mark renderer start
SCO_RENDER_SERVICE_TAG="r1_projection_frontier_ab_renderer_20260914" SCO_RENDER_STATE_DIR="${RENDER_STATE}" SCO_RENDER_ENDPOINT_FILE="${ENDPOINT_FILE}" SCO_RENDER_PORT="${RENDER_PORT}" SCO_RENDER_GPUS="0" SCO_RENDER_MAX_WORKERS="1" SCO_RENDER_MAX_INFLIGHT="2" SCO_RENDER_SKIP_WARMUP="1" bash examples/train/active_spatial/sco_renderer_8gpu_entry.sh &
renderer_pid=$!
for _ in $(seq 1 240); do
  [[ -s "${ENDPOINT_FILE}" ]] && break
  kill -0 "${renderer_pid}" 2>/dev/null || { echo "renderer exited before endpoint" >&2; exit 3; }
  sleep 2
done
[[ -s "${ENDPOINT_FILE}" ]] || { echo "renderer endpoint timeout" >&2; exit 4; }
renderer_url="$(<"${ENDPOINT_FILE}")/render"
for variant in baseline_v2 projection_frontier_v4; do
  mark "${variant}_official_rgb" start
  validation="${OUTPUT}/${variant}/independent_validation"
  "${PYTHON}" scripts/r1_projective_observability_audit.py --repaired "${validation}/candidate_rows.jsonl" --reachability "${validation}/reachability_manifest.jsonl" --gs-root "${GS_ROOT}" --renderer-url "${renderer_url}" --renderer-lock "${LOCK_FILE}" --output-dir "${OUTPUT}/${variant}/official_rgb"
  mark "${variant}_official_rgb" end
done
kill "${renderer_pid}" 2>/dev/null || true
wait "${renderer_pid}" 2>/dev/null || true
renderer_pid=''
mark renderer end

"${PYTHON}" scripts/r1_summarize_projection_frontier_ab.py \
  --selection "${SELECTION}" \
  --baseline-selector "${OUTPUT}/baseline_v2/selector_results.json" \
  --baseline-runtime "${OUTPUT}/baseline_v2/independent_validation/runtime_replay.json" \
  --baseline-rgb "${OUTPUT}/baseline_v2/official_rgb/observability_manifest.jsonl" \
  --baseline-diagnostic "${OUTPUT}/baseline_diagnostic/reverse_cap_diagnostic.json" \
  --frontier-selector "${OUTPUT}/projection_frontier_v4/selector_results.json" \
  --frontier-runtime "${OUTPUT}/projection_frontier_v4/independent_validation/runtime_replay.json" \
  --frontier-rgb "${OUTPUT}/projection_frontier_v4/official_rgb/observability_manifest.jsonl" \
  --phase-timestamps "${OUTPUT}/phase_timestamps.jsonl" --output "${OUTPUT}/aggregate_summary.json"

find "${OUTPUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUTPUT}/SHA256SUMS"
echo "[complete] ${OUTPUT}"
