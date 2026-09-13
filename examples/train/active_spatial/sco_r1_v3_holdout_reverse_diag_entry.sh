#!/usr/bin/env bash
# Frozen independent v2/v3 comparison plus reverse-cap diagnostics on one task worker.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
GS_ROOT="${GS_ROOT:-/mnt/umm/users/yinbaiqiao/InteriorGS}"
PYTHON="${PYTHON:-/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python}"
BASE="${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905"
OUTPUT="${R1_OUTPUT:-${BASE}/projective_medium_v3_independent_reverse_diag_20260913}"
V2="${BASE}/projective_difficulty_conditioned_medium_v2_canary10_788_20260911"
V3="${BASE}/projective_medium_v3_depth_banded_diagnostic24_20260913"
SOURCE_INVENTORY="${BASE}/path_first_canary10_v1/frozen_selection.json"
LEDGER="${BASE}/required_scene_ledger.json"
RENDER_PORT="${R1_RENDER_PORT:-8771}"
RENDER_STATE="${OUTPUT}/renderer"
ENDPOINT_FILE="${RENDER_STATE}/endpoint.txt"
LOCK_FILE="${OUTPUT}/official_rgb/renderer.lock"

export PYTHONDONTWRITEBYTECODE=1
export NO_PROXY='*'
export no_proxy='*'
cd "${ROOT}"
mkdir -p "${OUTPUT}"

if [[ ! -f "${OUTPUT}/run_environment.json" ]]; then
  "${PYTHON}" -c 'import json,os,platform,subprocess,sys; from pathlib import Path; out=Path(sys.argv[1]); payload={"version":"r1_v3_holdout_reverse_diag_environment_v1","hostname":platform.node(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"python":sys.executable,"python_version":sys.version,"gs_root":sys.argv[2],"renderer":{"port":int(sys.argv[3]),"requested_gpu_ids":[0],"max_workers":1,"max_inflight":2},"sco_resource":{"worker_spec":"N4lS.Iq.I80.8","allocated_gpus":8,"renderer_gpus_used":1,"reason":"current SCO template minimum"}}; out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")' "${OUTPUT}/run_environment.json" "${GS_ROOT}" "${RENDER_PORT}"
fi
nvidia-smi > "${OUTPUT}/nvidia_smi.txt"

INVENTORY_DIR="${OUTPUT}/verified_medium75_inventory"
if [[ ! -f "${INVENTORY_DIR}/verified_medium_inventory.json" ]]; then
  "${PYTHON}" scripts/r1_freeze_verified_medium_inventory.py \
    --frozen-source-inventory "${SOURCE_INVENTORY}" \
    --v2-candidate "${V2}/independent_validation_v1/candidate_rows.jsonl" \
    --v2-reachability "${V2}/independent_validation_v1/reachability_manifest.jsonl" \
    --v2-runtime "${V2}/independent_validation_v1/runtime_replay.json" \
    --v2-observability "${V2}/official_observability_v1_sco_20260912/observability_manifest.jsonl" \
    --v3-candidate "${V3}/independent_validation/candidate_rows.jsonl" \
    --v3-reachability "${V3}/independent_validation/reachability_manifest.jsonl" \
    --v3-runtime "${V3}/independent_validation/runtime_replay.json" \
    --v3-observability "${V3}/official_observability_v1_sco_20260913/observability_manifest.jsonl" \
    --output-dir "${INVENTORY_DIR}"
fi

HOLDOUT="${OUTPUT}/holdout128_frozen_selection.json"
if [[ ! -f "${HOLDOUT}" ]]; then
  "${PYTHON}" scripts/r1_freeze_independent_v3_holdout.py \
    --ledger "${LEDGER}" --source-inventory "${SOURCE_INVENTORY}" --output "${HOLDOUT}" \
    --scene-count 5 --source-cap 128 --minimum-projective-per-scene 20
fi

REVERSE_SELECTION="${OUTPUT}/reverse_cap16_controls4_frozen_selection.json"
if [[ ! -f "${REVERSE_SELECTION}" ]]; then
  "${PYTHON}" scripts/r1_reverse_cap_structure_diagnostic.py freeze \
    --selector "${V2}/selector_results.json" \
    --observability "${V2}/official_observability_v1_sco_20260912/observability_manifest.jsonl" \
    --output "${REVERSE_SELECTION}"
fi

if [[ ! -f "${OUTPUT}/reverse_cap_diagnostic/reverse_cap_diagnostic.json" ]]; then
  "${PYTHON}" scripts/r1_reverse_cap_structure_diagnostic.py run \
    --selection "${REVERSE_SELECTION}" --source-inventory "${SOURCE_INVENTORY}" \
    --gs-root "${GS_ROOT}" --output-dir "${OUTPUT}/reverse_cap_diagnostic"
fi

if [[ ! -f "${OUTPUT}/holdout_v2_screen/selector_results.json" ]]; then
  "${PYTHON}" scripts/r1_run_medium_selector_scene_shards.py \
    --selection "${HOLDOUT}" --sources "${HOLDOUT}" --gs-root "${GS_ROOT}" \
    --output-dir "${OUTPUT}/holdout_v2_screen" --selector v2 --max-workers 5 \
    --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 --lower-expansions 100000
fi

AB_SELECTION="${OUTPUT}/holdout_v3_ab_frozen_selection.json"
if [[ ! -f "${AB_SELECTION}" ]]; then
  "${PYTHON}" scripts/r1_select_holdout_v3_ab.py \
    --holdout-selection "${HOLDOUT}" \
    --v2-selector "${OUTPUT}/holdout_v2_screen/selector_results.json" \
    --output "${AB_SELECTION}" --shortcut-min 24 --shortcut-max 32 --positive-controls 5
fi

if [[ ! -f "${OUTPUT}/holdout_v3_ab/selector_results.json" ]]; then
  "${PYTHON}" scripts/r1_run_medium_selector_scene_shards.py \
    --selection "${AB_SELECTION}" --sources "${AB_SELECTION}" --gs-root "${GS_ROOT}" \
    --output-dir "${OUTPUT}/holdout_v3_ab" --selector v3 --max-workers 5 \
    --seed-cap 12 --per-seed-expansions 512 --candidate-cap-per-seed 48 --lower-expansions 100000
fi

for selector_name in holdout_v2_screen holdout_v3_ab; do
  validation_dir="${OUTPUT}/${selector_name}/independent_validation"
  if [[ ! -f "${validation_dir}/summary.json" ]]; then
    "${PYTHON}" scripts/r1_materialize_difficulty_conditioned.py \
      --selector "${OUTPUT}/${selector_name}/selector_results.json" --output-dir "${validation_dir}"
  fi
  if [[ ! -f "${validation_dir}/runtime_replay.json" ]]; then
    "${PYTHON}" scripts/r1_replay_projective_certificates.py \
      --rows "${validation_dir}/candidate_rows.jsonl" \
      --reachability "${validation_dir}/reachability_manifest.jsonl" \
      --gs-root "${GS_ROOT}" --output "${validation_dir}/runtime_replay.json"
  fi
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

if [[ ! -f "${OUTPUT}/holdout_v2_screen/official_rgb/summary.json" || ! -f "${OUTPUT}/holdout_v3_ab/official_rgb/summary.json" ]]; then
  SCO_RENDER_SERVICE_TAG="r1_v3_holdout_reverse_diag_renderer_20260913" \
  SCO_RENDER_STATE_DIR="${RENDER_STATE}" SCO_RENDER_ENDPOINT_FILE="${ENDPOINT_FILE}" \
  SCO_RENDER_PORT="${RENDER_PORT}" SCO_RENDER_GPUS="0" SCO_RENDER_MAX_WORKERS="1" \
  SCO_RENDER_MAX_INFLIGHT="2" SCO_RENDER_SKIP_WARMUP="1" \
    bash examples/train/active_spatial/sco_renderer_8gpu_entry.sh &
  renderer_pid=$!
  for _ in $(seq 1 240); do
    [[ -s "${ENDPOINT_FILE}" ]] && break
    kill -0 "${renderer_pid}" 2>/dev/null || { echo "renderer exited before endpoint" >&2; exit 3; }
    sleep 2
  done
  [[ -s "${ENDPOINT_FILE}" ]] || { echo "renderer endpoint timeout" >&2; exit 4; }
  renderer_url="$(cat "${ENDPOINT_FILE}")/render"
  for selector_name in holdout_v2_screen holdout_v3_ab; do
    validation_dir="${OUTPUT}/${selector_name}/independent_validation"
    rgb_dir="${OUTPUT}/${selector_name}/official_rgb"
    if [[ ! -f "${rgb_dir}/summary.json" ]]; then
      "${PYTHON}" scripts/r1_projective_observability_audit.py \
        --repaired "${validation_dir}/candidate_rows.jsonl" \
        --reachability "${validation_dir}/reachability_manifest.jsonl" \
        --gs-root "${GS_ROOT}" --renderer-url "${renderer_url}" \
        --renderer-lock "${LOCK_FILE}" --output-dir "${rgb_dir}"
    fi
  done
  kill "${renderer_pid}" 2>/dev/null || true
  wait "${renderer_pid}" 2>/dev/null || true
  renderer_pid=''
fi

find "${OUTPUT}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${OUTPUT}/SHA256SUMS"
echo "[complete] ${OUTPUT}"
