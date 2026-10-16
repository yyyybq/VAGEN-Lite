#!/usr/bin/env bash
# H800 submission entry for metadata-only AOSS staging, complete train-pair
# enumeration, and deterministic Phase-B request freezing.  No trajectories,
# renderer, model checkpoint, policy, or training process is launched here.
set -euo pipefail

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
BASE=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905
RUN=${BASE}/r1_train_pair_universe_v1_20260928
PHASE_A=${BASE}/r1_fullscale_canonical_expansion_scope_v1_20260928
PHASE_B=${BASE}/r1_fullscale_canonical_expansion_phase_b_scope_v1_20260928
INVENTORY=${BASE}/r1_full_clean_target_inventory_v1_20260928
CORPUS=${BASE}/r1_clean_training_corpus_v1_20260919
SALVAGE=${BASE}/r1_old_v46_target_salvage_v1_20260919/salvage_matrix
DEV=${BASE}/r1_canonical_dev_eval32_20260915/frozen_input/development_regression_manifest.jsonl
LOCAL=${BASE}/r1_independent_local_action_eval60_v3_20260916/frozen_input/parent_sources.jsonl
OLD=${ROOT}/exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl
ARCHIVE=${R1_PAIR_UNIVERSE_CODE_ARCHIVE:?required}
EXPECTED_ARCHIVE_SHA=${R1_PAIR_UNIVERSE_CODE_ARCHIVE_SHA256:?required}
EXPECTED_COMMIT=${R1_PAIR_UNIVERSE_COMMIT:?required}
WORK=$(mktemp -d /tmp/r1_pair_universe_code.XXXXXXXX)

cleanup() {
  status=$?
  printf '{"exit_status":%s,"stopped_utc":"%s"}\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
  [[ "${WORK}" == /tmp/r1_pair_universe_code.* ]] && rm -rf -- "${WORK}"
  exit "${status}"
}
trap cleanup EXIT INT TERM
mkdir -p "${RUN}"
actual_sha=$(sha256sum "${ARCHIVE}" | awk '{print $1}')
[[ "${actual_sha}" == "${EXPECTED_ARCHIVE_SHA}" ]] || { echo archive_sha_mismatch >&2; exit 2; }
tar -xzf "${ARCHIVE}" -C "${WORK}"
cd "${WORK}"
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${WORK}:${WORK}/scripts
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 NO_PROXY='*' no_proxy='*'
{
  echo "hostname=$(hostname)"; echo "expected_commit=${EXPECTED_COMMIT}"; echo "archive_sha256=${actual_sha}"; "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  echo required_cluster=H800; echo required_pool=h800; echo resource=N4lS.Iq.I80.1; echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RUN}/worker_environment.txt"

"${PY}" scripts/r1_stage_aoss_scene_metadata.py \
  --old-train "${OLD}" --canonical-eval "${DEV}" --local-action-parents "${LOCAL}" \
  --cache-root "${RUN}/metadata_cache" --ledger "${RUN}/metadata_ledger.json"

"${PY}" scripts/r1_enumerate_train_pair_universe.py \
  --metadata-root "${RUN}/metadata_cache/ready" --old-train "${OLD}" \
  --canonical-eval "${DEV}" --local-action-parents "${LOCAL}" \
  --exclude-manifest "${CORPUS}/repaired_canonical_train_manifest.jsonl" \
  --exclude-manifest "${SALVAGE}/projective_repairable_runtime_rgb_accepted.jsonl" \
  --exclude-manifest "${SALVAGE}/fov_repairable_rgb_v2_provisional.jsonl" \
  --phase-a-sources "${PHASE_A}/fresh_sources.jsonl" --output-dir "${RUN}/universe"

"${PY}" scripts/r1_prepare_phase_b_from_pair_universe.py \
  --universe "${RUN}/universe/train_pair_universe.jsonl" --old-train "${OLD}" \
  --projective-inventory "${INVENTORY}/projective_train_ready_deduplicated.jsonl" \
  --fov-provisional-inventory "${INVENTORY}/fov_provisional_deduplicated.jsonl" \
  --projective-requests 1791 --fov-requests 1309 --output-dir "${PHASE_B}"

find "${RUN}" "${PHASE_B}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN}/SHA256SUMS"
printf '{"exit_status":0,"stopped_utc":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN}/worker_exit.json"
trap - EXIT INT TERM
[[ "${WORK}" == /tmp/r1_pair_universe_code.* ]] && rm -rf -- "${WORK}"
