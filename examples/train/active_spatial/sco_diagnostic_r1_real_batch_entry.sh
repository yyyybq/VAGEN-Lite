#!/usr/bin/env bash
# Authorized one-batch H800 capture. No optimizer step is allowed.
set -euo pipefail
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
OUT=${ROOT}/exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5
SNAPSHOT_ROOT=${OUT}/snapshot
CRITIC_INIT=${OUT}/critic_initial_pre_update
PORT=8898
MODEL=${ROOT}/exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898
MODEL_SHA=46f05ffcc6127a4caa9a3e8c11ddf298b9a5263c8680afe6b5d017ea91702c8b
[[ ! -e "${SNAPSHOT_ROOT}" && ! -e "${CRITIC_INIT}" ]] || { echo existing_output >&2; exit 2; }
mkdir -p "${OUT}"
cd "${ROOT}"
export PATH=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin:${PATH}
export PYTHONPATH=${ROOT} HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 NO_PROXY='*' no_proxy='*'
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR="${SNAPSHOT_ROOT}"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_STOP_AFTER_WRITE=1
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_ACTOR_CHECKPOINT_PATH="${MODEL}"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_ACTOR_CHECKPOINT_SHA256="${MODEL_SHA}"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_REFERENCE_CHECKPOINT_PATH="${MODEL}"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_REFERENCE_CHECKPOINT_SHA256="${MODEL_SHA}"
export VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_CRITIC_INITIAL_EXPORT_DIR="${CRITIC_INIT}"
cleanup() {
  code=$?
  [[ -z "${renderer_pid:-}" ]] || { kill "${renderer_pid}" 2>/dev/null || true; wait "${renderer_pid}" 2>/dev/null || true; }
  printf '{"exit_status":%s}\n' "${code}" > "${OUT}/worker_exit.json"
  exit "${code}"
}
trap cleanup EXIT INT TERM
bash examples/train/active_spatial/start_gs_render_http_service.sh --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS --host 127.0.0.1 --port "${PORT}" --gpus 7 --max-workers 1 --max-inflight 1 --admit-timeout 300 --conda-env /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite > "${OUT}/renderer.log" 2>&1 &
renderer_pid=$!
for _ in $(seq 1 180); do
  curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" > "${OUT}/renderer_health.json" && break
  kill -0 "${renderer_pid}" 2>/dev/null || { echo renderer_failed >&2; exit 3; }
  sleep 2
done
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null
"${PY}" scripts/active_spatial_r1_rgb_runtime_preflight.py --policy-jsonl docs/diagnosis/active_spatial_dense_score_diagnostic_real_batch_20260916/policy_input_rows.jsonl --audit-jsonl docs/diagnosis/active_spatial_dense_score_diagnostic_real_batch_20260916/audit_manifest.jsonl --renderer-url "http://127.0.0.1:${PORT}/render" --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS --output "${OUT}/runtime_rgb_preflight.json"
bash examples/train/active_spatial/run_experiment.sh examples/train/active_spatial/experiments/diagnostic_r1_real_batch_no_update.sh > "${OUT}/capture.log" 2>&1
[[ -f "${SNAPSHOT_ROOT}/global_step_000000_pre_update/manifest.json" ]] || { echo snapshot_missing >&2; exit 4; }
grep -q 'stopping before critic/actor optimizer steps' "${OUT}/capture.log" || { echo no_update_proof_missing >&2; exit 4; }
