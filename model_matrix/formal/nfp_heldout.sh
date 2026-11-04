#!/usr/bin/env bash
# Decision 48 甲 (2026-11-04): model_matrix/formal/nfp_heldout.py for one or more checkpoints as one 1-card sco job
# (model_matrix.sco_job).  Usage: nfp_heldout.sh OUT_DIR LABEL=CKPT [LABEL=CKPT ...]  ->  OUT_DIR/LABEL/pairs.jsonl,
# OUT_DIR/LABEL/summary.json, OUT_DIR/logs/.  LIMIT=N scores only the first N episodes (a check).  Caches go to
# OUT_DIR/tmp (removed on exit); the job runs as root, so OUT_DIR is handed back to its owner on exit.
set -uo pipefail
OUT=${1:?Usage: nfp_heldout.sh OUT_DIR LABEL=CKPT...}; shift
WT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
mkdir -p "${OUT}/logs" "${OUT}/tmp"
OWNER=$(stat -c %u:%g "${OUT}")
trap 'status=$?; rm -rf "${OUT}/tmp"; printf "{\"exit_status\":%s,\"ended_utc\":\"%s\"}\n" "${status}" "$(date -u +%FT%TZ)" > "${OUT}/logs/exit.json"; chown -R "${OWNER}" "${OUT}" 2>/dev/null || true; exit ${status}' EXIT
exec > >(tee -a "${OUT}/logs/job-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "${WT}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH="${WT}"
export NO_PROXY='*' no_proxy='*' TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${OUT}/tmp/hf_home" XDG_CACHE_HOME="${OUT}/tmp/xdg"
# the model cache (SigLIP2) and the Cambrian source: the trainer's, set by nfp_heldout.training_env (model_types)
git -C "${WT}" rev-parse HEAD || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
rc=0
for spec in "$@"; do
  label=${spec%%=*}; ckpt=${spec#*=}
  echo "=== $(date -u +%FT%TZ) ${label} ${ckpt}"
  "${PY}" -m model_matrix.formal.nfp_heldout --model "${ckpt}" --out "${OUT}/${label}" ${LIMIT:+--limit "${LIMIT}"}
  r=$?
  [[ ${r} -ne 0 ]] && rc=${r}
  echo "=== $(date -u +%FT%TZ) ${label} exit ${r}"
done
exit ${rc}
