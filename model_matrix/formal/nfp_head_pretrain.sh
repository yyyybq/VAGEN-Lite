#!/usr/bin/env bash
# Decision 54 丙 (2026-11-05): model_matrix/formal/nfp_head_pretrain.py as one 1-card sco job (model_matrix.sco_job),
# then, when its gate passed, nfp_heldout.py on the start model it built (the RL checkpoints' own held-out script:
# the head loads from OUT/start as the RL trainer will load it).  Usage: nfp_head_pretrain.sh OUT_DIR [ARGS...]
# (ARGS go to nfp_head_pretrain) -> OUT_DIR/run/ (head.safetensors, history.json, heldout/, start/),
# OUT_DIR/heldout_start/, OUT_DIR/logs/.  LIMIT=N reads only the first N episodes of each file (a check).  Caches go
# to OUT_DIR/tmp (removed on exit); the job runs as root, so OUT_DIR is handed back to its owner on exit.
set -uo pipefail
OUT=${1:?Usage: nfp_head_pretrain.sh OUT_DIR [ARGS...]}; shift
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
echo "=== $(date -u +%FT%TZ) pretrain"
"${PY}" -m model_matrix.formal.nfp_head_pretrain --out "${OUT}/run" ${LIMIT:+--limit "${LIMIT}"} "$@"
rc=$?
echo "=== $(date -u +%FT%TZ) pretrain exit ${rc}"
if [[ ${rc} -eq 0 && -d "${OUT}/run/start" ]]; then
  echo "=== $(date -u +%FT%TZ) held-out on the start model"
  "${PY}" -m model_matrix.formal.nfp_heldout --model "${OUT}/run/start" --out "${OUT}/heldout_start" ${LIMIT:+--limit "${LIMIT}"}
  rc=$?
  echo "=== $(date -u +%FT%TZ) held-out exit ${rc}"
fi
exit ${rc}
