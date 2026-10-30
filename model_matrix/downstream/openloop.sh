#!/usr/bin/env bash
# The open-loop check (model_matrix/downstream/openloop.py, the user's decision 34 of 2026-10-30) in StarVLA's env,
# as the policy server of eval_ckpt.sh runs: from StarVLA's root, through the Cambrian backbone entry for a Cambrian-S
# start (cambrian_env.sh), plainly for any other.  One group of checkpoints of one backbone per call.
# Usage: [EVAL_DATA=FOLDER] openloop.sh START_PATH OUT.json DATA_CONFIG LABEL=CKPT [LABEL=CKPT ...]   (writes only
# OUT.json, and with EVAL_DATA the statistics / index caches StarVLA keeps in that folder's meta/; decision 35)
set -uo pipefail
START="${1:?Usage: openloop.sh START_PATH OUT.json DATA_CONFIG LABEL=CKPT...}"
OUT="${2:?}"; DATA="${3:?}"; shift 3
DS=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream
MWT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-models
STARVLA_ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
CKPTS=()
for c in "$@"; do CKPTS+=(--ckpt "$c"); done
[[ -n "${EVAL_DATA:-}" ]] && CKPTS+=(--eval-data "$EVAL_DATA")
[[ -n "${WAYS:-}" ]] && CKPTS+=(--ways "$WAYS")  # decision 36: e.g. images,jpeg,blur,color
cd "$STARVLA_ROOT"
source "$MWT/examples/train/robocasa/setup_starvla_env.sh" > /dev/null
set +e
export PYTHONPATH="$DS:$STARVLA_ROOT"
source "$MWT/model_matrix/downstream/cambrian_env.sh"
starvla_entry "$START"
echo "[openloop] $(date -u +%FT%TZ) start=$START out=$OUT"
python ${ENTRY[@]+"${ENTRY[@]}"} "$MWT/model_matrix/downstream/openloop.py" --data-config "$DATA" "${CKPTS[@]}" --out "$OUT"
rc=$?
echo "[openloop] $(date -u +%FT%TZ) rc=$rc"
exit $rc
