#!/usr/bin/env bash
# Restore only actor HF exports needed for Active Spatial evaluation from AOSS.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
AOSS_DIR="${AOSS_DIR:-/mnt/umm/users/yinbaiqiao/aoss}"
CONF="${CONF:-${AOSS_DIR}/petreloss.conf}"
ADS_CLI="${ADS_CLI:-${AOSS_DIR}/ads-cli}"
HOST="${AOSS_API_HOST:-aoss-internal.cn-fz-01.fjscmsapi-oss.com}"
BUCKET="${BUCKET:-baiqiao}"
THREADS="${THREADS:-32}"
LISTERS="${LISTERS:-8}"
TARGET="${1:-all}"
LOG_DIR="${ROOT}/evaluation/sweeps/active_spatial/aoss_restore_logs"

export NO_PROXY="localhost,127.0.0.1,::1,${BUCKET}.${HOST},.${HOST},${HOST}"
export no_proxy="${NO_PROXY}"

AK="$(awk '
  /^\[fj\]/ {inside=1; next}
  /^\[/ {inside=0}
  inside && /^access_key[[:space:]]*=/ {sub(/^access_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
' "${CONF}")"
SK="$(awk '
  /^\[fj\]/ {inside=1; next}
  /^\[/ {inside=0}
  inside && /^secret_key[[:space:]]*=/ {sub(/^secret_key[[:space:]]*=[[:space:]]*/, ""); print; exit}
' "${CONF}")"

if [[ -z "${AK}" || -z "${SK}" ]]; then
  echo "[fatal] unable to read [fj] AOSS credentials from ${CONF}" >&2
  exit 1
fi
if [[ ! -x "${ADS_CLI}" ]]; then
  echo "[fatal] ads-cli unavailable: ${ADS_CLI}" >&2
  exit 1
fi

restore_one() {
  local experiment="$1"
  local step="$2"
  local relative="${experiment}/checkpoints/global_step_${step}/actor/huggingface"
  local source="s3://${AK}:${SK}@${BUCKET}.${HOST}/VAGEN-Lite/exps/vagen_active_spatial/${relative}/"
  local destination="${ROOT}/exps/vagen_active_spatial/${relative}/"
  local safe_name="${experiment}_step${step}"
  local log="${LOG_DIR}/${safe_name}.log"

  mkdir -p "${destination}" "${LOG_DIR}"
  echo "[restore] ${experiment} step=${step}"
  echo "[restore] destination=${destination}"
  "${ADS_CLI}" -p "${THREADS}" -l "${LISTERS}" \
    --conntimeout 120 --timeout 600 \
    sync "${source}" "${destination}" >"${log}" 2>&1

  local has_weight=0
  if find "${destination}" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name 'pytorch_model.bin' -o -name 'tf_model.h5' \
    -o -name 'flax_model.msgpack' -o -name 'model.safetensors.index.json' \
    -o -name 'pytorch_model.bin.index.json' \) -print -quit | grep -q .; then
    has_weight=1
  fi
  if [[ ! -f "${destination}/config.json" || "${has_weight}" -ne 1 ]]; then
    echo "[fatal] incomplete restored HF export: ${relative}; see ${log}" >&2
    exit 1
  fi
  echo "[restore] complete ${experiment} step=${step}"
}

case "${TARGET}" in
  v50)
    restore_one qwen_v50_clean_d0pass_20260824_full 50
    restore_one qwen_v50_clean_d0pass_20260824_full 100
    restore_one qwen_v50_clean_d0pass_20260824_full 700
    ;;
  c8)
    restore_one cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full 50
    restore_one cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full 100
    restore_one cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full 1000
    ;;
  c4)
    restore_one cambrian_c4_clean_d0pass_20260822_h800_r5_full 960
    ;;
  all)
    restore_one "qwen_v50_clean_d0pass_20260824_full" 50
    restore_one "qwen_v50_clean_d0pass_20260824_full" 100
    restore_one "qwen_v50_clean_d0pass_20260824_full" 700
    restore_one "cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full" 50
    restore_one "cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full" 100
    restore_one "cambrian_c8_clean_d0pass_20260824_sco_renderer_r3_full" 1000
    restore_one "cambrian_c4_clean_d0pass_20260822_h800_r5_full" 960
    ;;
  *)
    echo "usage: $0 {all|v50|c8|c4}" >&2
    exit 2
    ;;
esac
