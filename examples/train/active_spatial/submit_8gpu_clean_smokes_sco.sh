#!/usr/bin/env bash
# Submit bounded eight-GPU acceptance smokes through the established SCO helper.
set -euo pipefail

ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
SUBMIT_SH="/mnt/umm/users/yinbaiqiao/submit.sh"
TARGET="${1:-both}"

submit_one() {
  local anchor="$1" job_name entry
  case "${anchor}" in
    qwen)
      job_name="qwen_v46_clean_d0pass_20260819_8gpu_smoke"
      entry="${ROOT}/examples/train/active_spatial/sco_qwen_8gpu_smoke_entry.sh"
      ;;
    cambrian)
      job_name="cambrian_c8_clean_d0pass_20260819_8gpu_smoke"
      entry="${ROOT}/examples/train/active_spatial/sco_cambrian_8gpu_smoke_entry.sh"
      ;;
    *) echo "usage: $0 {qwen|cambrian|both}" >&2; exit 2 ;;
  esac
  bash "${SUBMIT_SH}" h800 1 8 "${job_name}" "${entry}"
}

case "${TARGET}" in
  qwen) submit_one qwen ;;
  cambrian) submit_one cambrian ;;
  both) submit_one qwen; submit_one cambrian ;;
  *) echo "usage: $0 {qwen|cambrian|both}" >&2; exit 2 ;;
esac
