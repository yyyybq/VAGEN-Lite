#!/usr/bin/env bash
# Poll SCO state and on-disk activity for selected Active Spatial eval sweeps.
set -euo pipefail

ROOT="${ROOT:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite}"
WORKSPACE="${WORKSPACE:-aigc}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
ONCE=0

usage() {
  echo "usage: $0 [--once] JOB_ID:SWEEP_NAME [...]" >&2
}

if [[ "${1:-}" == "--once" ]]; then
  ONCE=1
  shift
fi
if [[ "$#" -eq 0 ]]; then
  usage
  exit 2
fi

export PATH="${HOME}/.sco/bin:${PATH}"
cd "${ROOT}"

report_one() {
  local spec="$1" job_id sweep state nav_count qa_status newest
  job_id="${spec%%:*}"
  sweep="${spec#*:}"
  state="$(sco acp jobs describe --workspace-name="${WORKSPACE}" -o json "${job_id}" \
    | sed -n 's/^[[:space:]]*"state": "\([^"]*\)".*/\1/p' | head -1 || true)"
  nav_count="$(find "evaluation/sweeps/active_spatial/${sweep}" -path '*/model/results_model.json' \
    -type f 2>/dev/null | wc -l)"
  qa_status="missing"
  if [[ -f "evaluation/sweeps/active_spatial/${sweep}/qa_parallel_completion.json" ]]; then
    qa_status="$(tr '\n' ' ' < "evaluation/sweeps/active_spatial/${sweep}/qa_parallel_completion.json")"
  fi
  newest="$(find "evaluation/sweeps/active_spatial/${sweep}" -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p\n' \
    2>/dev/null | sort | tail -1 || true)"
  printf 'utc=%s job=%s state=%s nav_results=%s qa=%s newest=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${job_id}" "${state:-UNKNOWN}" \
    "${nav_count}" "${qa_status}" "${newest:-none}"
}

while true; do
  for spec in "$@"; do
    report_one "${spec}"
  done
  if [[ "${ONCE}" -eq 1 ]]; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
