#!/usr/bin/env bash
# Remote RoboCasa eval (VAGEN python talks to the policy-venv server).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config.yaml}"
if [[ "${1:-}" != "" ]]; then
  shift
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
VAGEN_PYTHON="${VAGEN_PYTHON:-python}"

LOG_FILE="${LOG_FILE:-run.log}"
"${VAGEN_PYTHON}" -m vagen.evaluate.run_eval --config "${CONFIG}" "$@" \
  2>&1 | tee "${LOG_FILE}"
