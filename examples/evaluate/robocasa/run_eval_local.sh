#!/usr/bin/env bash
# In-process RoboCasa eval using the policy venv python.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${1:-$SCRIPT_DIR/config_local.yaml}"
if [[ "${1:-}" != "" ]]; then
  shift
fi

ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365}"
POLICY_VENV="${POLICY_VENV:-$ROBOCASA365_ROOT/policies/diffusion_policy/.venv}"
ROBOCASA_SRC="${ROBOCASA_SRC:-$ROBOCASA365_ROOT/robocasa}"
ROBOMIMIC_SRC="${ROBOMIMIC_SRC:-$ROBOCASA365_ROOT/robomimic}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHONPATH_PARTS=("${REPO_ROOT}")
[[ -d "${ROBOCASA_SRC}" ]] && PYTHONPATH_PARTS+=("${ROBOCASA_SRC}")
[[ -d "${ROBOMIMIC_SRC}" ]] && PYTHONPATH_PARTS+=("${ROBOMIMIC_SRC}")
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "${POLICY_VENV}/bin/python" ]]; then
  PYTHON_BIN="${POLICY_VENV}/bin/python"
else
  PYTHON_BIN="${ROBOCASA_PYTHON:-python}"
fi

echo "[robocasa] local eval python=${PYTHON_BIN}"
exec "${PYTHON_BIN}" -m vagen.evaluate.run_eval --config "${CONFIG}" "$@"
