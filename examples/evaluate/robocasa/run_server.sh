#!/usr/bin/env bash
# Start the RoboCasa GymService in the policy venv (Python 3.10).
# Do not activate the kitchen 3.13 venv here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365}"
POLICY_VENV="${POLICY_VENV:-$ROBOCASA365_ROOT/policies/diffusion_policy/.venv}"
ROBOCASA_SRC="${ROBOCASA_SRC:-$ROBOCASA365_ROOT/robocasa}"
ROBOMIMIC_SRC="${ROBOMIMIC_SRC:-$ROBOCASA365_ROOT/robomimic}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

PYTHONPATH_PARTS=("${REPO_ROOT}")
if [[ -d "${ROBOCASA_SRC}" ]]; then
  PYTHONPATH_PARTS+=("${ROBOCASA_SRC}")
fi
if [[ -d "${ROBOMIMIC_SRC}" ]]; then
  PYTHONPATH_PARTS+=("${ROBOMIMIC_SRC}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}")${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "${POLICY_VENV}/bin/python" ]]; then
  PYTHON_BIN="${POLICY_VENV}/bin/python"
  # shellcheck disable=SC1091
  source "${POLICY_VENV}/bin/activate"
else
  echo "[warn] Policy venv not found at ${POLICY_VENV}; using current python." >&2
  PYTHON_BIN="${ROBOCASA_PYTHON:-python}"
fi

PORT="${PORT:-8010}"
echo "[robocasa] server python=${PYTHON_BIN} port=${PORT}"
echo "[robocasa] PYTHONPATH=${PYTHONPATH}"
exec "${PYTHON_BIN}" -m vagen.envs.robocasa.serve --port "${PORT}" "$@"
