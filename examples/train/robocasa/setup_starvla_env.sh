#!/usr/bin/env bash
# Isolated official StarVLA stack (Python 3.10 / torch 2.6+cu124).
# Do not mix with vagen-lite (3.12) or kitchen (3.13).
#
#   source examples/train/robocasa/setup_starvla_env.sh
set -euo pipefail

STARVLA_ENV="${STARVLA_ENV:-/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA}"
export PATH="${STARVLA_ENV}/bin:${PATH}"
export CC="${STARVLA_ENV}/bin/gcc"
export CXX="${STARVLA_ENV}/bin/g++"
export CUDA_HOME="${STARVLA_ENV}"
export PYTHONPATH="${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

if [[ ! -x "${STARVLA_ENV}/bin/python" ]]; then
  echo "[fatal] starVLA env missing: ${STARVLA_ENV}" >&2
  return 2 2>/dev/null || exit 2
fi

echo "[starVLA] python=${STARVLA_ENV}/bin/python"
echo "[starVLA] CUDA_HOME=${CUDA_HOME} nvcc=$(${STARVLA_ENV}/bin/nvcc -V 2>/dev/null | awk '/release/{print $5,$6}')"
