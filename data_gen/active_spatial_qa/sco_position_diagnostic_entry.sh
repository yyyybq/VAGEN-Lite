#!/usr/bin/env bash
set -euo pipefail
umask 002
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
OUT=${POSITION_DIAG_OUT:-${ROOT}/data_gen/active_spatial_qa/artifacts/position_repair_v6_20261002}
export POSITION_DIAG_OUT="${OUT}"
mkdir -p "${OUT}"
exec > >(tee -a "${OUT}/worker.log") 2>&1
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PATH="${ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export VAGEN_GSPLAT_PREBUILT=1
export TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=${ENV}
export TORCH_CUDA_ARCH_LIST=9.0
export CPATH="${ENV}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
trap 'printf "{\"exit_status\":%d}\n" "$?" > "${OUT}/worker_exit.json"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv
POSITION_DIAG_OUT="${OUT}" python -m data_gen.active_spatial_qa.restore_diagnostic_scene
python -m data_gen.active_spatial_qa.position_render_run
