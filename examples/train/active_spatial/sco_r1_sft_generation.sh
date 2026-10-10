#!/usr/bin/env bash
# Generate the complete gate-aligned R1 SFT corpus and visualization on one H800.
set -euo pipefail

[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || {
  echo "incorrect artifact owner: $(id -u):$(id -g)" >&2
  exit 3
}

ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
RUN=${ROOT}/exps/vagen_active_spatial/R1-gate-aligned-sft-v2-20261008
PACKAGE=${RUN}/package
OUTPUT=${RUN}/output
ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
PY=${ENV}/bin/python
SOURCE=${ROOT}/exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/train.jsonl
AUDIT=${ROOT}/exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/audit_only.jsonl
ENV_YAML=${ROOT}/exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005/frozen/train.yaml
ASSETS=${ROOT}/exps/vagen_active_spatial/R1-clean-Projective-v0/assets/ready

[[ ! -e ${OUTPUT} ]] || { echo "refusing existing output: ${OUTPUT}" >&2; exit 4; }
[[ -f ${SOURCE} && -f ${AUDIT} && -f ${ENV_YAML} && -d ${ASSETS} ]]

export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export NO_PROXY='*' no_proxy='*' WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=${RUN}/torch_extensions
export TORCH_CUDA_ARCH_LIST=9.0
export CC=${ENV}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${ENV}/bin/x86_64-conda-linux-gnu-g++

configure_cuda_devel() {
  local root header library
  local wheel_root=${ENV}/lib/python3.12/site-packages/nvidia/curand
  if [[ -x ${ENV}/bin/nvcc && -f ${wheel_root}/include/curand.h ]] && \
     compgen -G "${wheel_root}/lib/libcurand.so*" >/dev/null; then
    export CUDA_HOME=${ENV}
    export CURAND_INCLUDE_DIR=${wheel_root}/include
    export CURAND_LIBRARY_DIR=${wheel_root}/lib
  else
    for root in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12.6 \
                /usr/local/cuda-12.4 /usr/local/cuda-12.1; do
      [[ -x ${root}/bin/nvcc ]] || continue
      if [[ -f ${root}/targets/x86_64-linux/include/curand.h ]]; then
        header=${root}/targets/x86_64-linux/include
      elif [[ -f ${root}/include/curand.h ]]; then
        header=${root}/include
      else
        continue
      fi
      if compgen -G "${root}/targets/x86_64-linux/lib/libcurand.so*" >/dev/null; then
        library=${root}/targets/x86_64-linux/lib
      elif compgen -G "${root}/lib64/libcurand.so*" >/dev/null; then
        library=${root}/lib64
      else
        continue
      fi
      export CUDA_HOME=${root}
      export CURAND_INCLUDE_DIR=${header}
      export CURAND_LIBRARY_DIR=${library}
      break
    done
  fi
  [[ -n ${CUDA_HOME:-} && -f ${CURAND_INCLUDE_DIR:-}/curand.h ]]
  compgen -G "${CURAND_LIBRARY_DIR}/libcurand.so*" >/dev/null
  export PATH=${CUDA_HOME}/bin:${ENV}/bin:${PATH}
  export CPATH=${CURAND_INCLUDE_DIR}:${ENV}/targets/x86_64-linux/include:${CPATH:-}
  export LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=${CURAND_LIBRARY_DIR}:${ENV}/lib:${ENV}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
}

configure_cuda_devel

ATTEMPT=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname)
LOG_DIR=${RUN}/sco/attempts/${ATTEMPT}
mkdir -p "${LOG_DIR}" "${TORCH_EXTENSIONS_DIR}"
exec > >(tee -a "${LOG_DIR}/worker.log") 2>&1

cleanup() {
  status=$?
  printf '{"exit_status":%s,"ended_utc":"%s"}\n' \
    "${status}" "$(date -u +%FT%TZ)" > "${LOG_DIR}/exit.json"
}
trap cleanup EXIT

cd "${PACKAGE}"
sha256sum -c SHA256SUMS
WORK=$(mktemp -d /tmp/r1_sft_generation.XXXXXXXX)
tar -xzf source.tar.gz -C "${WORK}"
cd "${WORK}"
export PYTHONPATH=${WORK}/verl:${WORK}

{
  hostname
  id
  date -u +%FT%TZ
  "${PY}" --version
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
  printf 'CUDA_HOME=%s\nCURAND_INCLUDE_DIR=%s\nCURAND_LIBRARY_DIR=%s\n' \
    "${CUDA_HOME}" "${CURAND_INCLUDE_DIR}" "${CURAND_LIBRARY_DIR}"
  command -v nvcc
  nvcc --version
  sha256sum "${SOURCE}" "${AUDIT}" "${ENV_YAML}" "${PACKAGE}/source.tar.gz"
} > "${LOG_DIR}/environment.txt"

"${PY}" data_gen/active_spatial_sft/run_r1_sft_pipeline.py \
  --env-yaml "${ENV_YAML}" \
  --jsonl-path "${SOURCE}" \
  --audit-jsonl "${AUDIT}" \
  --output-dir "${OUTPUT}" \
  --render-backend local \
  --gs-root "${ASSETS}" \
  --gpu-device 0 \
  --qwen-format parquet \
  --also-no-think \
  --beam-width 16 \
  --max-dashboards -1 \
  --verbose

"${PY}" - "${OUTPUT}" "${SOURCE}" "${AUDIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

output, source, audit = map(Path, sys.argv[1:])
records = [json.loads(line) for line in (output / "sft_data.jsonl").open() if line.strip()]
assert len(records) == 210
assert all(row["success"] for row in records)
assert all(row["conversations"][-1]["role"] == "assistant" for row in records)
for row in records:
    assert len(row["primitive_image_paths"]) == row["total_actions"] + 1
    for relative in row["image_paths"] + row["primitive_image_paths"]:
        assert (output / relative).is_file(), relative

with_think = pd.read_parquet(output / "qwen_vl_sft.parquet")
no_think = pd.read_parquet(output / "qwen_vl_sft_no_think.parquet")
assert len(with_think) == len(no_think) == 210
assert len(list((output / "visualization/dashboards").glob("*.png"))) == 210
summary = json.loads((output / "visualization/score_guidance_summary.json").read_text())
assert summary["records"] == summary["successes"] == 210
certified = summary["certified_shortest_comparison"]
assert certified["records"] == certified["matches"] == 147

def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

gate = {
    "status": "PASS",
    "schema_version": "active_spatial_r1_sft_release_gate_v1",
    "records": len(records),
    "successful_trajectories": sum(bool(row["success"]) for row in records),
    "turn_images": sum(len(row["image_paths"]) for row in records),
    "primitive_images": sum(len(row["primitive_image_paths"]) for row in records),
    "dashboards": 210,
    "certified_shortest_matches": certified["matches"],
    "source_sha256": digest(source),
    "audit_sha256": digest(audit),
    "sft_jsonl_sha256": digest(output / "sft_data.jsonl"),
    "qwen_parquet_sha256": digest(output / "qwen_vl_sft.parquet"),
    "qwen_no_think_parquet_sha256": digest(output / "qwen_vl_sft_no_think.parquet"),
    "site_index_sha256": digest(output / "visualization/index.html"),
}
(output / "release_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
print(json.dumps(gate, indent=2, sort_keys=True))
PY

touch "${RUN}/COMPLETE"
