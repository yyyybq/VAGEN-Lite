#!/bin/bash
# Batch-run active_spatial_pipeline.run_pipeline over a list of scene IDs.
#
# Usage:
#   ./run_100scenes.sh [SCENE_LIST_FILE] [OUTPUT_DIR]
#
# Defaults:
#   SCENE_LIST_FILE = ./scenes_100.txt
#   OUTPUT_DIR      = ./output_100scenes
#
# This script processes scenes one-by-one (so a single bad scene does not
# kill the whole batch), writing per-scene results into:
#     OUTPUT_DIR/scenes/<scene_id>/data.jsonl
#     OUTPUT_DIR/train_data_<scene_id>.jsonl
#     OUTPUT_DIR/dataset_<scene_id>.json
#     OUTPUT_DIR/metadata_<scene_id>.json
#     OUTPUT_DIR/task_statistics_<scene_id>.png
# and a batch log:
#     OUTPUT_DIR/_batch_log.txt

set -u  # treat unset vars as errors; do NOT `set -e` so we can keep going on per-scene failures

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE_LIST_FILE="${1:-${SCRIPT_DIR}/scenes_100.txt}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/output_100scenes}"

SCENES_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
NUM_CAMERAS="${NUM_CAMERAS:-5}"
CONDA_ENV="${CONDA_ENV:-vagen}"
CONDA_SH="${CONDA_SH:-/scratch/by2593/miniconda3/bin/activate}"

if [ ! -f "${SCENE_LIST_FILE}" ]; then
    echo "[ERROR] Scene list file not found: ${SCENE_LIST_FILE}" >&2
    exit 1
fi

# Activate environment (idempotent)
# shellcheck disable=SC1090
source "${CONDA_SH}" "${CONDA_ENV}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/_batch_log.txt"
: > "${LOG_FILE}"

# Read scenes (skip comments / blank lines)
mapfile -t SCENES < <(grep -vE '^\s*(#|$)' "${SCENE_LIST_FILE}" | awk '{print $1}')

TOTAL=${#SCENES[@]}
if [ "${TOTAL}" -eq 0 ]; then
    echo "[ERROR] No scene IDs parsed from ${SCENE_LIST_FILE}" >&2
    exit 1
fi

echo "=============================================="
echo "Batch Pipeline — ${TOTAL} scenes"
echo "  Scenes root : ${SCENES_ROOT}"
echo "  Scene list  : ${SCENE_LIST_FILE}"
echo "  Output dir  : ${OUTPUT_DIR}"
echo "  Cameras/obj : ${NUM_CAMERAS}"
echo "=============================================="

OK=0
FAIL=0
FAILED_SCENES=()

for i in "${!SCENES[@]}"; do
    SCENE="${SCENES[$i]}"
    IDX=$((i + 1))
    SCENE_DIR="${SCENES_ROOT}/${SCENE}"

    echo ""
    echo "[${IDX}/${TOTAL}] Scene: ${SCENE}"

    if [ ! -d "${SCENE_DIR}" ]; then
        echo "  [SKIP] Scene directory does not exist: ${SCENE_DIR}"
        echo "${SCENE}	MISSING_DIR" >> "${LOG_FILE}"
        FAIL=$((FAIL + 1))
        FAILED_SCENES+=("${SCENE}")
        continue
    fi
    if [ ! -f "${SCENE_DIR}/labels.json" ]; then
        echo "  [SKIP] labels.json not found in ${SCENE_DIR}"
        echo "${SCENE}	MISSING_LABELS" >> "${LOG_FILE}"
        FAIL=$((FAIL + 1))
        FAILED_SCENES+=("${SCENE}")
        continue
    fi

    # Skip if already produced (idempotent re-run)
    if [ -f "${OUTPUT_DIR}/train_data_${SCENE}.jsonl" ]; then
        echo "  [SKIP] Already done (train_data_${SCENE}.jsonl exists)"
        echo "${SCENE}	ALREADY_DONE" >> "${LOG_FILE}"
        OK=$((OK + 1))
        continue
    fi

    python "${SCRIPT_DIR}/run_pipeline.py" \
        --scenes_root "${SCENES_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --scene_id "${SCENE}" \
        --num_cameras "${NUM_CAMERAS}" \
        --verbose
    RC=$?

    if [ ${RC} -eq 0 ] && [ -f "${OUTPUT_DIR}/train_data_${SCENE}.jsonl" ]; then
        N=$(wc -l < "${OUTPUT_DIR}/train_data_${SCENE}.jsonl")
        echo "  [OK] ${N} items"
        echo "${SCENE}	OK	${N}" >> "${LOG_FILE}"
        OK=$((OK + 1))
    else
        echo "  [FAIL] run_pipeline.py exit=${RC}"
        echo "${SCENE}	FAIL	rc=${RC}" >> "${LOG_FILE}"
        FAIL=$((FAIL + 1))
        FAILED_SCENES+=("${SCENE}")
    fi
done

echo ""
echo "=============================================="
echo "Done. OK=${OK}  FAIL=${FAIL}  TOTAL=${TOTAL}"
echo "  Per-scene log: ${LOG_FILE}"
if [ ${FAIL} -gt 0 ]; then
    echo "  Failed scenes:"
    for s in "${FAILED_SCENES[@]}"; do echo "    - ${s}"; done
fi
echo "=============================================="

# Merge all per-scene jsonl into one combined dataset.
COMBINED="${OUTPUT_DIR}/train_data_all.jsonl"
echo ""
echo "[Merge] Concatenating per-scene jsonl -> ${COMBINED}"
: > "${COMBINED}"
shopt -s nullglob
for f in "${OUTPUT_DIR}"/train_data_*.jsonl; do
    # avoid concatenating the merged file or any split outputs back into itself
    base="$(basename "${f}")"
    if [ "${base}" = "train_data_all.jsonl" ] || \
       [ "${base}" = "train.jsonl" ] || \
       [ "${base}" = "test.jsonl" ]; then
        continue
    fi
    cat "${f}" >> "${COMBINED}"
done
shopt -u nullglob

if [ -s "${COMBINED}" ]; then
    TOTAL_ITEMS=$(wc -l < "${COMBINED}")
    echo "[Merge] ${TOTAL_ITEMS} items in ${COMBINED}"

    echo ""
    echo "[Split] Running object-level 80/20 train/test split..."
    python "${SCRIPT_DIR}/split_train_test_by_object.py" \
        --input "${COMBINED}" \
        --train_out "${OUTPUT_DIR}/train.jsonl" \
        --test_out  "${OUTPUT_DIR}/test.jsonl" \
        --test_ratio 0.2 \
        --seed 42
else
    echo "[Merge] No data produced; skipping split."
fi
