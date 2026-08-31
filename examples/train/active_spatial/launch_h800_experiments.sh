#!/bin/bash
# =============================================================================
# H800 Multi-Node Experiment Launcher (updated node map)
# =============================================================================
set -e

SSH_KEY="/mnt/umm/users/yinbaiqiao/id_rsa"
SSH_KNOWN_HOSTS="/tmp/known_hosts"
SSH_OPTS="-i ${SSH_KEY} -o UserKnownHostsFile=${SSH_KNOWN_HOSTS} -o StrictHostKeyChecking=no -o ConnectTimeout=15 -n"
VAGEN_ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"

CHECK_ONLY=false
if [ "${1:-}" = "--check" ]; then
    CHECK_ONLY=true
fi

# Node → (experiment, CUDA_VISIBLE_DEVICES)
# Prefer GPUs with >=60GB free and low util.
declare -A EXPERIMENT_MAP=(
    ["10.119.24.208"]="v46_7b_nodelta_w3.sh"       # all 8 nearly idle
    ["10.119.24.157"]="v47_7b_nodelta_w3_kl40.sh"   # all 8 low util
    ["10.119.28.68"]="v48_7b_nodelta_w1.sh"         # GPUs 1,2,4,5,7 free enough
)
declare -A GPU_MAP=(
    ["10.119.24.208"]="0,1,2,3,4"   # train 0-3, render 4
    ["10.119.24.157"]="0,1,2,3,6"   # train 0-3, render 6 (skip 5: most used)
    ["10.119.28.68"]="1,2,4,5,7"    # train 1,2,4,5 → logical 0-3; render 7 → logical 4
)

echo "=========================================================================="
echo "VAGEN-Lite H800 Multi-Node Experiment Launcher"
echo "$(date)"
echo "=========================================================================="
echo ""

echo "=== Pre-flight Checks ==="
PYTHON_BIN="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    echo "  [FAIL] conda env not found"; exit 1
fi
echo "  [OK]  conda env: ${PYTHON_BIN}"

if ! "${PYTHON_BIN}" -c "import torch; print('  [OK]  torch', torch.__version__)" 2>/dev/null; then
    echo "  [FAIL] torch not installed"; exit 1
fi

INTERIORGS_DIR="/mnt/umm/users/yinbaiqiao/InteriorGS"
SCENE_COUNT=$(ls "${INTERIORGS_DIR}/" 2>/dev/null | wc -l)
if [ "${SCENE_COUNT}" -lt 90 ]; then
    echo "  [FAIL] InteriorGS not ready: ${SCENE_COUNT} scenes"; exit 1
fi
echo "  [OK]  InteriorGS: ${SCENE_COUNT} scenes"

TRAIN_JSONL="${VAGEN_ROOT}/data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_6types.jsonl"
OOD_JSONL="${VAGEN_ROOT}/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"
for f in "${TRAIN_JSONL}" "${OOD_JSONL}"; do
    [ -f "$f" ] && echo "  [OK]  JSONL: $(basename $f)" || { echo "  [FAIL] Missing $f"; exit 1; }
done
[ -f "${SSH_KEY}" ] && echo "  [OK]  SSH key" || { echo "  [FAIL] SSH key"; exit 1; }

echo ""
if [ "${CHECK_ONLY}" = true ]; then
    echo "Dry-run complete. Planned launches:"
    for NODE in "${!EXPERIMENT_MAP[@]}"; do
        echo "  ${NODE}: ${EXPERIMENT_MAP[$NODE]}  GPUs=${GPU_MAP[$NODE]}"
    done
    exit 0
fi

echo "=== Launching Experiments ==="
echo ""

LAUNCHED=0
for NODE in "${!EXPERIMENT_MAP[@]}"; do
    EXP="${EXPERIMENT_MAP[$NODE]}"
    GPUS="${GPU_MAP[$NODE]}"
    EXP_NAME="${EXP%.sh}"
    EXP_CONFIG="${VAGEN_ROOT}/examples/train/active_spatial/experiments/${EXP}"
    LOG_FILE="${VAGEN_ROOT}/exps/vagen_active_spatial/${EXP_NAME}.log"
    SCRIPT="${VAGEN_ROOT}/examples/train/active_spatial/run_experiment.sh"

    echo "--- Node: ${NODE} ---"
    echo "  Experiment: ${EXP_NAME}"
    echo "  GPUs: ${GPUS}"
    echo "  Log: ${LOG_FILE}"

    mkdir -p "$(dirname "${LOG_FILE}")"

    ssh ${SSH_OPTS} root@${NODE} "
        mkdir -p '$(dirname ${LOG_FILE})'
        nohup bash -c '
            export CUDA_VISIBLE_DEVICES=${GPUS}
            export USE_GPU_HOLDER=false
            cd ${VAGEN_ROOT}
            bash ${SCRIPT} ${EXP_CONFIG}
        ' > '${LOG_FILE}' 2>&1 &
        echo \$!
    " 2>&1 | while read -r line; do
        echo "  Remote PID: ${line}"
    done

    echo "  Monitor: tail -f ${LOG_FILE}"
    echo ""
    LAUNCHED=$((LAUNCHED + 1))
done

echo "=========================================================================="
echo "Launched ${LAUNCHED} experiments."
echo "  v46 → 10.119.24.208  GPUs 0,1,2,3,4"
echo "  v47 → 10.119.24.157  GPUs 0,1,2,3,6"
echo "  v48 → 10.119.28.68   GPUs 1,2,4,5,7"
echo "=========================================================================="
