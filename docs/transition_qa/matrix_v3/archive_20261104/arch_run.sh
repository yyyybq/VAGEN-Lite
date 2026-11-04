# Archive the unpicked SFT epochs of B' and D_noP0 (the user's 甲, 2026-11-04) with the round-12 AOSS tool.
#   bash arch_run.sh upload   # phase 1: upload and size-check every file, delete nothing
#   bash arch_run.sh delete   # phase 2: the tool checks every size again and deletes a cell's files only if all match
# Keep list and targets from /tmp/yb_msgs/arch_plan.json (arch_plan.py); 4 upload threads, nice 19, no proxies.
set -euo pipefail
PHASE=$1
case "$PHASE" in upload) DEL="" ;; delete) DEL="--delete" ;; *) echo "phase upload|delete"; exit 2 ;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python
TOOL=/mnt/umm/users/yinbaiqiao/aoss_archive_manifests/archive_aoss_r12.py
S=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft
KEEP=$(python3 -c 'import json; d=json.load(open("/tmp/yb_msgs/arch_plan.json")); print(" ".join("--keep " + k for c in d.values() for k in c["keep"]))')
echo "phase $PHASE start $(date -u +%FT%TZ)"
nice -n 19 $PY $TOOL --workers 4 $DEL $KEEP $S/B/model $S/D_noP0/model
echo "phase $PHASE end $(date -u +%FT%TZ)"
