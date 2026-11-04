# E2 and Cr's unpicked SFT epochs (the user's 甲, 2026-11-04): upload to AOSS, check every file in the manifest,
# recheck the live jobs and the cells, then the tool's second pass (every size again) deletes; record and quota last.
# Each gate stops the run before anything is deleted.
set -uo pipefail
R=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/archive_20261104b
SCO=/mnt/umm/users/yinbaiqiao/.sco/bin/sco
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
start=$(date -u +%Y%m%dT%H%M)
bash /tmp/yb_msgs/arch_run2.sh upload >> $R/upload.log 2>&1 || { echo "upload failed: nothing deleted"; exit 1; }
tail -n 4 $R/upload.log
nice -n 19 $PY -I /tmp/yb_msgs/arch_check2.py $start /tmp/yb_msgs/arch_plan2.json > $R/check.txt 2>&1
cat $R/check.txt
grep -q '^ALL_OK$' $R/check.txt || { echo "manifest check failed: nothing deleted"; exit 1; }
$SCO acp jobs list --workspace-name=aigc --page-size 500 -o json > $R/jobs_before_delete.json
nice -n 19 $PY -I /tmp/yb_msgs/arch_plan2.py $R/jobs_before_delete.json > $R/recheck.txt 2>&1
cat $R/recheck.txt
grep -q 'problems 0 \[\]' $R/recheck.txt || { echo "recheck failed: nothing deleted"; exit 1; }
echo "gates passed $(date -u +%FT%TZ)"
bash /tmp/yb_msgs/arch_run2.sh delete >> $R/delete.log 2>&1 || { echo "delete phase failed"; exit 1; }
tail -n 8 $R/delete.log
M1=$(grep -o "manifest_[0-9TZ]*\.jsonl" $R/upload.log | tail -1)
M2=$(grep -o "manifest_[0-9TZ]*\.jsonl" $R/delete.log | tail -1)
nice -n 19 $PY -I /tmp/yb_msgs/arch_record2.py $M1 $M2
$SCO afs dir-quota list -i 019ec9f9-6d12-7d49-aad4-864b15c9eb06 2>&1 | grep yinbaiqiao | cut -c1-200
