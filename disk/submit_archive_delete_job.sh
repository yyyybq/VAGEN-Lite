#!/usr/bin/env bash
# Archive to AOSS, then delete, approved and planned batches in ONE one-card sco job (14 vCPU): the login-node rule of
# 2026-10-28 allows at most 4 cores per task there (2 files x 2 threads, ~0.2 GB/s); in the job 8 files x 8 threads
# (~1.4 GB/s). For each batch in order: disk/archive_delete.py archive on its existing plan (plan.tsv, made on the
# login node as the owner), and only if that ends rc=0 with "archive done: 0 not ok", delete --yes (which itself
# refuses unless every planned file is on AOSS with the same size and unchanged). Stops at the first batch that is not
# clean. Records and markers are handed back to the owner of disk_runs at the end (the job runs as root).
#   bash disk/submit_archive_delete_job.sh POOL BATCH [BATCH ...]
set -euo pipefail
POOL=$1; shift
U=/mnt/umm/users/yinbaiqiao
W=$U/VAGEN-Lite-disk
SCO=$U/.sco/bin/sco
PY=$U/.conda/envs/probe-spatial/bin/python
for B in "$@"; do
  [ -f "$W/docs/disk/batches/$B.json" ] && [ -f "$U/disk_runs/batches/$B/plan.tsv" ] || { echo "no spec or plan for $B"; exit 1; }
  grep -q '"approved": "PENDING' "$W/docs/disk/batches/$B.json" && { echo "$B is not approved"; exit 1; }
done
[ -e "$U/downstream_runs/PAUSE_SUBMISSIONS" ] && { echo "PAUSE_SUBMISSIONS exists: not submitting"; exit 1; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$U/disk_runs/logs/archive_delete_job_$STAMP.log
OWN=$(stat -c %u:%g "$U/disk_runs")
BATCHES="$*"
STEPS=""
for B in "$@"; do
  S=$W/docs/disk/batches/$B.json; L=$U/disk_runs/logs/${B}_job.log
  STEPS="$STEPS echo == $B archive \$(date -u +%H:%M:%S) >> $LOG; $PY -I $W/disk/archive_delete.py archive $S --files 8 --threads 8 > $L 2>&1; rc=\$?; tail -1 $L >> $LOG;"
  STEPS="$STEPS if [ \$rc -ne 0 ] || ! grep -q 'archive done: 0 not ok' $L; then echo == $B upload not clean, stop >> $LOG; fail=1; fi;"
  STEPS="$STEPS if [ -z \"\$fail\" ]; then echo == $B delete \$(date -u +%H:%M:%S) >> $LOG; $PY -I $W/disk/archive_delete.py delete $S --yes >> $L 2>&1 || fail=1; grep -E '^deleted' $L >> $LOG; fi;"
  STEPS="$STEPS [ -n \"\$fail\" ] && { chown -R $OWN $U/disk_runs/batches $LOG $U/disk_runs/logs/*_job.log; exit 1; };"
done
CMD="cd /tmp && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; fail=; echo start \$(date -u +%FT%TZ) batches $BATCHES > $LOG; $STEPS echo ALLDONE \$(date -u +%FT%TZ) >> $LOG; chown -R $OWN $U/disk_runs/batches $LOG $U/disk_runs/logs/*_job.log"
NAME="disk-archive-delete-$STAMP"
NAME=$(echo "$NAME" | tr 'A-Z' 'a-z')
echo "job $NAME on $POOL, batches $BATCHES, log $LOG"
OUT=$("$SCO" acp jobs create --workspace-name=aigc --aec2-name="$POOL" --job-name="$NAME" --priority=HIGHEST \
  --quota-type=reserved --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.1 --command="$CMD" 2>&1) || true
echo "$OUT"
JOB=$(echo "$OUT" | grep -o 'pt-[a-z0-9]*' | head -1)
printf '{"batches": "%s", "job_name": "%s", "job_id": "%s", "pool": "%s", "submitted_utc": "%s", "log": "%s"}\n' \
  "$BATCHES" "$NAME" "$JOB" "$POOL" "$(date -u +%FT%TZ)" "$LOG" | tee "$U/disk_runs/logs/archive_delete_job_$STAMP.json"
