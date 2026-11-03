#!/usr/bin/env bash
# Upload planned batches to AOSS in ONE one-card sco job (14 vCPU, 8 files x 8 threads), WITHOUT deleting anything:
# for batches whose deletion is not approved yet but whose upload is (the user, 2026-11-03: Qwen2.5 weights, upload all).
# Each batch needs its spec and its plan (disk/archive_delete.py plan, made on the login node as the owner). Deleting later
# goes through submit_archive_delete_job.sh once the spec is approved (archive then skips the objects already there).
#   bash disk/submit_archive_job.sh POOL BATCH [BATCH ...]
set -euo pipefail
POOL=$1; shift
U=/mnt/umm/users/yinbaiqiao
W=$U/VAGEN-Lite-disk
SCO=$U/.sco/bin/sco
PY=$U/.conda/envs/probe-spatial/bin/python
for B in "$@"; do
  [ -f "$W/docs/disk/batches/$B.json" ] && [ -f "$U/disk_runs/batches/$B/plan.tsv" ] || { echo "no spec or plan for $B"; exit 1; }
done
[ -e "$U/downstream_runs/PAUSE_SUBMISSIONS" ] && { echo "PAUSE_SUBMISSIONS exists: not submitting"; exit 1; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG=$U/disk_runs/logs/archive_job_$STAMP.log
OWN=$(stat -c %u:%g "$U/disk_runs")
BATCHES="$*"
STEPS=""
for B in "$@"; do
  S=$W/docs/disk/batches/$B.json; L=$U/disk_runs/logs/${B}_archive_job.log
  STEPS="$STEPS echo == $B archive \$(date -u +%H:%M:%S) >> $LOG; $PY -I $W/disk/archive_delete.py archive $S --files 8 --threads 8 > $L 2>&1; rc=\$?; tail -1 $L >> $LOG;"
  STEPS="$STEPS if [ \$rc -ne 0 ] || ! grep -q 'archive done: 0 not ok' $L; then echo == $B upload not clean >> $LOG; fail=1; fi;"
done
CMD="cd /tmp && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; fail=; echo start \$(date -u +%FT%TZ) archive only, batches $BATCHES > $LOG; $STEPS echo ALLDONE \$(date -u +%FT%TZ) fail=\$fail >> $LOG; chown -R $OWN $U/disk_runs/batches $LOG $U/disk_runs/logs/*_archive_job.log; [ -z \"\$fail\" ]"
NAME=$(echo "disk-archive-$STAMP" | tr 'A-Z' 'a-z')
echo "job $NAME on $POOL, batches $BATCHES (archive only), log $LOG"
OUT=$("$SCO" acp jobs create --workspace-name=aigc --aec2-name="$POOL" --job-name="$NAME" --priority=HIGHEST \
  --quota-type=reserved --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.1 --command="$CMD" 2>&1) || true
echo "$OUT"
JOB=$(echo "$OUT" | grep -o 'pt-[a-z0-9]*' | head -1)
printf '{"batches": "%s", "mode": "archive only", "job_name": "%s", "job_id": "%s", "pool": "%s", "submitted_utc": "%s", "log": "%s"}\n' \
  "$BATCHES" "$NAME" "$JOB" "$POOL" "$(date -u +%FT%TZ)" "$LOG" | tee "$U/disk_runs/logs/archive_job_$STAMP.json"
