#!/usr/bin/env bash
# Delete one approved, archived and audited batch of ROOT-OWNED files with a one-card sco job (root inside the job).
# Only for batches the user approved with "delete through a root job" (first use: b3_20261026, 2026-10-26).
#   bash disk/submit_root_delete.sh BATCH [POOL]        # POOL default h800
# The job runs disk/archive_delete.py delete --yes on docs/disk/batches/BATCH.json: it refuses unless every planned
# file is on AOSS with the same size and unchanged, prints the list, deletes, writes the per-file record and markers
# and gives them back to the folder owner. The submission is recorded in disk_runs/batches/BATCH/sco_job.json.
set -euo pipefail
B=$1; POOL=${2:-h800}
U=/mnt/umm/users/yinbaiqiao
W=$U/VAGEN-Lite-disk
RUN=$U/disk_runs/batches/$B
SCO=$U/.sco/bin/sco
[ -f "$W/docs/disk/batches/$B.json" ] && [ -f "$RUN/manifest.jsonl" ] || { echo "no spec or manifest for $B"; exit 1; }
[ -e "$U/downstream_runs/PAUSE_SUBMISSIONS" ] && { echo "PAUSE_SUBMISSIONS exists: not submitting"; exit 1; }
LOG=$RUN/delete_rootjob.log
OWN=$(stat -c %u:%g "$U/disk_runs")
CMD="cd /tmp && ( env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY $U/.conda/envs/probe-spatial/bin/python -I $W/disk/archive_delete.py delete $W/docs/disk/batches/$B.json --yes ) > $LOG 2>&1; rc=\$?; chown $OWN $LOG; exit \$rc"
NAME="disk-root-delete-${B//_/-}"
echo "job $NAME on $POOL:"; echo "  $CMD"
OUT=$("$SCO" acp jobs create --workspace-name=aigc --aec2-name="$POOL" --job-name="$NAME" --priority=HIGHEST \
  --quota-type=reserved --container-image-url=registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617 \
  --storage-mount=019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm --training-framework=pytorch --worker-nodes=1 \
  --worker-spec=N4lS.Iq.I80.1 --command="$CMD" 2>&1) || true
echo "$OUT"
JOB=$(echo "$OUT" | grep -o 'pt-[a-z0-9]*' | head -1)
printf '{"batch": "%s", "job_name": "%s", "job_id": "%s", "pool": "%s", "submitted_utc": "%s", "log": "%s"}\n' \
  "$B" "$NAME" "$JOB" "$POOL" "$(date -u +%FT%TZ)" "$LOG" > "$RUN/sco_job.json"
cat "$RUN/sco_job.json"
