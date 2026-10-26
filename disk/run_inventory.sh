#!/usr/bin/env bash
# One inventory pass over the user folder, read only (step 1 of the disk line):
#   1. disk/xtree.py: directory sizes from quarkfs's recursive xattrs (minutes);
#   2. disk/walk.py: one full walk (sizes, newest / oldest mtime, age buckets, big files, clean-up records);
#   3. GNU du over the same tree, timed (the user asked how long one du of the whole folder takes).
# Outputs go to /mnt/umm/users/yinbaiqiao/disk_runs/raw/<stamp>/, the log to disk_runs/logs/inventory_<stamp>.log.
# Start it detached:  (setsid nohup bash disk/run_inventory.sh > /dev/null 2>&1 < /dev/null &)
set -u
U=/mnt/umm/users/yinbaiqiao
PY=$U/.conda/envs/vagen-lite/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$U/disk_runs/raw/inventory_$STAMP
LOG=$U/disk_runs/logs/inventory_$STAMP.log
mkdir -p "$OUT"
exec >> "$LOG" 2>&1
echo "start $STAMP host $(hostname) code $(git -C "$HERE/.." rev-parse --short HEAD 2>/dev/null)"

if [ "${SKIP_XTREE:-0}" != 1 ]; then
  t=$(date +%s)
  nice -n 19 "$PY" -u "$HERE/xtree.py" "$U" --out "$OUT/xtree.tsv" --min-bytes 20e9 --min-file 1e9 --max-depth 9
  echo "xtree done rc=$? in $(( $(date +%s) - t )) s"
fi

t=$(date +%s)
nice -n 19 "$PY" -u "$HERE/walk.py" "$U" --out "$OUT/walk" --agg-depth 6
echo "walk done rc=$? in $(( $(date +%s) - t )) s"

t=$(date +%s)
nice -n 19 du -s -B1 --apparent-size "$U" > "$OUT/du_apparent.txt" 2> "$OUT/du_apparent.err"
echo "du --apparent-size done rc=$? in $(( $(date +%s) - t )) s: $(cat "$OUT/du_apparent.txt")"
echo "end $(date -u +%Y%m%dT%H%M%SZ)"
