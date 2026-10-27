#!/usr/bin/env bash
# The sha256 of every weight shard of a Hugging Face cache snapshot against its blob name (the cache names each LFS
# blob by its sha256), as a 1-card sco job: the user's decision 8 of 2026-10-24 (Qwen3-VL's shards are root-only, so
# only a job reads them).  Writes OUT_JSON; the job runs as root, so OUT_JSON's folder is handed back to its owner.
#
#   bash sha256_check.sh SNAPSHOT_DIR OUT_JSON
set -euo pipefail
SNAP=${1:?snapshot folder}
OUT=${2:?output json}
OWNER=$(stat -c %u:%g "$(dirname "${OUT}")")
trap 'chown -R "${OWNER}" "$(dirname "${OUT}")" 2>/dev/null || true' EXIT
cd "${SNAP}"
{
  echo "{\"snapshot\": \"${SNAP}\", \"utc\": \"$(date -u +%FT%TZ)\", \"host\": \"$(hostname)\", \"files\": ["
  first=1
  for f in *.safetensors; do
    blob=$(basename "$(readlink "${f}")")
    sum=$(sha256sum "${f}" | cut -d" " -f1)
    size=$(stat -L -c %s "${f}")
    [[ ${first} == 1 ]] || echo ","
    first=0
    printf '{"file": "%s", "bytes": %s, "blob": "%s", "sha256": "%s", "match": %s}' \
      "${f}" "${size}" "${blob}" "${sum}" "$([[ ${blob} == "${sum}" ]] && echo true || echo false)"
  done
  echo "], \"index_bytes\": $(stat -L -c %s model.safetensors.index.json)}"
} > "${OUT}"
cat "${OUT}"
