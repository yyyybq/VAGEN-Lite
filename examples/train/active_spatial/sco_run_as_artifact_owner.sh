#!/usr/bin/env bash
# Run an SCO payload as the shared-filesystem owner instead of container root.
# This prevents root-owned ledgers/evidence from becoming unreadable on the
# control host. Fail closed if the worker cannot switch to the requested uid.
set -euo pipefail

TARGET_UID=${R1_ARTIFACT_UID:-20325}
TARGET_GID=${R1_ARTIFACT_GID:-20325}
TARGET_USER=${R1_ARTIFACT_USER:-yinbaiqiao}
TARGET_HOME=${R1_ARTIFACT_HOME:-/mnt/umm/users/yinbaiqiao}

[[ $# -gt 0 ]] || { echo "usage: $0 COMMAND [ARG ...]" >&2; exit 2; }

if [[ $(id -u) -eq 0 ]]; then
  command -v setpriv >/dev/null 2>&1 || {
    echo "setpriv is required to avoid root-owned R1 artifacts" >&2
    exit 3
  }
  exec setpriv \
    --reuid="${TARGET_UID}" \
    --regid="${TARGET_GID}" \
    --clear-groups \
    env HOME="${TARGET_HOME}" USER="${TARGET_USER}" LOGNAME="${TARGET_USER}" \
    bash -c 'umask 022; exec "$@"' bash "$@"
fi

[[ $(id -u) -eq ${TARGET_UID} && $(id -g) -eq ${TARGET_GID} ]] || {
  echo "refusing to create R1 artifacts as uid=$(id -u) gid=$(id -g); expected ${TARGET_UID}:${TARGET_GID}" >&2
  exit 4
}
export HOME="${TARGET_HOME}" USER="${TARGET_USER}" LOGNAME="${TARGET_USER}"
umask 022
exec "$@"
