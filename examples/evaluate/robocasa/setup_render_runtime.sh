#!/usr/bin/env bash
# Ubuntu 20.04 runtime overlay. Downloads/extracts libraries; no system install.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEST="${DEST:-$ROOT/playground/Runtime/egl}"
mkdir -p "$DEST"
cd "$DEST"
apt-get download libegl1 libosmesa6 libglapi-mesa
for package in ./*.deb; do dpkg-deb -x "$package" "$DEST"; done
LIB="$DEST/usr/lib/x86_64-linux-gnu"
ln -sfn libEGL.so.1 "$LIB/libEGL.so"
ln -sfn libOSMesa.so.6 "$LIB/libOSMesa.so"
LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$LIB/libOSMesa.so.6"
echo "Runtime overlay: $LIB (requires host libLLVM12 and libGLdispatch)"
