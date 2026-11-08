#!/usr/bin/env bash
# TVR render smoke, staging on the LOGIN node (WP-TVR, plan section 3.7).  No GPU code, 1 core, nice -n 19.
# Ported from the WP-TVR design draft (scratchpad vpd_tvr/design/stage_tvr_login.sh).
#
#   nice -n 19 bash vpd/tvr/stage_tvr_login.sh [--local]       steps 0-3 and 8 (copies from this node only)
#   VPD_TVR_NET_OK=1 nice -n 19 bash vpd/tvr/stage_tvr_login.sh --all     + network steps 4-7, then 8
#   VPD_TVR_NET_OK=1 nice -n 19 bash vpd/tvr/stage_tvr_login.sh --net     network steps 4-7, then 8
#
# Steps 4-7 create a conda env from conda-forge, pip-install from PyPI, download the 836 MB AI2-THOR build and
# clone TVRBench: STOP-AND-ASK items (DECISIONS A3).  They refuse to run unless VPD_TVR_NET_OK=1 is set, which
# only the user (or the orchestrator after the user's OK) does.  Every step is idempotent.
#   0. the login node runs the job nodes' driver (595.71.05)
#   1. 12 NVIDIA 595.71.05 graphics user-space libs -> NVGL/lib (+ soname links, SHA256SUMS checked vs source)
#   2. glvnd 1.7.0 libEGL.so.1*, libOpenGL.so.0*, libGLdispatch.so.0* from ~/.conda/pkgs -> NVGL/glvnd
#   3. absolute-path ICD JSONs: NVGL/vulkan/nvidia_icd.json, NVGL/egl/10_nvidia.json
#   4. [net] conda env ENV (python 3.10, vulkan-tools 1.4.357.0, mesa-lavapipe 26.2.6) + pip ai2thor==5.0.0 ...;
#      asserts ai2thor._builds.COMMIT_ID
#   5. [net] CloudRendering build zip (pinned sha256) -> AI2THOR_HOME/.ai2thor/releases/thor-CloudRendering-<c>/
#   6. [net] git clone TVRBench -> REPO, checkout --detach the pinned sha
#   7. [net] ProcTHOR-10k train.jsonl.gz for REPO/data/procthor-10k: status recorded, never blocks
#   8. ldd of the staged libs (+ the Unity player if staged) with LD_LIBRARY_PATH set -> T/staging_report.txt
#
# Paths mirror vpd/paths.py (TVR_OUT, NVGL, AI2THOR_HOME, TVR_REPO, TVR_ENV); vpd/tests/test_tvr.py checks that.
# Test overrides: VPD_TVR_OUT, VPD_TVR_SYS, VPD_TVR_PKGS, VPD_TVR_DRIVER_FILE, VPD_TVR_CHECK_PY.
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CONDA_REPODATA_THREADS=1 CONDA_FETCH_THREADS=1 CONDA_VERIFY_THREADS=1 CONDA_EXECUTE_THREADS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1

MODE=local
case "${1:-}" in
  ""|--local) MODE=local ;;
  --net) MODE=net ;;
  --all) MODE=all ;;
  *) echo "usage: $0 [--local|--net|--all]" >&2; exit 2 ;;
esac
if [[ $MODE != local && ${VPD_TVR_NET_OK:-0} != 1 ]]; then
  echo "REFUSED: steps 4-7 download / install / clone from outside the cluster (DECISIONS A3, stop-and-ask)." >&2
  echo "         Set VPD_TVR_NET_OK=1 only after the user approved TVR staging." >&2
  exit 2
fi

OUT=${VPD_TVR_OUT:-/mnt/umm/users/yinbaiqiao/eval_external_assets/vpd}
DRV=595.71.05
T=$OUT/tvr
NVGL=$T/nvgl_$DRV
ENV=$OUT/envs/tvr
AI2THOR_HOME=$T/ai2thor_home
REPO=$T/repos/TVRBench
SYS=${VPD_TVR_SYS:-/usr/lib/x86_64-linux-gnu}
PKGS=${VPD_TVR_PKGS:-/mnt/umm/users/yinbaiqiao/.conda/pkgs}
DRIVER_FILE=${VPD_TVR_DRIVER_FILE:-/proc/driver/nvidia/version}
CHECK_PY=${VPD_TVR_CHECK_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin/python}
CONDA=/opt/conda/bin/conda
COMMIT=f0825767cd50d69f666c7f282e54abfe58f1e917   # ai2thor 5.0.0 ai2thor/_builds.py COMMIT_ID
BUILD_SHA=1fd5f998644a6dd522a4cf6604c05360a75a5c27f450a7486c20e52dc2b67bb9  # thor-CloudRendering-<COMMIT>.zip
BUILD_NAME=thor-CloudRendering-$COMMIT
BUILD_URL=http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/$BUILD_NAME.zip
TVR_URL=https://github.com/aim-uofa/TVRBench.git
TVR_SHA=ad296bb7ff880b8a0758187edc3ddc97eff92791  # aim-uofa/TVRBench main, checked 2026-11-08
LIBS=(libGLX_nvidia libEGL_nvidia libnvidia-glcore libnvidia-eglcore libnvidia-glsi libnvidia-tls
      libnvidia-glvkspirv libnvidia-gpucomp libnvidia-rtcore libnvidia-allocator libGLESv2_nvidia libGLESv1_CM_nvidia)
# soname -> versioned file (the other libs use the full versioned name as their soname)
SONAMES=(libGLX_nvidia.so.0 libEGL_nvidia.so.0 libGLESv2_nvidia.so.2 libGLESv1_CM_nvidia.so.1 libnvidia-allocator.so.1)
GLVND=("libegl-1.7.0-ha4b6fd6_5/lib/libEGL.so.1" "libopengl-1.7.0-ha4b6fd6_5/lib/libOpenGL.so.0"
       "libglvnd-1.7.0-ha4b6fd6_5/lib/libGLdispatch.so.0")

say() { echo "[stage $(date -u +%H:%M:%SZ)] $*"; }
die() { echo "[stage] FAIL: $*" >&2; exit 1; }
put() { # put SRC DST: copy unless identical (atomic rename)
  if [[ -f $2 && ! -L $2 ]] && cmp -s "$1" "$2"; then echo "  = $(basename "$2")"; return 0; fi
  cp -p "$1" "$2.part" && mv -f "$2.part" "$2" && echo "  + $(basename "$2")"
}
write_if_changed() { # write_if_changed DST (content on stdin)
  local tmp="$1.part"; cat > "$tmp"
  if [[ -f $1 ]] && cmp -s "$tmp" "$1"; then rm -f "$tmp"; echo "  = $(basename "$1")"; else mv -f "$tmp" "$1"; echo "  + $(basename "$1")"; fi
}

step0() {
  say "0. driver"
  grep -q " $DRV " "$DRIVER_FILE" || die "the driver in $DRIVER_FILE is not $DRV"
  echo "  driver $DRV ok"
}

step1() {
  say "1. NVIDIA $DRV graphics user-space libs -> $NVGL/lib"
  mkdir -p "$NVGL/lib"
  local f
  for f in "${LIBS[@]}"; do
    [[ -s $SYS/$f.so.$DRV ]] || die "missing or empty $SYS/$f.so.$DRV"
    put "$SYS/$f.so.$DRV" "$NVGL/lib/$f.so.$DRV"
  done
  local s
  for s in "${SONAMES[@]}"; do
    ln -sfn "${s%.so.*}.so.$DRV" "$NVGL/lib/$s"
  done
  (cd "$NVGL/lib" && for f in "${LIBS[@]}"; do sha256sum "./$f.so.$DRV"; done) | write_if_changed "$NVGL/SHA256SUMS"
  (cd "$SYS" && sha256sum --quiet -c "$NVGL/SHA256SUMS") || die "SHA256SUMS of the copies do not match $SYS"
  echo "  SHA256SUMS match the source ($SYS)"
  cat "$DRIVER_FILE" > "$NVGL/login_driver_version.txt"
}

step2() {
  say "2. glvnd 1.7.0 -> $NVGL/glvnd"
  mkdir -p "$NVGL/glvnd"
  local g base src sums=""
  for g in "${GLVND[@]}"; do
    base=$(basename "$g")
    for src in "$PKGS/$(dirname "$g")/$base"*; do
      [[ -e $src ]] || die "missing $PKGS/$g*"
      if [[ -L $src ]]; then
        ln -sfn "$(readlink "$src")" "$NVGL/glvnd/$(basename "$src")"
      else
        put "$src" "$NVGL/glvnd/$(basename "$src")"
        sums+="$(cd "$(dirname "$src")" && sha256sum "./$(basename "$src")")"$'\n'
      fi
    done
    [[ -e $NVGL/glvnd/$base ]] || die "$NVGL/glvnd/$base does not resolve"
  done
  printf '%s' "$sums" | write_if_changed "$NVGL/glvnd/SHA256SUMS"
  (cd "$NVGL/glvnd" && sha256sum --quiet -c SHA256SUMS) || die "glvnd copies do not match their sums"
}

step3() {
  say "3. ICD / vendor JSONs (absolute library paths)"
  mkdir -p "$NVGL/vulkan" "$NVGL/egl"
  write_if_changed "$NVGL/vulkan/nvidia_icd.json" <<EOF
{
    "file_format_version" : "1.0.1",
    "ICD": {
        "library_path": "$NVGL/lib/libGLX_nvidia.so.0",
        "api_version" : "1.4.329"
    }
}
EOF
  write_if_changed "$NVGL/egl/10_nvidia.json" <<EOF
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "$NVGL/lib/libEGL_nvidia.so.0"
    }
}
EOF
  "$CHECK_PY" -I -c 'import json,sys,os
for p in sys.argv[1:]:
    lib = json.load(open(p))["ICD"]["library_path"]
    assert os.path.isabs(lib) and os.path.exists(lib), (p, lib)
print("  json ok")' "$NVGL/vulkan/nvidia_icd.json" "$NVGL/egl/10_nvidia.json"
}

step4() {
  say "4. [net] conda env $ENV + pip ai2thor==5.0.0"
  local specs=(python=3.10 vulkan-tools=1.4.357.0 mesa-lavapipe=26.2.6)
  if [[ -x $ENV/bin/python && -x $ENV/bin/vulkaninfo ]]; then
    echo "  = env present"
  elif [[ -d $ENV ]]; then
    "$CONDA" install -y -p "$ENV" -c conda-forge --override-channels "${specs[@]}"
  else
    mkdir -p "$(dirname "$ENV")"
    "$CONDA" create -y -p "$ENV" -c conda-forge --override-channels "${specs[@]}"
  fi
  if "$ENV/bin/python" -I -c "import importlib.metadata as m; assert m.version('ai2thor') == '5.0.0'" 2>/dev/null; then
    echo "  = ai2thor 5.0.0 present"
  else
    "$ENV/bin/pip" install --no-cache-dir "ai2thor==5.0.0" numpy Pillow openai PyYAML scikit-image matplotlib
  fi
  "$ENV/bin/python" -I -c "import ai2thor._builds as b; assert b.COMMIT_ID == '$COMMIT', b.COMMIT_ID; print('  COMMIT_ID', b.COMMIT_ID)"
  [[ -x $ENV/bin/vulkaninfo ]] || die "no vulkaninfo in $ENV/bin"   # present only; never run it on the login node
}

step5() {
  say "5. [net] AI2-THOR CloudRendering build -> $AI2THOR_HOME/.ai2thor/releases/$BUILD_NAME"
  local rel=$AI2THOR_HOME/.ai2thor/releases tmpd=$AI2THOR_HOME/.ai2thor/tmp
  local zip=$tmpd/$BUILD_NAME.zip
  mkdir -p "$rel" "$tmpd"
  if [[ -x $rel/$BUILD_NAME/$BUILD_NAME ]]; then
    echo "  = build present"
  else
    if ! echo "$BUILD_SHA  $zip" | sha256sum -c --status 2>/dev/null; then
      curl -fL --retry 3 --retry-delay 10 -o "$zip.part" "$BUILD_URL"
      mv -f "$zip.part" "$zip"
    fi
    echo "$BUILD_SHA  $zip" | sha256sum -c || die "build zip sha256 mismatch"
    # same layout as ai2thor.build.Build.download: extract into tmp/<name>, rename to releases/<name>, chmod exe
    rm -rf "$tmpd/$BUILD_NAME.extract"
    "$ENV/bin/python" -I -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
      "$zip" "$tmpd/$BUILD_NAME.extract"
    mv "$tmpd/$BUILD_NAME.extract" "$rel/$BUILD_NAME"
    chmod 755 "$rel/$BUILD_NAME/$BUILD_NAME"
  fi
  echo "$BUILD_SHA  $zip" > "$T/build_zip.sha256"
  [[ -f $zip ]] && echo "$BUILD_SHA  $zip" | sha256sum -c --status && echo "  zip sha256 ok (kept at $zip)"
}

step6() {
  say "6. [net] TVRBench -> $REPO @ $TVR_SHA"
  mkdir -p "$(dirname "$REPO")"
  if [[ ! -d $REPO/.git ]]; then
    rm -rf "$REPO.part"
    git clone "$TVR_URL" "$REPO.part"
    mv "$REPO.part" "$REPO"
  fi
  git -C "$REPO" cat-file -e "$TVR_SHA^{commit}" 2>/dev/null || git -C "$REPO" fetch origin
  git -C "$REPO" -c advice.detachedHead=false checkout -q --detach "$TVR_SHA"
  [[ $(git -C "$REPO" rev-parse HEAD) == "$TVR_SHA" ]] || die "TVRBench HEAD is not $TVR_SHA"
  echo "  HEAD $(git -C "$REPO" rev-parse HEAD)"
}

step7() {
  say "7. [net] ProcTHOR-10k train.jsonl.gz (source UNVERIFIED; never blocks)"
  local p=$REPO/data/procthor-10k/train.jsonl.gz status why=""
  if [[ ! -f $p ]]; then
    status=missing; why="not in the TVRBench checkout; git-lfs: $(git lfs version 2>&1 | head -1 || true)"
  elif head -c 24 "$p" | grep -q "version https://git-lfs"; then
    if git lfs version >/dev/null 2>&1; then
      git -C "$REPO" lfs pull --include data/procthor-10k/train.jsonl.gz || true
    fi
    if head -c 24 "$p" | grep -q "version https://git-lfs"; then status=lfs_pointer; why="git-lfs not available or pull failed"; else status=ok; fi
  else
    status=ok
  fi
  if [[ $status == ok ]] && ! gzip -t "$p" 2>/dev/null; then status=bad_gzip; why="gzip -t failed"; fi
  local n=null sha=null bytes=null
  if [[ $status == ok ]]; then
    n=$(gzip -dc "$p" | wc -l); sha="\"$(sha256sum "$p" | cut -d' ' -f1)\""; bytes=$(stat -c %s "$p")
  fi
  why=$(printf '%s' "$why" | tr -d '"\\')
  printf '{"status": "%s", "path": "%s", "lines": %s, "bytes": %s, "sha256": %s, "why": "%s", "utc": "%s"}\n' \
    "$status" "$p" "$n" "$bytes" "$sha" "$why" "$(date -u +%FT%TZ)" > "$T/procthor_status.json"
  cat "$T/procthor_status.json"
  [[ $status == ok ]] || echo "  ProcTHOR not staged: the smoke runs the iTHOR task only and records it (plan 3.7)"
}

step8() {
  say "8. ldd report -> $T/staging_report.txt"
  local rep=$T/staging_report.txt f nf=0
  mkdir -p "$T"
  {
    echo "# TVR staging report  $(date -u +%FT%TZ)  host=$(hostname)  mode=$MODE"
    echo "# LD_LIBRARY_PATH=$NVGL/lib:$NVGL/glvnd"
    echo "## driver"; cat "$DRIVER_FILE" 2>&1 | head -1
    echo "## staged"
    for f in "$NVGL/SHA256SUMS" "$NVGL/glvnd/SHA256SUMS" "$NVGL/vulkan/nvidia_icd.json" "$NVGL/egl/10_nvidia.json" \
             "$ENV/bin/python" "$ENV/bin/vulkaninfo" "$AI2THOR_HOME/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME" \
             "$REPO/.git" "$REPO/data/tasks/eval.json" "$T/procthor_status.json"; do
      if [[ -e $f ]]; then echo "present $f"; else echo "ABSENT  $f"; fi
    done
    [[ -d $REPO/.git ]] && echo "tvrbench_head $(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
    [[ -f $T/procthor_status.json ]] && echo "procthor $(cat "$T/procthor_status.json")"
    echo "## sizes"; du -sh "$NVGL" "$ENV" "$AI2THOR_HOME" "$REPO" 2>/dev/null || true
    echo "## ldd output (an unresolved dependency reads: name, arrow, not found)"
    local targets=("$NVGL"/lib/*.so."$DRV" "$NVGL/glvnd/libEGL.so.1" "$NVGL/glvnd/libOpenGL.so.0" "$NVGL/glvnd/libGLdispatch.so.0")
    local b=$AI2THOR_HOME/.ai2thor/releases/$BUILD_NAME
    [[ -e $b/$BUILD_NAME ]] && targets+=("$b/$BUILD_NAME")
    [[ -e $b/UnityPlayer.so ]] && targets+=("$b/UnityPlayer.so")
    [[ -e $b/$BUILD_NAME ]] || echo "(Unity player not staged yet: steps 5 pending; its ldd is not in this report)"
    for f in "${targets[@]}"; do
      [[ -e $f ]] || continue
      echo "-- $f"
      LD_LIBRARY_PATH="$NVGL/lib:$NVGL/glvnd" ldd "$f" 2>&1 | sed 's/^/   /' || true
    done
  } > "$rep.part" 2>&1
  nf=$(grep -c "=> not found" "$rep.part" || true)
  echo "## NOT_FOUND_COUNT=$nf" >> "$rep.part"
  grep "=> not found" "$rep.part" | sort | uniq -c | sed 's/^/## NOT_FOUND /' >> "$rep.part" || true
  mv -f "$rep.part" "$rep"
  grep -e "^ABSENT" -e "^## NOT_FOUND" -e "^tvrbench_head" -e "^procthor" "$rep" || true
  if [[ $nf -gt 0 ]]; then
    echo "ACTION NEEDED: $nf 'not found' ldd lines; add those libs to $NVGL/lib before submitting the smoke"
  fi
}

say "mode=$MODE OUT=$OUT"
step0
if [[ $MODE == local || $MODE == all ]]; then step1; step2; step3; fi
if [[ $MODE == net || $MODE == all ]]; then step4; step5; step6; step7; fi
step8
say "done"
