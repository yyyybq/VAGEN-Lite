#!/usr/bin/env bash
# vpd-tvr-vkre: NVIDIA Vulkan recheck, the sco JOB payload (EXPLORATORY; gate tvr; DECISIONS G2; 1 x H800,
# non-root; no downloads).  ROUND3 section 3 (TVR) asks for one cheap recheck: the SYSTEM NVIDIA driver libraries
# with the newer Vulkan loader (envs/tvr 1.4.357) and an ICD given by an ABSOLUTE path, once on zoetrope and once
# on another pool.  Never feeds the frozen TVR smoke verdict (vpd/tvr/SMOKE_CRITERIA.md): a PASS here only allows
# running that smoke on the pool (DECISIONS G2).  The probes are debug_probe.py's (debug/gl_tri.py, vk_tri.py,
# thor_diag.py, unchanged); the bookkeeping is vkre.py.
#
#   RUN=$VPD_OUT/jobs/tvr-vkre-<pool>-$(date -u +%Y%m%d%H%M)
#   $PY_EVAL -m vpd.sco prepare --run $RUN --name vpd-tvr-vkre --gate tvr --pool <pool> --est-gph 0.25 \
#       --env NVIDIA_DRIVER_CAPABILITIES:all \
#       -- env VPD_JOB_DEADLINE_S=660 timeout -k 30 720 \
#          bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/tvr/vk_recheck.sh $RUN/out
#   $PY_EVAL -m vpd.sco submit --run $RUN
#   (after it ends: $PY_EVAL -I $RUN/code/vpd/tvr/vkre.py expect --out $RUN/out, then vpd.sco delete --run $RUN)
#
# Variants, in this order; each writes OUT/<name>/result.json (pass null = skipped, with the reason); no failure
# stops the run; lingering Unity players are reaped after each.  Profiles (set in the variant's subshell):
#   R               the image's own stack (only if v00 finds the graphics stack mounted: caps_honored)
#   newloader       the 11-09 debug job's S overlay (NVGL/lib + NVGL/glvnd, staged absolute ICD / EGL JSON) with
#                   ENV/lib (loader 1.4.357) first
#   lvp             lavapipe ICD only, ENV/lib first, LP_NUM_THREADS=4
#   sysnew          the system NVIDIA libs + the env loader 1.4.357: VK_ICD_FILENAMES = VK_DRIVER_FILES =
#                   DIR/sys_icd.json (library_path SYS_LIB/libGLX_nvidia.so.0, versions copied from the system
#                   nvidia_icd.json), __EGL_VENDOR_LIBRARY_FILENAMES = DIR/sys_egl.json (library_path
#                   SYS_LIB/libEGL_nvidia.so.0), both written at run time; LD_LIBRARY_PATH = ENV/lib:NVGL/glvnd
#                   exactly (glvnd dispatch only, never NVGL/lib, so every NVIDIA library comes from the system)
#   sysnew_noglvnd  sysnew with LD_LIBRARY_PATH = ENV/lib
#   v00_facts               <=30   nvidia-smi -q / -L, node facts, NVIDIA_* env, /proc/driver/nvidia/version,
#                                  ldconfig (NVIDIA + glvnd / Vulkan libs), system graphics libs and ICD / layer
#                                  JSON; vkre.py facts -> kernel_driver, GPU UUID / PCI bus id / VBIOS,
#                                  system_equals_staged (NVGL/SHA256SUMS), caps_honored against the kernel driver
#   v01_gl_tri_R            <=45   control: EGL + GL triangle on R                       (debug/gl_tri.py)
#   v02_vk_nvS_env          <=45   control: the old best NVIDIA path (newloader; needs kernel driver = 595.71.05)
#   v02_vk_lvp              <=45   control: Vulkan triangle on lavapipe (probe health)   (debug/vk_tri.py)
#   v11_vk_sysnew           <=45   NVIDIA Vulkan triangle on sysnew (the user's variant)
#   v12_vk_sysnew_noglvnd   <=45   NVIDIA Vulkan triangle on sysnew_noglvnd (with v11: does the glvnd libEGL.so.1
#                                  the image lacks explain v02_vk_nvR's INCOMPATIBLE_DRIVER of 11-09?)
#   v13_thor_sysnew         <=240  AI2-THOR on sysnew: FloorPlan1 640x360 Ultra AA4, 40 steps + depth + seg,
#                                  gpu_device 0; Unity only: LD_DEBUG=libs,files, VK_LOADER_DEBUG=all
#                                  (debug/thor_diag.py); PASS = >= 90 % of the step frames with RGB std >= 2 AND
#                                  Player.log 'Vulkan renderer=[NVIDIA H800...]' (vkre.py thor-check)
#   summary                 OUT/summary.json (vkre.py summarize): per-variant status, verdict (PASS on this pool
#                           iff v13 PASS; FAIL; NOT_TESTED), flags (nv_vk_pass, glvnd_explains_R, nv_all_fail, ...),
#                           readings; OUT/expect.json
# VK_LOADER_DEBUG=all for every Vulkan triangle (the loader's log lands in DIR/log.txt; v13: in Player.log).
# NVIDIA_DRIVER_CAPABILITIES comes from `vpd.sco prepare --env` (this script only records it).
# Time: VPD_JOB_DEADLINE_S (default 660) is the soft deadline; every variant's inner cap is min(its cap, time left
# - RESERVE_S (60, for the summary)); with < MIN_VARIANT_S (30) left a variant is skipped as no_time.  Every cap
# together: 30+5*45+240 = 495 s (+ ~6 s of reaping / bookkeeping each), under 660 - 60.
# Test overrides: VPD_VKRE_OUT, VPD_VKRE_PY (probe interpreter), VPD_VKRE_TOOL_PY (bookkeeping interpreter),
#   VPD_VKRE_SYS_LIB, VPD_VKRE_VK_SYS_DIRS, VPD_VKRE_EGL_SYS_DIRS, VPD_VKRE_DEV_DIR, VPD_VKRE_DRIVER_FILE,
#   VPD_VKRE_BUILD_NAME, VPD_VKRE_REAP_WAIT, VPD_VKRE_RESERVE_S, VPD_VKRE_MIN_VARIANT_S (summary.json lists any set).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh"
cd "$VPD_WORKTREE" || exit 2
set +e   # common_env sets -euo pipefail; keep -u and pipefail, but run every variant

[[ $# -eq 1 ]] || { echo "usage: $0 OUTDIR" >&2; exit 2; }
OUTDIR=$1
if [[ -e $OUTDIR/steps.jsonl || -e $OUTDIR/summary.json ]]; then
  echo "refusing: $OUTDIR already holds a recheck run; give a new OUTDIR" >&2; exit 2
fi
mkdir -p "$OUTDIR/logs" || exit 2
OUTDIR=$(cd "$OUTDIR" && pwd)
export RUN=$OUTDIR   # vpd_step writes $RUN/steps.jsonl
vpd_chown_trap "$OUTDIR"   # no-op for the normal non-root run

T_START=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-660}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
DEADLINE=$((T_START + DEADLINE_S))
RESERVE_S=${VPD_VKRE_RESERVE_S:-60}
MIN_VARIANT_S=${VPD_VKRE_MIN_VARIANT_S:-30}

VOUT=${VPD_VKRE_OUT:-$VPD_OUT}
DRV=595.71.05   # the driver of the staged overlay; only the newloader control needs the kernel driver to match
T=$VOUT/tvr
NVGL=$T/nvgl_$DRV
ENV=$VOUT/envs/tvr
AI2THOR_HOME=$T/ai2thor_home
BUILD_NAME=${VPD_VKRE_BUILD_NAME:-thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917}
BUILD=$AI2THOR_HOME/.ai2thor/releases/$BUILD_NAME
# ai2thor starts the player as $HOME/.ai2thor/releases/<name>/<name> (HOME = the per-variant one), so lingering
# players are matched on this HOME-independent suffix (as debug_job.sh does).
UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"
S=$VPD_WORKTREE/vpd/tvr   # the job's code snapshot when run by vpd.sco
PROBE=$S/debug_probe.py
VKRE=$S/vkre.py
PY=${VPD_VKRE_PY:-$ENV/bin/python}
TOOL_PY=${VPD_VKRE_TOOL_PY:-}
if [[ -z $TOOL_PY ]]; then
  if [[ -x $PY ]]; then TOOL_PY=$PY; else TOOL_PY=$PY_EVAL; fi   # bookkeeping is stdlib only
fi
SYS_LIB=${VPD_VKRE_SYS_LIB:-/usr/lib/x86_64-linux-gnu}
VK_SYS_DIRS=${VPD_VKRE_VK_SYS_DIRS:-/etc/vulkan:/usr/share/vulkan}
EGL_SYS_DIRS=${VPD_VKRE_EGL_SYS_DIRS:-/usr/share/glvnd/egl_vendor.d:/etc/glvnd/egl_vendor.d}
DEV_DIR=${VPD_VKRE_DEV_DIR:-/dev}
DRIVER_FILE=${VPD_VKRE_DRIVER_FILE:-/proc/driver/nvidia/version}
WANT_RENDERER="NVIDIA H800"
LVP=$(ls "$ENV"/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -1)
ELDR=$ENV/lib/libvulkan.so.1
ORIG_LD=${LD_LIBRARY_PATH-}
ORIG_PATH=$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset ACTIVESPATIAL_USE_X11 ACTIVESPATIAL_X11_FORCE_VULKAN DISPLAY

say() { echo "[tvr-vkre $(date -u +%H:%M:%SZ)] $*"; }
jget() { "$TOOL_PY" -I "$PROBE" get "$1" "$2" 2>/dev/null; }
first() { local x; for x in "$@"; do [[ -n $x ]] && { echo "$x"; return 0; }; done; return 0; }

# Kill Unity players left behind by a variant, so the next one gets the card: TERM, wait up to VPD_VKRE_REAP_WAIT
# tenths of a second, then KILL.  Logged to logs/reap.log.
reap_unity() {
  local pids k wait=${VPD_VKRE_REAP_WAIT:-50}
  if ! command -v pgrep > /dev/null || ! command -v pkill > /dev/null; then
    echo "[tvr-vkre] $1: pgrep/pkill missing, cannot reap Unity players" | tee -a "$OUTDIR/logs/reap.log"
    return 0
  fi
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  [[ -n ${pids// /} ]] || return 0
  echo "[tvr-vkre] $(date -u +%TZ) $1: TERM lingering Unity player(s): $pids" | tee -a "$OUTDIR/logs/reap.log"
  pkill -TERM -f -- "$UNITY_PAT"
  for ((k = 0; k < wait; k++)); do
    pgrep -f -- "$UNITY_PAT" > /dev/null || return 0
    sleep 0.1
  done
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  echo "[tvr-vkre] $(date -u +%TZ) $1: KILL Unity player(s) still alive: $pids" | tee -a "$OUTDIR/logs/reap.log"
  pkill -KILL -f -- "$UNITY_PAT"
  sleep 0.5
  return 0
}

summary() {
  "$TOOL_PY" -I "$VKRE" summarize --out "$OUTDIR" > "$OUTDIR/logs/summary.log" 2>&1
  local rc=$?
  cat "$OUTDIR/logs/summary.log"
  return $rc
}
trap 'say "TERM: reaping and writing the summary"; reap_unity term; summary; exit 124' TERM

# --- environment profiles (run inside the variant's subshell; $1 = the variant's output dir) --------------------
# CUDA_VISIBLE_DEVICES stays unset: ai2thor keys gpu_device 0 by the in-container `nvidia-smi -L` index.
base_env() {  # per-variant HOME and caches (the NVIDIA driver writes ~/.nv / ~/.cache), our PATH, no overrides
  unset VK_ICD_FILENAMES VK_DRIVER_FILES VK_ADD_DRIVER_FILES VK_LAYER_PATH VK_ADD_LAYER_PATH VK_INSTANCE_LAYERS \
        VK_LOADER_LAYERS_ENABLE VK_LOADER_LAYERS_DISABLE VK_LOADER_DEBUG __EGL_VENDOR_LIBRARY_FILENAMES \
        __EGL_VENDOR_LIBRARY_DIRS XDG_DATA_HOME XDG_CONFIG_HOME __NV_PRIME_RENDER_OFFLOAD LP_NUM_THREADS \
        CUDA_VISIBLE_DEVICES LD_DEBUG
  if [[ -n $ORIG_LD ]]; then export LD_LIBRARY_PATH=$ORIG_LD; else unset LD_LIBRARY_PATH; fi
  export HOME="$1/home" XDG_CACHE_HOME="$1/home/.cache" PATH="$ENV/bin:$ORIG_PATH"
  mkdir -p "$HOME/.ai2thor"
}
prof_R() {  # the image's own stack (graphics capability)
  base_env "$1"
  export __EGL_VENDOR_LIBRARY_DIRS="$EGL_SYS_DIRS"
}
prof_newloader() {  # the 11-09 S overlay (staged 595.71.05 libs + glvnd, absolute staged JSON), env loader first
  base_env "$1"
  export LD_LIBRARY_PATH="$ENV/lib:$NVGL/lib:$NVGL/glvnd${ORIG_LD:+:$ORIG_LD}"
  export VK_ICD_FILENAMES="$NVGL/vulkan/nvidia_icd.json" VK_DRIVER_FILES="$NVGL/vulkan/nvidia_icd.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NVGL/egl/10_nvidia.json"
}
prof_lvp() {
  base_env "$1"
  export VK_ICD_FILENAMES="$LVP" VK_DRIVER_FILES="$LVP" LD_LIBRARY_PATH="$ENV/lib${ORIG_LD:+:$ORIG_LD}" LP_NUM_THREADS=4
}
prof_sysnew() {  # system NVIDIA libs + env loader 1.4.357 + absolute-path ICD / EGL vendor JSON written here
  base_env "$1"
  "$TOOL_PY" -I "$VKRE" icd-json --out "$1/sys_icd.json" --lib "$SYS_LIB/libGLX_nvidia.so.0" \
    ${SYS_ICD_SRC:+--src "$SYS_ICD_SRC"} || return 1
  "$TOOL_PY" -I "$VKRE" egl-json --out "$1/sys_egl.json" --lib "$SYS_LIB/libEGL_nvidia.so.0" || return 1
  export VK_ICD_FILENAMES="$1/sys_icd.json" VK_DRIVER_FILES="$1/sys_icd.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$1/sys_egl.json"
  export LD_LIBRARY_PATH="$ENV/lib:$NVGL/glvnd"
}
prof_sysnew_noglvnd() {
  prof_sysnew "$1" || return 1
  export LD_LIBRARY_PATH="$ENV/lib"
}
prof_failed() {  # DIR PROFILE (inside the subshell): record a harness error, end the subshell
  echo "profile $2 could not be set up" | tee "$1/profile_error.txt" >&2
  exit 3
}
envdump() {  # DIR: the environment the probe starts with
  env | grep -E '^(LD_LIBRARY_PATH|VK_[A-Z_]+|__EGL_VENDOR_LIBRARY_[A-Z]+|HOME|XDG_[A-Z_]+|LP_NUM_THREADS|CUDA_VISIBLE_DEVICES|__NV_PRIME_RENDER_OFFLOAD)=' \
    | sort > "$1/env_profile.txt"
}

# --- variant bodies: FN DIR CAP ARGS... (stdout / stderr go to DIR/log.txt) --------------------------------------
f_facts() {
  local d=$1 f x vkd   # (DIR CAP): every external call has its own timeout; the parse (with hashing) <= 20 s
  {
    echo "## date"; date -u +%FT%TZ
    echo "## host"; hostname
    echo "## id"; id
    echo "## nproc"; nproc
    echo "## cpu.max"; cat /sys/fs/cgroup/cpu.max
    echo "## nvidia-smi"
    timeout 10 nvidia-smi --query-gpu=name,driver_version,memory.total,pci.bus_id,uuid --format=csv
  } > "$d/node.txt" 2>&1
  timeout 20 nvidia-smi -q > "$d/nvidia_smi_q.txt" 2>&1
  timeout 10 nvidia-smi -L > "$d/nvidia_smi_L.txt" 2>&1
  cat "$DRIVER_FILE" > "$d/driver_version.txt" 2>&1
  env | grep -E '^(NVIDIA_|CUDA_|VK_|__EGL|__GL|__NV|LD_LIBRARY_PATH=|XDG_)' | sort > "$d/env.txt"
  timeout 10 ldconfig -p > "$d/ldconfig_p.txt" 2>&1
  grep -i nvidia "$d/ldconfig_p.txt" > "$d/ldconfig_nvidia.txt"
  grep -E 'lib(EGL|OpenGL|GLX|GL|GLdispatch|GLESv2|vulkan)\.so' "$d/ldconfig_p.txt" > "$d/ldconfig_gfx.txt"
  ls -la "$DEV_DIR"/dri "$DEV_DIR"/nvidia* > "$d/dev.txt" 2>&1
  {
    for f in "$SYS_LIB"/libGLX_nvidia.so* "$SYS_LIB"/libEGL_nvidia.so* "$SYS_LIB"/libnvidia-glvkspirv* \
             "$SYS_LIB"/libnvidia-glcore* "$SYS_LIB"/libnvidia-eglcore* "$SYS_LIB"/libvulkan.so* \
             "$SYS_LIB"/libEGL.so* "$SYS_LIB"/libOpenGL.so* "$SYS_LIB"/libGLdispatch.so*; do
      [[ -e $f || -L $f ]] && echo "$f -> $(readlink -f "$f") size=$(stat -L -c %s "$f" 2>/dev/null)"
    done
    IFS=: read -r -a vkd <<< "$VK_SYS_DIRS"
    for x in "${vkd[@]}"; do
      ls -la "$x"/icd.d "$x"/implicit_layer.d "$x"/explicit_layer.d 2>&1
      for f in "$x"/icd.d/*.json "$x"/implicit_layer.d/*.json; do
        [[ -f $f ]] && { echo "== $f ($(stat -c %s "$f") bytes)"; cat "$f"; }
      done
    done
  } > "$d/system_gfx.txt" 2>&1
  timeout 20 "$TOOL_PY" -I "$VKRE" facts --out "$d" --sys-lib "$SYS_LIB" --vk-dirs "$VK_SYS_DIRS" \
    --egl-vendor-dirs "$EGL_SYS_DIRS" --dev-dir "$DEV_DIR" --driver-file "$DRIVER_FILE" --staged-driver "$DRV" \
    --staged-sums "$NVGL/SHA256SUMS" --nvgl "$NVGL" --env-dir "$ENV" --build "$BUILD/$BUILD_NAME" --lvp-icd "$LVP"
}
f_gl() {  # DIR CAP PROFILE EGL_LIB GL_LIB
  ( "prof_$3" "$1" || prof_failed "$1" "$3"
    envdump "$1"
    exec timeout -k 10 "$2" "$PY" -I -u "$PROBE" gl-tri --out "$1" --egl-lib "$4" --gl-lib "$5" )
}
f_vk() {  # DIR CAP PROFILE LOADER WANT
  ( "prof_$3" "$1" || prof_failed "$1" "$3"
    export VK_LOADER_DEBUG=all
    envdump "$1"
    exec timeout -k 10 "$2" "$PY" -I -u "$PROBE" vk-tri --out "$1" --loader "$4" --want "$5" )
}
f_thor() {  # DIR CAP PROFILE [thor_diag args; later ones override the defaults]
  local d=$1 cap=$2 prof=$3 mon rc st dl
  shift 3
  st=$(( cap > 80 ? cap - 40 : cap / 2 ))
  dl=$(( cap > 50 ? cap - 25 : cap - cap / 4 ))
  mkdir -p "$d/home/.ai2thor" "$d/nvcache"
  ln -sfn "$AI2THOR_HOME/.ai2thor/releases" "$d/home/.ai2thor/releases"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.gr,memory.used --format=csv,noheader,nounits -lms 500 \
    > "$d/gpu.csv" 2> "$d/gpu.err" &
  mon=$!
  (
    "prof_$prof" "$d" || prof_failed "$d" "$prof"
    envdump "$d"
    exec timeout -k 10 "$cap" "$PY" -I -u "$PROBE" thor --out "$d" --variant "$(basename "$d")" \
      --startup-timeout "$st" --deadline-s "$dl" \
      --child-env "LD_DEBUG=libs,files" --child-env "LD_DEBUG_OUTPUT=$d/lddebug" \
      --child-env "VK_LOADER_DEBUG=all" --child-env "__GL_SHADER_DISK_CACHE=1" \
      --child-env "__GL_SHADER_DISK_CACHE_PATH=$d/nvcache" --child-env "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP=1" "$@"
  )
  rc=$?
  kill "$mon" 2> /dev/null; wait "$mon" 2> /dev/null
  reap_unity "$(basename "$d")"
  local plog="$d/home/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"
  [[ -f $plog ]] && cp -p "$plog" "$d/Player_after_exit.log"
  [[ -f $d/home/.ai2thor/log/unity.log ]] && cp -p "$d/home/.ai2thor/log/unity.log" "$d/unity.log"
  "$TOOL_PY" -I "$PROBE" post-thor --dir "$d"
  "$TOOL_PY" -I "$VKRE" thor-check --dir "$d" --want-renderer "$WANT_RENDERER" --nvgl "$NVGL"
  grep -h -m6 -e "Vulkan renderer" -e "Caught fatal" -e "requestedDeviceIndex" -e "deviceType" "$d"/Player*.log \
    2> /dev/null
  return $rc
}

# variant NAME CAP FN ARGS...: FN DIR CAP ARGS under the soft deadline, then reap, finalize, print a line.
variant() {
  local name=$1 cap=$2 fn=$3 d left t0 rc
  shift 3
  d=$OUTDIR/$name
  mkdir -p "$d"
  left=$(( DEADLINE - $(date +%s) - RESERVE_S ))
  if (( left < MIN_VARIANT_S )); then
    skip "$name" "no_time: ${left}s left before the summary reserve"
    return 0
  fi
  (( cap > left )) && cap=$left
  t0=$(date +%s)
  vpd_step "$name" "$fn" "$d" "$cap" "$@" > "$d/log.txt" 2>&1
  rc=$?
  reap_unity "$name"
  "$TOOL_PY" -I "$PROBE" finalize --dir "$d" --name "$name" --rc "$rc" --secs $(( $(date +%s) - t0 )) --cap "$cap"
  say "$name rc=$rc $(( $(date +%s) - t0 ))s cap=${cap}s pass=$(jget "$d/result.json" pass)"
  tail -4 "$d/log.txt"
  return 0
}
skip() {  # NAME REASON
  mkdir -p "$OUTDIR/$1"
  "$TOOL_PY" -I "$PROBE" finalize --dir "$OUTDIR/$1" --name "$1" --skipped "$2"
  vpd_step "$1:skipped" true > /dev/null
  say "$1 skipped: $2"
}
# maybe REASON NAME CAP FN ARGS...: skip with REASON when it is non-empty, else run the variant
maybe() {
  local why=$1
  shift
  if [[ -n $why ]]; then skip "$1" "$why"; else variant "$@"; fi
}

say "start OUTDIR=$OUTDIR uid=$(id -u) deadline=${DEADLINE_S}s NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES-unset}"
variant v00_facts 30 f_facts
CAPS=$(jget "$OUTDIR/v00_facts/result.json" metrics.caps_honored)
STAGED_OK=$(jget "$OUTDIR/v00_facts/result.json" metrics.staged_driver_match)
KDRV=$(jget "$OUTDIR/v00_facts/result.json" metrics.kernel_driver)
say "caps_honored=$CAPS kernel_driver=$KDRV staged_driver_match=$STAGED_OK"

SYS_ICD_SRC=""   # the system ICD manifest the sysnew JSON copies its version fields from
IFS=: read -r -a VKD <<< "$VK_SYS_DIRS"
for x in "${VKD[@]}"; do
  [[ -s $x/icd.d/nvidia_icd.json ]] && { SYS_ICD_SRC=$x/icd.d/nvidia_icd.json; break; }
done
PY_WHY=""; [[ -x $PY ]] || PY_WHY="no probe interpreter $PY"
S_WHY=""
for f in lib/libGLX_nvidia.so.0 lib/libEGL_nvidia.so.0 glvnd/libEGL.so.1 glvnd/libOpenGL.so.0 \
         vulkan/nvidia_icd.json egl/10_nvidia.json; do
  [[ -s $NVGL/$f ]] || { S_WHY="staged overlay incomplete: $NVGL/$f"; break; }
done
[[ -z $S_WHY && $STAGED_OK != true ]] && S_WHY="kernel driver $KDRV is not $DRV (the staged user-space libs would not match)"
R_WHY=""; [[ $CAPS == true ]] || R_WHY="caps_not_honored (NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES-unset})"
SYS_WHY=$R_WHY
[[ -z $SYS_WHY && ! -s $SYS_LIB/libEGL_nvidia.so.0 ]] && SYS_WHY="no system $SYS_LIB/libEGL_nvidia.so.0"
[[ -z $SYS_WHY && -z $SYS_ICD_SRC ]] && SYS_WHY="no system nvidia_icd.json under $VK_SYS_DIRS"
GLVND_WHY=""
for f in libEGL.so.1 libGLdispatch.so.0; do
  [[ -s $NVGL/glvnd/$f ]] || { GLVND_WHY="no staged glvnd $NVGL/glvnd/$f"; break; }
done
BUILD_WHY=""; [[ -x $BUILD/$BUILD_NAME ]] || BUILD_WHY="no AI2-THOR build at $BUILD"
LVP_WHY=""; [[ -n $LVP ]] || LVP_WHY="no lavapipe ICD under $ENV/share/vulkan/icd.d"
ENVLDR_WHY=""; [[ -e $ELDR ]] || ENVLDR_WHY="no env loader $ELDR"
# R: glvnd dispatch libs from the image if it has them, else the staged copies (they only dispatch)
R_EGL=$SYS_LIB/libEGL.so.1; [[ -s $R_EGL ]] || R_EGL=$NVGL/glvnd/libEGL.so.1
R_GL=$SYS_LIB/libOpenGL.so.0; [[ -s $R_GL ]] || R_GL=$NVGL/glvnd/libOpenGL.so.0
THOR_V3=(--gpu-device 0 --scene FloorPlan1 --width 640 --height 360 --quality Ultra --aa 4 --steps 40 --depth --seg)

maybe "$(first "$PY_WHY" "$R_WHY")" v01_gl_tri_R 45 f_gl R "$R_EGL" "$R_GL"
maybe "$(first "$PY_WHY" "$S_WHY" "$ENVLDR_WHY")" v02_vk_nvS_env 45 f_vk newloader "$ELDR" nvidia
maybe "$(first "$PY_WHY" "$LVP_WHY" "$ENVLDR_WHY")" v02_vk_lvp 45 f_vk lvp "$ELDR" cpu
maybe "$(first "$PY_WHY" "$SYS_WHY" "$GLVND_WHY" "$ENVLDR_WHY")" v11_vk_sysnew 45 f_vk sysnew "$ELDR" nvidia
maybe "$(first "$PY_WHY" "$SYS_WHY" "$ENVLDR_WHY")" v12_vk_sysnew_noglvnd 45 f_vk sysnew_noglvnd "$ELDR" nvidia
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$SYS_WHY" "$GLVND_WHY" "$ENVLDR_WHY")" v13_thor_sysnew 240 f_thor sysnew \
  "${THOR_V3[@]}"
vpd_step summary summary
[[ -f $OUTDIR/summary.json ]] || { say "summary.json was not written"; exit 5; }
say "verdict $(jget "$OUTDIR/summary.json" verdict) pool $(jget "$OUTDIR/summary.json" pool); end after $(( $(date +%s) - T_START )) s"
exit 0
