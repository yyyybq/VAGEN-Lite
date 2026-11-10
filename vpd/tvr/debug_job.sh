#!/usr/bin/env bash
# vpd-tvr-debug: TVR render debug job, the sco JOB payload (EXPLORATORY; gate tvr; 1 x H800, non-root; no
# downloads).  Separates three explanations of smoke_run1's all-zero AI2-THOR CloudRendering frames: (1) our
# compute-only container + hand-staged 595.71.05 libs, (2) Unity 2020.3 / ai2thor 5.0.0 x driver 595 x Hopper,
# (3) the server (H800 3D pipeline unusable from job containers); and measures lavapipe for the first time.
# Never feeds the frozen TVR smoke verdict (vpd/tvr/SMOKE_CRITERIA.md).
#
#   RUN=$VPD_OUT/jobs/tvr-debug-$(date -u +%Y%m%d%H%M)
#   $PY_EVAL -m vpd.sco prepare --run $RUN --name vpd-tvr-debug --gate tvr --est-gph 1.0 \
#       -- env VPD_JOB_DEADLINE_S=3120 timeout -k 60 3300 \
#          bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/tvr/debug_job.sh $RUN/out
#   $PY_EVAL -m vpd.sco submit --run $RUN          # STOP-AND-ASK: the orchestrator, after the user's OK
#   (after it ends: check $RUN/out/summary.json, then $PY_EVAL -m vpd.sco delete --run $RUN)
#
# Variants, in this order; each writes OUT/<name>/result.json {pass, stage, metrics, notes, rc, secs, cap_s}
# (pass null = skipped, with the reason); no failure stops the run; lingering Unity players are reaped after each.
# Profiles: S = staged overlay (NVGL/lib + glvnd, absolute ICD / EGL vendor JSON; the smoke's environment),
# R = the image's own libraries / ICD / vendor JSON (only if V0 finds them mounted: caps_honored).
#   v00_facts            <=30   nvidia-smi -q, NVIDIA_* env, system libGLX_nvidia / glvkspirv / ICD / layer
#                               JSON, /dev/dri, ldconfig -> caps_honored
#   v01_gl_tri_S|_R      <=60   EGL pbuffer + GLSL 330 red triangle, depth test on      (debug/gl_tri.py)
#   v02_vk_nvS_img       <=60   Vulkan triangle, NVIDIA, S, image loader 1.2.131        (debug/vk_tri.py)
#   v02_vk_nvS_env       <=60   ... S, env loader 1.4.357 (envs/tvr/lib first)
#   v02_vk_nvR           <=60   ... R (only if caps_honored)
#   v02_vk_lvp           <=60   ... lavapipe ICD, env loader
#   v03_thor_diag_S      <=240  the failing smoke config: FloorPlan1 640x360 Ultra AA4, gpu_device 0, 40 steps +
#                               depth + instance segmentation; Unity only: LD_DEBUG=libs,files, VK_LOADER_DEBUG=all,
#                               __GL_SHADER_DISK_CACHE_PATH; nvidia-smi sampled every 500 ms (debug/thor_diag.py)
#   v04_thor_R           <=240  V3 on R (only if caps_honored)
#   v05_thor_lowq        <=240  V3 with 'Very Low', no antiAliasing, 300x300
#   v06_thor_newloader   <=240  V3 with envs/tvr/lib first (loader 1.4.357)
#   v07_thor_layers      <=240  V3 + VK_LAYER_NV_optimus implicit layer (XDG_DATA_HOME) + __NV_PRIME_RENDER_OFFLOAD=1
#   v08_thor_direct      <=240  V3 + Unity -force-gfx-direct
#   v09_open3d_egl       <=120  probe-spatial open3d 0.19 OffscreenRenderer lit box on EGL (positive control)
#   v10_thor_lvp_forced  <=900  lavapipe forced: cuda-vulkan-mapping {"0": 0}, CUDA_VISIBLE_DEVICES=0,
#                               gpu_device 0 (-> -force-device-index 0), 'Very Low' 300x300, 20 steps + depth,
#                               LP_NUM_THREADS=12, fails fast on 'Caught fatal signal'
#   summary              OUT/summary.json (debug/summarize.py: per-variant table + decision-table rows) and
#                        OUT/view/<variant>__{rgb,depth,seg,tri}.png
# Time: VPD_JOB_DEADLINE_S (default 3120) is the soft deadline; every variant's inner cap is
# min(its cap, time left - RESERVE_S (60, for the summary)); with < MIN_VARIANT_S (30) left a variant is skipped
# as no_time.  Every cap together: 30+2*60+4*60+6*240+120+900 = 2850 s (+ reaps), under 3120.
# Test overrides: VPD_COMMON_ENV, VPD_TVR_OUT, VPD_TVR_PY (probe interpreter), VPD_TVR_TOOL_PY (bookkeeping
#   interpreter), VPD_TVR_O3D_PY, VPD_TVR_SYS_LIB, VPD_TVR_VK_SYS_DIRS, VPD_TVR_EGL_SYS_DIRS, VPD_TVR_DEV_DIR,
#   VPD_TVR_DRIVER_FILE, VPD_TVR_REAP_WAIT, VPD_TVR_RESERVE_S, VPD_TVR_MIN_VARIANT_S.
source "${VPD_COMMON_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../jobs" && pwd)/common_env.sh}"
set +e   # common_env sets -euo pipefail; keep -u and pipefail, but run every variant

[[ $# -eq 1 ]] || { echo "usage: $0 OUTDIR" >&2; exit 2; }
OUTDIR=$1
if [[ -e $OUTDIR/steps.jsonl || -e $OUTDIR/summary.json ]]; then
  echo "refusing: $OUTDIR already holds a debug run; give a new OUTDIR" >&2; exit 2
fi
mkdir -p "$OUTDIR/logs" || exit 2
OUTDIR=$(cd "$OUTDIR" && pwd)
export RUN=$OUTDIR   # vpd_step writes $RUN/steps.jsonl
vpd_chown_trap "$OUTDIR"   # no-op for the normal non-root run

T_START=$(date +%s)
DEADLINE_S=${VPD_JOB_DEADLINE_S:-3120}
[[ $DEADLINE_S =~ ^[1-9][0-9]*$ ]] || { echo "VPD_JOB_DEADLINE_S must be a positive integer" >&2; exit 2; }
DEADLINE=$((T_START + DEADLINE_S))
RESERVE_S=${VPD_TVR_RESERVE_S:-60}
MIN_VARIANT_S=${VPD_TVR_MIN_VARIANT_S:-30}

VOUT=${VPD_TVR_OUT:-$VPD_OUT}
DRV=595.71.05
T=$VOUT/tvr
NVGL=$T/nvgl_$DRV
ENV=$VOUT/envs/tvr
AI2THOR_HOME=$T/ai2thor_home
BUILD_NAME=thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
BUILD=$AI2THOR_HOME/.ai2thor/releases/$BUILD_NAME
# ai2thor starts the player as $HOME/.ai2thor/releases/<name>/<name> (HOME = the per-variant one), so lingering
# players are matched on this HOME-independent suffix (as tvr_smoke.sh does).
UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE=$S/debug_probe.py
PY=${VPD_TVR_PY:-$ENV/bin/python}
TOOL_PY=${VPD_TVR_TOOL_PY:-}
if [[ -z $TOOL_PY ]]; then
  if [[ -x $PY ]]; then TOOL_PY=$PY; else TOOL_PY=$PY_EVAL; fi   # bookkeeping is stdlib only
fi
O3D_PY=${VPD_TVR_O3D_PY:-/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python}
SYS_LIB=${VPD_TVR_SYS_LIB:-/usr/lib/x86_64-linux-gnu}
VK_SYS_DIRS=${VPD_TVR_VK_SYS_DIRS:-/etc/vulkan:/usr/share/vulkan}
EGL_SYS_DIRS=${VPD_TVR_EGL_SYS_DIRS:-/usr/share/glvnd/egl_vendor.d:/etc/glvnd/egl_vendor.d}
DEV_DIR=${VPD_TVR_DEV_DIR:-/dev}
DRIVER_FILE=${VPD_TVR_DRIVER_FILE:-/proc/driver/nvidia/version}
LVP=$(ls "$ENV"/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -1)
ORIG_LD=${LD_LIBRARY_PATH-}
ORIG_PATH=$PATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset ACTIVESPATIAL_USE_X11 ACTIVESPATIAL_X11_FORCE_VULKAN DISPLAY

say() { echo "[tvr-debug $(date -u +%H:%M:%SZ)] $*"; }
jget() { "$TOOL_PY" -I "$PROBE" get "$1" "$2" 2>/dev/null; }
first() { local x; for x in "$@"; do [[ -n $x ]] && { echo "$x"; return 0; }; done; return 0; }

# Kill Unity players left behind by a variant, so the next one gets the card: TERM, wait up to VPD_TVR_REAP_WAIT
# tenths of a second, then KILL (reap_unity of tvr_smoke.sh).  Logged to logs/reap.log.
reap_unity() {
  local pids k wait=${VPD_TVR_REAP_WAIT:-50}
  if ! command -v pgrep > /dev/null || ! command -v pkill > /dev/null; then
    echo "[tvr-debug] $1: pgrep/pkill missing, cannot reap Unity players" | tee -a "$OUTDIR/logs/reap.log"
    return 0
  fi
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  [[ -n ${pids// /} ]] || return 0
  echo "[tvr-debug] $(date -u +%TZ) $1: TERM lingering Unity player(s): $pids" | tee -a "$OUTDIR/logs/reap.log"
  pkill -TERM -f -- "$UNITY_PAT"
  for ((k = 0; k < wait; k++)); do
    pgrep -f -- "$UNITY_PAT" > /dev/null || return 0
    sleep 0.1
  done
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  echo "[tvr-debug] $(date -u +%TZ) $1: KILL Unity player(s) still alive: $pids" | tee -a "$OUTDIR/logs/reap.log"
  pkill -KILL -f -- "$UNITY_PAT"
  sleep 0.5
  return 0
}

summary() {
  "$TOOL_PY" -I "$PROBE" summarize --out "$OUTDIR" > "$OUTDIR/logs/summary.log" 2>&1
  local rc=$?
  cat "$OUTDIR/logs/summary.log"
  return $rc
}
trap 'say "TERM: reaping and writing the summary"; reap_unity term; summary; exit 124' TERM

# --- environment profiles (run inside the variant's subshell; DIR = the variant's output dir) -------------------
# CUDA_VISIBLE_DEVICES is unset (as in smoke_run1): ai2thor then keys gpu_device 0 by the in-container
# `nvidia-smi -L` index; only V10 sets it (to 0) next to its pre-written cuda-vulkan mapping.
base_env() {  # per-variant HOME and caches (the NVIDIA driver writes ~/.nv / ~/.cache), our PATH, no overrides
  unset VK_ICD_FILENAMES VK_DRIVER_FILES VK_ADD_DRIVER_FILES VK_LAYER_PATH VK_ADD_LAYER_PATH VK_INSTANCE_LAYERS \
        VK_LOADER_LAYERS_ENABLE VK_LOADER_LAYERS_DISABLE __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS \
        XDG_DATA_HOME XDG_CONFIG_HOME __NV_PRIME_RENDER_OFFLOAD LP_NUM_THREADS CUDA_VISIBLE_DEVICES
  if [[ -n $ORIG_LD ]]; then export LD_LIBRARY_PATH=$ORIG_LD; else unset LD_LIBRARY_PATH; fi
  export HOME="$1/home" XDG_CACHE_HOME="$1/home/.cache" PATH="$ENV/bin:$ORIG_PATH"
  mkdir -p "$HOME/.ai2thor"
}
prof_S() {  # the smoke's overlay: our 595.71.05 libs first, glvnd 1.7.0, absolute-path ICD and EGL vendor JSON
  base_env "$1"
  export LD_LIBRARY_PATH="$NVGL/lib:$NVGL/glvnd${ORIG_LD:+:$ORIG_LD}"
  export VK_ICD_FILENAMES="$NVGL/vulkan/nvidia_icd.json" VK_DRIVER_FILES="$NVGL/vulkan/nvidia_icd.json"
  export __EGL_VENDOR_LIBRARY_FILENAMES="$NVGL/egl/10_nvidia.json"
}
prof_R() {  # the image's own stack (graphics capability); glvnd dispatch from the image if present, else staged
  base_env "$1"
  export __EGL_VENDOR_LIBRARY_DIRS="$EGL_SYS_DIRS"
}
prof_newloader() { prof_S "$1"; export LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH"; }
prof_layers() {
  prof_S "$1"
  export XDG_DATA_HOME="$1/xdg_data" __NV_PRIME_RENDER_OFFLOAD=1
  "$TOOL_PY" -I "$PROBE" layers-json --out "$XDG_DATA_HOME/vulkan/implicit_layer.d/nvidia_layers.json" \
    --lib "$NVGL/lib/libGLX_nvidia.so.0"
}
prof_lvp() {
  base_env "$1"
  export VK_ICD_FILENAMES="$LVP" VK_DRIVER_FILES="$LVP" LD_LIBRARY_PATH="$ENV/lib${ORIG_LD:+:$ORIG_LD}" LP_NUM_THREADS=4
}
prof_lvpforced() {  # B2 done right: a device index for Unity, which rejects the CPU device without one
  prof_lvp "$1"
  export CUDA_VISIBLE_DEVICES=0 LP_NUM_THREADS=12
  printf '{"0": 0}\n' > "$HOME/.ai2thor/cuda-vulkan-mapping.json"
}

# --- variant bodies: FN DIR CAP ARGS... (stdout / stderr go to DIR/log.txt) --------------------------------------
f_facts() {
  local d=$1 f   # (DIR CAP): nvidia-smi -q <= 20 s, -L <= 10 s, the parse <= 15 s
  timeout 20 nvidia-smi -q > "$d/nvidia_smi_q.txt" 2>&1
  timeout 10 nvidia-smi -L > "$d/nvidia_smi_L.txt" 2>&1
  env | grep -E '^(NVIDIA_|CUDA_|VK_|__EGL|__GL|__NV|LD_LIBRARY_PATH=|XDG_)' | sort > "$d/env.txt"
  ldconfig -p 2>&1 | grep -i nvidia > "$d/ldconfig_nvidia.txt"
  ls -la "$DEV_DIR"/dri "$DEV_DIR"/nvidia* > "$d/dev.txt" 2>&1
  {
    for f in "$SYS_LIB"/libGLX_nvidia.so* "$SYS_LIB"/libEGL_nvidia.so* "$SYS_LIB"/libnvidia-glvkspirv* \
             "$SYS_LIB"/libnvidia-glcore* "$SYS_LIB"/libnvidia-eglcore* "$SYS_LIB"/libvulkan.so* \
             "$SYS_LIB"/libEGL.so* "$SYS_LIB"/libOpenGL.so*; do
      [[ -e $f || -L $f ]] && echo "$f -> $(readlink -f "$f") size=$(stat -L -c %s "$f" 2>/dev/null)"
    done
    local x vkd
    IFS=: read -r -a vkd <<< "$VK_SYS_DIRS"
    for x in "${vkd[@]}"; do
      ls -la "$x"/icd.d "$x"/implicit_layer.d "$x"/explicit_layer.d 2>&1
      for f in "$x"/icd.d/*.json "$x"/implicit_layer.d/*.json; do
        [[ -f $f ]] && { echo "== $f ($(stat -c %s "$f") bytes)"; cat "$f"; }
      done
    done
  } > "$d/system_gfx.txt" 2>&1
  timeout 15 "$TOOL_PY" -I "$PROBE" facts --out "$d" --sys-lib "$SYS_LIB" --vk-dirs "$VK_SYS_DIRS" \
    --egl-vendor-dirs "$EGL_SYS_DIRS" --dev-dir "$DEV_DIR" --driver "$DRV" --driver-file "$DRIVER_FILE" \
    --nvgl "$NVGL" --env-dir "$ENV" --build "$BUILD/$BUILD_NAME" --lvp-icd "$LVP" --o3d-py "$O3D_PY"
}
f_gl() {  # DIR CAP PROFILE EGL_LIB GL_LIB
  ( "prof_$3" "$1"; exec timeout -k 10 "$2" "$PY" -I -u "$PROBE" gl-tri --out "$1" --egl-lib "$4" --gl-lib "$5" )
}
f_vk() {  # DIR CAP PROFILE LOADER WANT
  ( "prof_$3" "$1"; exec timeout -k 10 "$2" "$PY" -I -u "$PROBE" vk-tri --out "$1" --loader "$4" --want "$5" )
}
f_o3d() {  # DIR CAP
  ( prof_S "$1"; exec timeout -k 10 "$2" "$O3D_PY" -I -u "$PROBE" open3d --out "$1" )
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
    "prof_$prof" "$d"
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
DRV_OK=$(jget "$OUTDIR/v00_facts/result.json" metrics.driver_match)
say "caps_honored=$CAPS driver_match=$DRV_OK"

PY_WHY=""; [[ -x $PY ]] || PY_WHY="no probe interpreter $PY"
S_WHY=""
for f in lib/libGLX_nvidia.so.0 lib/libEGL_nvidia.so.0 glvnd/libEGL.so.1 glvnd/libOpenGL.so.0 \
         vulkan/nvidia_icd.json egl/10_nvidia.json; do
  [[ -s $NVGL/$f ]] || { S_WHY="staged overlay incomplete: $NVGL/$f"; break; }
done
[[ -z $S_WHY && $DRV_OK != true ]] && S_WHY="kernel driver is not $DRV (the staged user-space libs would not match)"
R_WHY=""; [[ $CAPS == true ]] || R_WHY="caps_not_honored (NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES-unset})"
BUILD_WHY=""; [[ -x $BUILD/$BUILD_NAME ]] || BUILD_WHY="no AI2-THOR build at $BUILD"
LVP_WHY=""; [[ -n $LVP ]] || LVP_WHY="no lavapipe ICD under $ENV/share/vulkan/icd.d"
ENVLDR_WHY=""; [[ -e $ENV/lib/libvulkan.so.1 ]] || ENVLDR_WHY="no env loader $ENV/lib/libvulkan.so.1"
O3D_WHY=""; [[ -x $O3D_PY ]] || O3D_WHY="no open3d interpreter $O3D_PY"
# R: glvnd dispatch libs from the image if it has them, else the staged copies (they only dispatch)
R_EGL=$SYS_LIB/libEGL.so.1; [[ -s $R_EGL ]] || R_EGL=$NVGL/glvnd/libEGL.so.1
R_GL=$SYS_LIB/libOpenGL.so.0; [[ -s $R_GL ]] || R_GL=$NVGL/glvnd/libOpenGL.so.0
IMG_LDR=$SYS_LIB/libvulkan.so.1
THOR_V3=(--gpu-device 0 --scene FloorPlan1 --width 640 --height 360 --quality Ultra --aa 4 --steps 40 --depth --seg)

maybe "$(first "$PY_WHY" "$S_WHY")" v01_gl_tri_S 60 f_gl S "$NVGL/glvnd/libEGL.so.1" "$NVGL/glvnd/libOpenGL.so.0"
maybe "$(first "$PY_WHY" "$R_WHY")" v01_gl_tri_R 60 f_gl R "$R_EGL" "$R_GL"
maybe "$(first "$PY_WHY" "$S_WHY")" v02_vk_nvS_img 60 f_vk S "$IMG_LDR" nvidia
maybe "$(first "$PY_WHY" "$S_WHY" "$ENVLDR_WHY")" v02_vk_nvS_env 60 f_vk newloader "$ENV/lib/libvulkan.so.1" nvidia
maybe "$(first "$PY_WHY" "$R_WHY")" v02_vk_nvR 60 f_vk R "$IMG_LDR" nvidia
maybe "$(first "$PY_WHY" "$LVP_WHY" "$ENVLDR_WHY")" v02_vk_lvp 60 f_vk lvp "$ENV/lib/libvulkan.so.1" cpu
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$S_WHY")" v03_thor_diag_S 240 f_thor S "${THOR_V3[@]}"
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$R_WHY")" v04_thor_R 240 f_thor R "${THOR_V3[@]}"
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$S_WHY")" v05_thor_lowq 240 f_thor S "${THOR_V3[@]}" \
  --quality "Very Low" --aa none --width 300 --height 300
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$S_WHY" "$ENVLDR_WHY")" v06_thor_newloader 240 f_thor newloader "${THOR_V3[@]}"
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$S_WHY")" v07_thor_layers 240 f_thor layers "${THOR_V3[@]}"
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$S_WHY")" v08_thor_direct 240 f_thor S "${THOR_V3[@]}" \
  --unity-arg=-force-gfx-direct   # the = form: argparse would read a bare -force-... as an option
maybe "$(first "$O3D_WHY" "$S_WHY")" v09_open3d_egl 120 f_o3d
LEFT=$(( DEADLINE - $(date +%s) - RESERVE_S ))
V10_CAP=900; (( LEFT < V10_CAP )) && V10_CAP=$LEFT
maybe "$(first "$PY_WHY" "$BUILD_WHY" "$LVP_WHY" "$ENVLDR_WHY")" v10_thor_lvp_forced 900 f_thor lvpforced \
  --gpu-device 0 --scene FloorPlan1 --width 300 --height 300 --quality "Very Low" --aa none --steps 20 --depth \
  --startup-timeout $(( V10_CAP > 360 ? V10_CAP - 240 : V10_CAP / 2 ))
vpd_step summary summary
[[ -f $OUTDIR/summary.json ]] || { say "summary.json was not written"; exit 5; }
say "end after $(( $(date +%s) - T_START )) s"
exit 0
