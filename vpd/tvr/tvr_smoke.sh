#!/usr/bin/env bash
# TVR render smoke, the sco JOB payload (WP-TVR, plan section 3.7; criteria frozen in vpd/tvr/SMOKE_CRITERIA.md).
# Ported from the WP-TVR design draft (scratchpad vpd_tvr/design/tvr_smoke.sh).  1 x H800, non-root (the
# sco_job owner wrapper setprivs to 20325), hard cap `timeout 1800` = 0.5 GPU-h = the whole TVR reservation.
#
#   PY_EVAL vpd/sco.py prepare --run OUT/jobs/tvr-smoke-<ts> --name vpd-tvr-smoke --est-gph 0.5 \
#       -- timeout 1800 bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/tvr/tvr_smoke.sh OUT/tvr/smoke_run1/out
#   PY_EVAL vpd/sco.py submit --run OUT/jobs/tvr-smoke-<ts>          # STOP-AND-ASK: orchestrator, after user OK
#
# Steps (each recorded in OUTDIR/steps.jsonl by vpd_step; a failed step never stops the run, the verdict decides):
#   0 node facts (+ /dev/nvidia-modeset, /dev/dri, df, cpu.max); exit 3 if the driver is not 595.71.05,
#     exit 4 if a staged input is missing (both after writing verdict.json).  On a 1-GPU node with a digit
#     CUDA_VISIBLE_DEVICES other than 0, CVD is re-exported as 0 (ai2thor maps gpu_device through CVD, and the
#     cuda-vulkan mapping is keyed by the in-container nvidia-smi index); recorded as facts.json cvd_remap.
#   1 ldd 'not found' check of the overlay and the Unity player
#   2 vulkaninfo --summary and full; cuda-vulkan-mapping decision (ai2thor native, else 1-card fallback)
#   3 thor_smoke.py: 300 steps 640x360 + 50 at 1280x720, then TVRBench ThorEnv on the first iTHOR/ProcTHOR task
#   4 thor_smoke.py --nproc 4 x 150 steps
#   5 egl_probe.py (E1-E5 context + clear readback; E6 GLSL triangle draw, reported as EGL_DRAW)
#   6 optional: 50-step lavapipe timing (skipped when the time budget is short): CVD=0, gpu_device 0 and a
#     pre-written cuda-vulkan-mapping {"0": 0} so Unity gets -force-device-index 0; thor_smoke --fail-fast.
#     (Fixed 2026-11-09: smoke_run1's step 6 passed --no-gpu-device and Unity aborted, so it measured nothing.)
#   7 verdict.py -> OUTDIR/verdict.json {TVR_GATE, EGL_GATE, reasons, caveats, steps_per_s, next}
# Time budget (worst case, every inner cap hit; the outer cap is timeout 1800):
#   0+1 ~10 s | 2 vulkaninfo 60+60 | 3 thor1 1020 (= 2 Controller start-ups of up to 300 s each + steps + tasks;
#   thor_smoke gets --deadline-s 990 and skips phase B if < 420 s are left) | 4 thor4 360 | 5 egl 90 |
#   6 lavapipe 240, only if it ends by t=1500 s | 7 verdict ~5 s  =>  <= ~1610 s without step 6, <= ~1510 s with it.
# thor_smoke.py rewrites its result.json after every stage, so an inner timeout keeps the finished measurements.
# Python runs as `-I -u`: -I ignores PYTHON* variables, so -u is what makes the logs unbuffered.
# Test overrides: VPD_COMMON_ENV, VPD_TVR_OUT, VPD_TVR_PY, VPD_TVR_VERDICT_PY, VPD_TVR_DRIVER_FILE,
#   VPD_TVR_REAP_WAIT (tenths of a second between TERM and KILL, default 50).
source "${VPD_COMMON_ENV:-/mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/common_env.sh}"
set +e   # common_env sets -euo pipefail; keep -u and pipefail, but run every step

[[ $# -eq 1 ]] || { echo "usage: $0 OUTDIR" >&2; exit 2; }
OUTDIR=$1
if [[ -e $OUTDIR/steps.jsonl || -e $OUTDIR/verdict.json ]]; then
  echo "refusing: $OUTDIR already holds a smoke run; give a new OUTDIR" >&2; exit 2
fi
mkdir -p "$OUTDIR/logs" || exit 2
OUTDIR=$(cd "$OUTDIR" && pwd)
export RUN=$OUTDIR   # vpd_step writes $RUN/steps.jsonl
vpd_chown_trap "$OUTDIR"   # no-op for the normal non-root run

VOUT=${VPD_TVR_OUT:-$VPD_OUT}
DRV=595.71.05
T=$VOUT/tvr
NVGL=$T/nvgl_$DRV
ENV=$VOUT/envs/tvr
AI2THOR_HOME=$T/ai2thor_home
REPO=$T/repos/TVRBench
BUILD_NAME=thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917
BUILD=$AI2THOR_HOME/.ai2thor/releases/$BUILD_NAME
# ai2thor starts the player as $HOME/.ai2thor/releases/<name>/<name> (HOME = the per-job symlinked one), so
# lingering players are matched on this HOME-independent suffix, never on $BUILD.
UNITY_PAT="/.ai2thor/releases/$BUILD_NAME/$BUILD_NAME"
THOR1_CAP=1020
THOR1_DEADLINE=990
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${VPD_TVR_PY:-$ENV/bin/python}
VERDICT_PY=${VPD_TVR_VERDICT_PY:-}
if [[ -z $VERDICT_PY ]]; then
  if [[ -x $PY ]]; then VERDICT_PY=$PY; else VERDICT_PY=$PY_EVAL; fi   # verdict.py is stdlib only
fi
DRIVER_FILE=${VPD_TVR_DRIVER_FILE:-/proc/driver/nvidia/version}
T_START=$(date +%s)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset ACTIVESPATIAL_USE_X11 ACTIVESPATIAL_X11_FORCE_VULKAN DISPLAY

final_verdict() { "$VERDICT_PY" -I -u "$S/verdict.py" verdict "$OUTDIR"; }
trap 'echo "[tvr] TERM: writing the verdict"; reap_unity term; final_verdict; exit 124' TERM

jstr() { local s=$1; s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/ }; printf '"%s"' "$s"; }
jenv() { if [[ -n ${!1+x} ]]; then jstr "${!1}"; else printf null; fi; }
jnum() { if [[ $1 =~ ^[0-9]+$ ]]; then printf '%s' "$1"; else printf null; fi; }
jbool() { if "$@" > /dev/null 2>&1; then printf true; else printf false; fi; }

node_facts() {
  vpd_node_facts "$OUTDIR/node.txt"
  {
    echo "## nvidia-smi -L"; nvidia-smi -L
    echo "## nvidia-smi uuid"; nvidia-smi --query-gpu=name,driver_version,uuid --format=csv
    echo "## driver"; cat "$DRIVER_FILE"
    echo "## env"; env | grep -E '^(NVIDIA|CUDA|VK_|__EGL|EGL|DISPLAY|LD_LIBRARY_PATH)' | sort
    echo "## image ICD / vendor files"
    ls -la /etc/vulkan/icd.d /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d; cat /etc/vulkan/icd.d/*.json
    echo "## image GL / Vulkan libs"
    ls -la /usr/lib/x86_64-linux-gnu | grep -e nvidia -e EGL -e GLX -e GLdispatch -e vulkan
    ls /usr/lib/x86_64-linux-gnu | grep -e '^libX11.so' -e '^libXext.so' -e '^libxcb' -e '^libwayland'
    echo "## os"; grep PRETTY /etc/os-release; ldd --version | head -1; uname -r
    echo "## df bytes (job-local disk for merges)"; df -B1 / /tmp /dev/shm
  } >> "$OUTDIR/node.txt" 2>&1
  nvidia-smi -L > "$OUTDIR/nvidia_smi_L.txt" 2>&1
  cat "$DRIVER_FILE" > "$OUTDIR/driver_version.txt" 2>&1
  local tmpfree shmfree cvd_json n_smi cvd_orig=${CUDA_VISIBLE_DEVICES-} remap=null
  tmpfree=$(df -B1 --output=avail /tmp 2>/dev/null | tail -1 | tr -d ' ')
  shmfree=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1 | tr -d ' ')
  cvd_json=$(jenv CUDA_VISIBLE_DEVICES)   # the value the job was given, before any remap
  # ai2thor 5.0.0 maps gpu_device=0 to int(CVD[0]); the cuda-vulkan mapping is keyed by the in-container
  # `nvidia-smi -L` index (0 on a 1-GPU job), so a physical index such as CVD=5 would be a KeyError.
  n_smi=$(grep -c '^GPU [0-9][0-9]*:' "$OUTDIR/nvidia_smi_L.txt")
  if [[ $n_smi -eq 1 && $cvd_orig =~ ^[0-9]+$ && $cvd_orig != 0 ]]; then
    export CUDA_VISIBLE_DEVICES=0
    remap="{\"from\": $(jstr "$cvd_orig"), \"to\": \"0\", \"n_smi_gpus\": 1}"
    echo "[tvr] CUDA_VISIBLE_DEVICES=$cvd_orig on a 1-GPU node: re-exported as 0 for ai2thor"
  fi
  printf '{"nvidia_modeset": %s, "dev_dri": %s, "render_node": %s, "cuda_visible_devices": %s, ' \
    "$(jbool test -e /dev/nvidia-modeset)" "$(jbool test -d /dev/dri)" \
    "$(jbool compgen -G '/dev/dri/renderD*')" "$cvd_json" > "$OUTDIR/facts.json"
  printf '"cvd_remap": %s, "n_smi_gpus": %s, ' "$remap" "$(jnum "$n_smi")" >> "$OUTDIR/facts.json"
  printf '"nvidia_driver_capabilities": %s, "nvidia_visible_devices": %s, "tmp_free_bytes": %s, ' \
    "$(jenv NVIDIA_DRIVER_CAPABILITIES)" "$(jenv NVIDIA_VISIBLE_DEVICES)" "$(jnum "$tmpfree")" >> "$OUTDIR/facts.json"
  printf '"shm_free_bytes": %s, "cpu_max": %s, "uid": %s, "host": %s}\n' "$(jnum "$shmfree")" \
    "$(jstr "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)")" "$(jnum "$(id -u)")" "$(jstr "$(hostname)")" \
    >> "$OUTDIR/facts.json"
  local pt=missing
  [[ -f $T/procthor_status.json ]] && pt=$(grep -o '"status": *"[a-z_]*"' "$T/procthor_status.json" | sed 's/.*"\([a-z_]*\)"$/\1/')
  {
    printf '{"nvgl_lib": %s, ' "$(jbool test -s "$NVGL/lib/libGLX_nvidia.so.0" -a -s "$NVGL/lib/libEGL_nvidia.so.0")"
    printf '"glvnd": %s, ' "$(jbool test -s "$NVGL/glvnd/libEGL.so.1" -a -s "$NVGL/glvnd/libOpenGL.so.0" -a -s "$NVGL/glvnd/libGLdispatch.so.0")"
    printf '"vulkan_icd": %s, "egl_vendor": %s, ' "$(jbool test -s "$NVGL/vulkan/nvidia_icd.json")" \
      "$(jbool test -s "$NVGL/egl/10_nvidia.json")"
    printf '"env_python": %s, "vulkaninfo": %s, ' "$(jbool test -x "$PY")" "$(jbool test -x "$ENV/bin/vulkaninfo")"
    printf '"build": %s, "tvrbench": %s, "procthor": %s}\n' "$(jbool test -x "$BUILD/$BUILD_NAME")" \
      "$(jbool test -f "$REPO/data/tasks/eval.json")" "$(jstr "${pt:-missing}")"
  } > "$OUTDIR/staging.json"
  cat "$OUTDIR/node.txt" "$OUTDIR/facts.json" "$OUTDIR/staging.json"
  return 0
}

ldd_check() {
  local f
  for f in "$NVGL/lib/libGLX_nvidia.so.0" "$NVGL/lib/libEGL_nvidia.so.0" "$NVGL/glvnd/libEGL.so.1" \
           "$NVGL/glvnd/libOpenGL.so.0" "$BUILD/$BUILD_NAME" "$BUILD/UnityPlayer.so"; do
    [[ -e $f ]] || continue
    echo "-- $f"; ldd "$f" 2>&1
  done > "$OUTDIR/ldd.txt"
  grep -e "=> not found" -e "^--" "$OUTDIR/ldd.txt"
  ! grep -q "=> not found" "$OUTDIR/ldd.txt"
}

vulkan() {
  timeout 60 vulkaninfo --summary > "$OUTDIR/vulkaninfo_summary.txt" 2> "$OUTDIR/logs/vulkaninfo_summary.err"
  local rc=$?
  timeout 60 vulkaninfo > "$OUTDIR/vulkaninfo_full.txt" 2> "$OUTDIR/logs/vulkaninfo_full.err"
  head -60 "$OUTDIR/vulkaninfo_summary.txt"; tail -5 "$OUTDIR/logs/vulkaninfo_summary.err"
  "$VERDICT_PY" -I -u "$S/verdict.py" mapping "$OUTDIR" "$HOME"
  return $rc
}

# Kill Unity players left behind by a thor step (e.g. the Vulkan-submit hang TVRBench reports on H200), so the
# next step gets the card: TERM, wait up to VPD_TVR_REAP_WAIT tenths of a second, then KILL.  Logged to
# logs/reap.log.
reap_unity() {
  local pids k wait=${VPD_TVR_REAP_WAIT:-50}
  if ! command -v pgrep > /dev/null || ! command -v pkill > /dev/null; then
    echo "[tvr] $1: pgrep/pkill missing, cannot reap Unity players" | tee -a "$OUTDIR/logs/reap.log"
    return 0
  fi
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  [[ -n ${pids// /} ]] || return 0
  echo "[tvr] $(date -u +%TZ) $1: TERM lingering Unity player(s): $pids" | tee -a "$OUTDIR/logs/reap.log"
  pkill -TERM -f -- "$UNITY_PAT"
  for ((k = 0; k < wait; k++)); do
    pgrep -f -- "$UNITY_PAT" > /dev/null || return 0
    sleep 0.1
  done
  pids=$(pgrep -f -- "$UNITY_PAT" | tr '\n' ' ')
  echo "[tvr] $(date -u +%TZ) $1: KILL Unity player(s) still alive: $pids" | tee -a "$OUTDIR/logs/reap.log"
  pkill -KILL -f -- "$UNITY_PAT"
  sleep 0.5
  return 0
}

thor1() {
  timeout $THOR1_CAP "$PY" -I -u "$S/thor_smoke.py" --out "$OUTDIR/thor1" --repo "$REPO" --steps 300 --gpu 0 \
    --deadline-s $THOR1_DEADLINE > "$OUTDIR/logs/thor1.log" 2>&1
  local rc=$?
  reap_unity thor1
  tail -60 "$OUTDIR/logs/thor1.log"
  # always keep the log as it was at exit: thor_smoke's own copies stop at the last stage it finished
  if [[ -f $PLAYERLOG ]]; then
    mkdir -p "$OUTDIR/thor1"; cp -p "$PLAYERLOG" "$OUTDIR/thor1/Player_after_exit.log"
  fi
  grep -i -h -m8 -e renderer -e vulkan -e llvmpipe -e lavapipe "$OUTDIR"/thor1/Player*.log 2>/dev/null
  return $rc
}

thor4() {
  timeout 360 "$PY" -I -u "$S/thor_smoke.py" --out "$OUTDIR/thor4" --repo "$REPO" --steps 150 --gpu 0 --nproc 4 \
    --no-tasks > "$OUTDIR/logs/thor4.log" 2>&1
  local rc=$?
  reap_unity thor4
  tail -20 "$OUTDIR/logs/thor4.log"
  return $rc
}

egl() {
  timeout 90 "$PY" -I -u "$S/egl_probe.py" --out "$OUTDIR/egl" > "$OUTDIR/logs/egl.log" 2>&1
  local rc=$?
  tail -30 "$OUTDIR/logs/egl.log"
  return $rc
}

lavapipe() {
  local lvp left
  left=$((1500 - ($(date +%s) - T_START)))
  if (( left < 240 )); then echo "skipped: only ${left}s left before the verdict budget"; return 0; fi
  lvp=$(ls "$ENV"/share/vulkan/icd.d/lvp_icd.*.json 2>/dev/null | head -1)
  [[ -n $lvp ]] || { echo "skipped: no lavapipe ICD under $ENV/share/vulkan/icd.d"; return 0; }
  (
    # Unity 2020.3 selects no CPU device by itself: without -force-device-index it aborts (smoke_run1:
    # "SelectPhysicalDevice requestedDeviceIndex=-1" -> "Selected physical device (nil)" -> SIGABRT, and the old
    # step then sat out its 240 s).  With the lvp ICD alone, llvmpipe is Vulkan device 0, so ai2thor must send
    # -force-device-index 0: gpu_device=0 with CUDA_VISIBLE_DEVICES=0, and a pre-written cuda-vulkan-mapping
    # {"0": 0} (ai2thor 5.0.0 controller.py:1030-1111 then skips its vulkaninfo / nvidia-smi UUID match, which
    # cannot work under an lvp-only ICD list).  LP_NUM_THREADS=12 of the job's 14 vCPUs rasterise.
    # --fail-fast: a fatal signal in Player.log or a dead player ends the step at once (exit 3).
    export CUDA_VISIBLE_DEVICES=0
    export VK_ICD_FILENAMES=$lvp VK_DRIVER_FILES=$lvp LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH" \
           HOME="$OUTDIR/home_lvp" LP_NUM_THREADS=12
    mkdir -p "$HOME/.ai2thor" && ln -sfn "$AI2THOR_HOME/.ai2thor/releases" "$HOME/.ai2thor/releases"
    printf '{"0": 0}\n' > "$HOME/.ai2thor/cuda-vulkan-mapping.json"
    timeout 240 "$PY" -I -u "$S/thor_smoke.py" --out "$OUTDIR/thor_lvp" --steps 50 --gpu 0 --no-tasks --fail-fast
  ) > "$OUTDIR/logs/thor_lvp.log" 2>&1
  local rc=$?
  reap_unity lavapipe
  tail -15 "$OUTDIR/logs/thor_lvp.log"
  return $rc
}

echo "[tvr] smoke start $(date -u +%FT%TZ) OUTDIR=$OUTDIR uid=$(id -u)"
vpd_step 0-node-facts node_facts
if ! grep -q " $DRV " "$OUTDIR/driver_version.txt"; then
  echo "[tvr] FAIL: the kernel driver is not $DRV; the copied user-space libs would abort (API mismatch)"
  vpd_step 7-verdict final_verdict
  exit 3
fi
if grep -q ': false' "$OUTDIR/staging.json"; then
  echo "[tvr] FAIL: a staged input is missing (see staging.json); run stage_tvr_login.sh first"
  vpd_step 7-verdict final_verdict
  exit 4
fi

# Overlay: our NVIDIA libs first, then glvnd 1.7.0.  The image's 0-byte 550.90.07 stubs are never loaded: every
# lookup goes through an absolute ICD/vendor path or a versioned soname that only exists in $NVGL/lib.
export LD_LIBRARY_PATH="$NVGL/lib:$NVGL/glvnd${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$NVGL/vulkan/nvidia_icd.json"   # the image's loader 1.2.131 reads only this name
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"                 # the env's loader 1.4.357 (vulkaninfo) prefers this
export __EGL_VENDOR_LIBRARY_FILENAMES="$NVGL/egl/10_nvidia.json"
export PATH="$ENV/bin:$PATH"                                # ai2thor runs `vulkaninfo` and `nvidia-smi -L`
# per-job HOME: ~/.ai2thor/cuda-vulkan-mapping.json and Unity's Player.log stay per job; the build is shared
export HOME="$OUTDIR/home"
mkdir -p "$HOME/.ai2thor"
ln -sfn "$AI2THOR_HOME/.ai2thor/releases" "$HOME/.ai2thor/releases"
PLAYERLOG="$HOME/.config/unity3d/Allen Institute for Artificial Intelligence/AI2-THOR/Player.log"

vpd_step 1-ldd ldd_check
vpd_step 2-vulkaninfo vulkan
vpd_step 3-thor1 thor1
vpd_step 4-thor4 thor4
vpd_step 5-egl egl
vpd_step 6-lavapipe lavapipe
vpd_step 7-verdict final_verdict
[[ -f $OUTDIR/verdict.json ]] || { echo "[tvr] verdict.json was not written"; exit 5; }
echo "[tvr] smoke end $(date -u +%FT%TZ) after $(( $(date +%s) - T_START )) s"
exit 0
