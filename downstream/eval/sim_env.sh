# Source me (bash): the RoboCasa365 simulator environment of the downstream evaluation (CPU, OSMesa), the same one as
# step 2 (downstream/step2/run_step2.sh).  Defines MAIN, STARVLA_ROOT, ROBOCASA365_ROOT, SIM_PY and
#   sim THREADS COMMAND...   run COMMAND with the simulator PYTHONPATH, OSMesa and THREADS llvmpipe threads
# and creates the per-user caches /tmp/ds-numba-$UID and /tmp/ds-mesa-$UID (the caller removes them).
# Needs WT (the worktree root) to be set.
MAIN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
STARVLA_ROOT="$MAIN/third_party/starVLA"
ROBOCASA365_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365
SIM_PY="$ROBOCASA365_ROOT/policies/diffusion_policy/.venv/bin/python"
SIM_PYTHONPATH="$MAIN/playground/Runtime/robocasa_client:$WT:$STARVLA_ROOT:$ROBOCASA365_ROOT/robocasa:$ROBOCASA365_ROOT/robomimic"
EGL_LIB_DIR="$MAIN/playground/Runtime/egl/usr/lib/x86_64-linux-gnu"
sim() {
  local threads=$1; shift
  env PYTHONPATH="$SIM_PYTHONPATH" LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 LP_NUM_THREADS="$threads" \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    NUMBA_CACHE_DIR="/tmp/ds-numba-$UID" MESA_SHADER_CACHE_DIR="/tmp/ds-mesa-$UID" "$@"
}
mkdir -p "/tmp/ds-numba-$UID" "/tmp/ds-mesa-$UID"
