#!/usr/bin/env bash
# source downstream/envs/activate_rlinf.sh ENV : the StarVLA + RoboCasa365 RLinf environment of the downstream line.
# PATH -> ENV/bin (its python sees the starVLA conda env's packages); RLinf, StarVLA and RoboCasa 1.0.1 from source;
# OSMesa rendering (EGL sees no device on the sco nodes).
_DS_ENV="${1:?usage: source activate_rlinf.sh ENV}"
_DS_MAIN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
_DS_RC=/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365
export PATH="$_DS_ENV/bin:/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin:$PATH"
export CUDA_HOME=/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA
export PYTHONPATH="$_DS_MAIN/third_party/RLinf:$_DS_MAIN/third_party/starVLA:$_DS_RC/robocasa:$_DS_RC/robomimic${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$_DS_MAIN/playground/Runtime/egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
