#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$ROOT/third_party/starVLA}"
MODE="${1:?Usage: run_starvla_mobile.sh server|client|preflight [args]}"
shift
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${STARVLA_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
cd "$STARVLA_ROOT"
if [[ "$MODE" == server ]]; then
  source "$ROOT/examples/train/robocasa/setup_starvla_env.sh"
  exec python deployment/model_server/server_policy.py --ckpt_path "${CKPT:?Set CKPT}" --port "${PORT:-5678}" --use_bf16 "$@"
fi
ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365}"
POLICY_PY="${POLICY_PY:-$ROBOCASA365_ROOT/policies/diffusion_policy/.venv/bin/python}"
EGL_LIB_DIR="${EGL_LIB_DIR:-$ROOT/playground/Runtime/egl/usr/lib/x86_64-linux-gnu}"
if [[ -f "$EGL_LIB_DIR/libEGL.so.1" ]]; then
  export LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="${ROOT}:${STARVLA_ROOT}:${ROBOCASA365_ROOT}/robocasa:${ROBOCASA365_ROOT}/robomimic${PYTHONPATH:+:$PYTHONPATH}"
CLIENT_DEPS="${CLIENT_DEPS:-$ROOT/playground/Runtime/robocasa_client}"
if [[ -d "$CLIENT_DEPS" ]]; then
  export PYTHONPATH="$CLIENT_DEPS:$PYTHONPATH"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONFAULTHANDLER=1
if [[ "$MUJOCO_GL" == osmesa ]]; then
  # This image's llvmlite optimizer crashes during kitchen placement with Mesa.
  # Use the Python implementations of the same functions for the fallback.
  export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
fi
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/vagen-robocasa-numba-${UID}}"
export MESA_SHADER_CACHE_DIR="${MESA_SHADER_CACHE_DIR:-/tmp/vagen-robocasa-mesa-${UID}}"
mkdir -p "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR"
case "$MODE" in
  preflight) exec "$POLICY_PY" "$ROOT/scripts/eval_starvla_mobile.py" --render-only "$@" ;;
  client) exec "$POLICY_PY" "$ROOT/scripts/eval_starvla_mobile.py" --ckpt "${CKPT:?Set CKPT}" --port "${PORT:-5678}" "$@" ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
