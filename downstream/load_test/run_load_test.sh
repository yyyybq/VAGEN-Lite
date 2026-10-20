#!/usr/bin/env bash
# Step-1 one-card job of the downstream line (see docs/downstream/reports/STEP1_INVENTORY_*.md):
#   1. CPU: are the start weights complete HF exports (all starts in downstream/starts.json)?
#   2. GPU: each "main" start loaded as StarVLA QwenOFT's backbone, fake batch forward + predict_action.
#   3. CPU: can a RoboCasa365 kitchen render off-screen on this node (EGL, then OSMesa)?
# Reads weights, StarVLA and RoboCasa only by path; writes only under OUT.  Usage: run_load_test.sh OUT
set -uo pipefail
OUT="${1:?Usage: run_load_test.sh OUT_DIR}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAIN=/mnt/umm/users/yinbaiqiao/VAGEN-Lite                 # main checkout: third_party/ and playground/ live only there
STARVLA_ROOT="$MAIN/third_party/starVLA"
ROBOCASA365_ROOT=/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365
SIM_PY="$ROBOCASA365_ROOT/policies/diffusion_policy/.venv/bin/python"   # the closed-loop client's simulator Python
CFG="$WT/examples/train/robocasa/starvla_qwenoft_navigatekitchen_smoke.yaml"
mkdir -p "$OUT"
cd "$WT"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled NO_PROXY='*' no_proxy='*' CUDA_VISIBLE_DEVICES=0
export HF_HOME=/mnt/umm/users/yinbaiqiao/.cache/huggingface
echo "[step1] worktree=$WT out=$OUT (code commit: job/prepared.json; the image has no git)"

starts() {  # names of starts with the given role, from starts.json
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(k for k,v in d.items() if not k.startswith("_") and (sys.argv[2]=="all" or v["role"]==sys.argv[2])))' "$WT/downstream/starts.json" "$1"
}
path_of() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["path"])' "$WT/downstream/starts.json" "$1"; }

# --- environment of StarVLA (Python 3.10, torch 2.6) ---
source "$WT/examples/train/robocasa/setup_starvla_env.sh"
set +e  # the setup script turns on -e; one failing start must not stop the others
export PYTHONPATH="$WT:$STARVLA_ROOT"

echo "[step1] 1/3 HF export check $(date -u +%T)"
ARGS=()
for s in $(starts all); do ARGS+=(--start "$s=$(path_of "$s")"); done
python "$WT/downstream/load_test/check_hf_export.py" "${ARGS[@]}" --out "$OUT/hf_export.json"

echo "[step1] 2/3 StarVLA load test $(date -u +%T)"
for s in $(starts main); do
  echo "[step1] load $s $(date -u +%T)"
  python "$WT/downstream/load_test/load_in_starvla.py" --name "$s" --path "$(path_of "$s")" --config "$CFG" \
    --out "$OUT/load_$s.json" 2>&1 | grep -vE "it/s\]$" | tail -40
  nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
done

echo "[step1] 3/3 render check $(date -u +%T)"
export PYTHONPATH="$WT:$STARVLA_ROOT:$ROBOCASA365_ROOT/robocasa:$ROBOCASA365_ROOT/robomimic"
[[ -d "$MAIN/playground/Runtime/robocasa_client" ]] && export PYTHONPATH="$MAIN/playground/Runtime/robocasa_client:$PYTHONPATH"
EGL_LIB_DIR="$MAIN/playground/Runtime/egl/usr/lib/x86_64-linux-gnu"
export NUMBA_CACHE_DIR=/tmp/ds-numba-$UID MESA_SHADER_CACHE_DIR=/tmp/ds-mesa-$UID
mkdir -p "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR"
ls /dev/dri 2>&1 | head -2; echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-unset}"
LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 \
  timeout 600 "$SIM_PY" "$WT/downstream/load_test/render_check.py" --out "$OUT/render_egl.json" 2>&1 | tail -15
echo "[step1] egl rc=${PIPESTATUS[0]}"
LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 \
  timeout 900 "$SIM_PY" "$WT/downstream/load_test/render_check.py" --out "$OUT/render_osmesa.json" 2>&1 | tail -15
echo "[step1] osmesa rc=${PIPESTATUS[0]}"
rm -rf "$NUMBA_CACHE_DIR" "$MESA_SHADER_CACHE_DIR"
echo "[step1] done $(date -u +%T)"
