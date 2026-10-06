#!/usr/bin/env bash
# One GPU, existing NavigateKitchen baseline; validates the new closed-loop path.
set -euo pipefail
ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"
export CKPT="${CKPT:-$ROOT/third_party/starVLA/playground/Checkpoints/starvla_qwenoft_NavigateKitchen_20261001_025134/final_model/pytorch_model.pt}"
export PORT="${PORT:-5688}"
OUT="${OUT:-$ROOT/playground/Experiments/starvla_transfer/closed_loop_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/worker.log") 2>&1
nvidia-smi -L
ENTRY="$ROOT/examples/evaluate/robocasa/run_starvla_mobile.sh"
if ! timeout 120 bash "$ENTRY" preflight --out "$OUT/render_egl"; then
  # Explicitly recorded CPU render fallback; policy inference stays on H800.
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  timeout 120 bash "$ENTRY" preflight --out "$OUT/render_osmesa"
fi
server_pid=""
cleanup() { if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
bash "$ENTRY" server --config_override framework.qwenvl.attn_implementation=sdpa > "$OUT/server.log" 2>&1 &
server_pid=$!
# The websocket client waits for the server; it validates checkpoint identity.
timeout 1800 bash "$ENTRY" client --task NavigateKitchen --episodes 2 --max-steps 200 \
  --split target --video --out "$OUT/eval"
