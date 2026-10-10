#!/usr/bin/env bash
# Run in a separate RLinf + StarVLA + RoboCasa365 Python environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="$(readlink -f "${1:?Usage: run_rlinf_starvla.sh prepared/train.yaml}")"
PY="${RLINF_PY:?Set RLINF_PY to the RLinf environment Python}"
export PYTHONPATH="$ROOT:$ROOT/third_party/RLinf:$ROOT/third_party/starVLA${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export ROBOT_PLATFORM=ROBOCASA365 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" "$ROOT/scripts/prepare_rlinf_robocasa.py" check --config "$CONFIG"
RLINF_ROOT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rlinf_root"])' "$(dirname "$CONFIG")/handoff.json")
export PYTHONPATH="$RLINF_ROOT:$PYTHONPATH"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
cd "$ROOT"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  # Compose the actual Hydra defaults without starting Ray, GPUs or simulation.
  exec "$PY" -c 'from hydra import compose, initialize_config_dir; from omegaconf import OmegaConf; import sys
with initialize_config_dir(config_dir=sys.argv[1], version_base="1.1"):
    cfg = compose(config_name="train")
    print(OmegaConf.to_yaml(cfg, resolve=True))' "$(dirname "$CONFIG")"
fi
exec "$PY" "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$(dirname "$CONFIG")" --config-name train
