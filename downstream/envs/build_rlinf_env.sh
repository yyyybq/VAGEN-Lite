#!/usr/bin/env bash
# Build the "StarVLA + RoboCasa365" RLinf environment (decision D7) on the login node.
#   ENV = a venv with --system-site-packages on top of the starVLA conda env (torch 2.6.0+cu124, flash-attn 2.7.4):
#   1. pip install -r rlinf_starvla_robocasa365.txt   2. pip install --no-deps -r rlinf_nodeps.txt   3. import check
# Neither the starVLA env nor the existing RLinf env (third_party/RLinf/venv_mnt) is changed; RLinf, StarVLA and
# RoboCasa 1.0.1 come from source via activate_rlinf.sh.  Installs from the Aliyun PyPI mirror without the proxy
# (files.pythonhosted.org through the proxy timed out).  Usage: build_rlinf_env.sh ENV LOG_DIR
set -uo pipefail
ENV="${1:?Usage: build_rlinf_env.sh ENV LOG_DIR}"
LOG="${2:?}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_PY=/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python
PIPX=(--no-cache-dir --timeout 120 --retries 10 -i https://mirrors.aliyun.com/pypi/simple/)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
mkdir -p "$LOG"
[[ -x "$ENV/bin/python" ]] || "$STARVLA_PY" -m venv --system-site-packages "$ENV" || exit 2
"$ENV/bin/python" -m pip install "${PIPX[@]}" -r "$HERE/rlinf_starvla_robocasa365.txt" > "$LOG/build_main.log" 2>&1 \
  || { echo "[build] main install failed: $LOG/build_main.log"; exit 3; }
"$ENV/bin/python" -m pip install "${PIPX[@]}" --no-deps -r "$HERE/rlinf_nodeps.txt" > "$LOG/build_nodeps.log" 2>&1 \
  || { echo "[build] no-deps install failed: $LOG/build_nodeps.log"; exit 4; }
"$ENV/bin/python" -m pip freeze > "$LOG/pip_freeze.txt"
source "$HERE/activate_rlinf.sh" "$ENV"
python - > "$LOG/import_check.json" 2>"$LOG/import_check.err" <<'PY'
import importlib, json
out = {}
for m in ["torch", "numpy", "transformers", "accelerate", "flash_attn", "ray", "hydra", "rlinf", "starVLA",
          "qwen_vl_utils", "mujoco", "robosuite", "robocasa", "robocasa.utils.dataset_registry_utils",
          "rlinf.envs.sim.robocasa365.robocasa365_env", "rlinf.models.embodiment.starvla"]:
    try:
        mod = importlib.import_module(m)
        out[m] = "ok " + str(getattr(mod, "__version__", ""))
    except Exception as exc:
        out[m] = f"FAIL {type(exc).__name__}: {str(exc)[:200]}"
print(json.dumps(out, indent=1))
PY
cat "$LOG/import_check.json"
du -sh "$ENV" | tee "$LOG/env_size.txt"
