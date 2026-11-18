#!/bin/bash
# q38 line (Qwen3.8-27B rerun): the training / rollout / evaluation environment, after pear's
# scripts/install/verl/install.sh (install_packages, install_cuda_toolkit, flash-attn-4), with pear's pinned versions
# (constraints.txt, overrides.txt of our copy of pear, f32aa1f7). Differences from pear's installer:
#   - our own directories under /mnt/umm/users/yinbaiqiao/q38 (nothing under embodied/);
#   - our copy of pear's verl tree (VAGEN-Lite-q38/verl, 8e03c039 + 43 pear patches) and our copy of the pear package;
#   - no source builds here (flash-attn 2 / 3 are built in a one-card sco job), no LMCache, no LIBERO;
#   - the login-node CPU rule: at most 4 cores (2 concurrent installs, 4 downloads), nice -n 19 by the caller;
#   - proxies unset: PyPI, download.pytorch.org and GitHub are 7-20 MB/s direct, 0.04-0.3 MB/s through the proxy
#     (measured 2026-11-17).
# Run: nohup nice -n 19 bash env/q38/install_env.sh >> /mnt/umm/users/yinbaiqiao/q38/logs/install_env.log 2>&1 &
set -euo pipefail
Q=/mnt/umm/users/yinbaiqiao/q38
PEAR=$Q/src/pear
VERL=/mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38/verl
ENV=$Q/envs/verl
PY=$ENV/bin/python
UV_VERSION=0.12.24
PYTHON_VERSION=3.12.15

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export UV_CACHE_DIR=$Q/cache/uv UV_PYTHON_INSTALL_DIR=$Q/envs/uv-python UV_TOOL_DIR=$Q/cache/uv_tools
export PIP_CACHE_DIR=$Q/cache/pip UV_CREDENTIALS_DIR=$Q/cache/uv_credentials
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1 MAX_JOBS=2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$Q"/{tools,cache,envs,wheels,logs}
step() { echo; echo "=== $(date -u +%FT%TZ) $*"; }

step "uv $UV_VERSION (PyPI wheel, into $Q/tools)"
if [ "$("$Q/tools/uv-$UV_VERSION/bin/uv" --version 2>/dev/null | cut -d ' ' -f 2)" != "$UV_VERSION" ]; then
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m pip install -q --no-deps \
    --target "$Q/tools/uv-$UV_VERSION" "uv==$UV_VERSION"
fi
UV=$Q/tools/uv-$UV_VERSION/bin/uv
$UV --version

step "python $PYTHON_VERSION and the environment $ENV"
$UV python install "$PYTHON_VERSION"
if [ "$("$PY" -c 'import platform; print(platform.python_version())' 2>/dev/null || true)" != "$PYTHON_VERSION" ]; then
  $UV venv --clear --python "$PYTHON_VERSION" "$ENV"
fi

export UV_CONSTRAINT=$PEAR/scripts/install/verl/constraints.txt UV_OVERRIDE=$PEAR/scripts/install/verl/overrides.txt
step "packages: pear's verl extra + verl's requirements.txt, pinned by pear's constraints / overrides"
$UV pip install --python "$PY" --torch-backend=cu130 -r "$PEAR/pyproject.toml" --extra verl -r "$VERL/requirements.txt"

step "editable: our verl copy and our pear copy (no dependencies)"
$UV pip install --python "$PY" --no-deps -e "$VERL" -e "$PEAR"

step "CUDA toolkit wheels (verl_gpu extra) and FlashInfer's precompiled kernels"
$UV pip install --python "$PY" --find-links https://flashinfer.ai/whl/flashinfer-cubin/ -r "$PEAR/pyproject.toml" --extra verl_gpu
CUDA_TOOLKIT=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13
ln -sfn lib "$CUDA_TOOLKIT/lib64"
for library in "$CUDA_TOOLKIT"/lib/lib*.so.[0-9]*; do
  unversioned=${library%%.so.*}.so
  [ -e "$unversioned" ] || ln -s "${library##*/}" "$unversioned"
done
echo "CUDA_TOOLKIT=$CUDA_TOOLKIT"

step "flash-attn-4 (pure-python CuTe DSL kernels, as pear installs it)"
$UV pip install --python "$PY" --no-deps --prerelease=allow flash-attn-4

step "VAGEN's extra runtime need (scipy, for vagen/envs/active_spatial/utils.py)"
$UV pip install --python "$PY" scipy

step "flash-attn 2.8.3.post1 source archive, for the one-card sco build job"
mkdir -p "$Q/wheels/src"
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python -m pip download -q --no-deps --no-binary :all: \
  "flash-attn==2.8.3.post1" -d "$Q/wheels/src" || echo "flash-attn sdist download failed (non-fatal)"
ls -la "$Q/wheels/src"

step "import check (CPU only)"
"$PY" -I -B - <<'EOF'
import importlib, platform
print("python", platform.python_version())
for name in ["torch", "transformers", "vllm", "verl", "pear", "fla", "flashinfer", "ray", "tensordict",
             "transfer_queue", "peft", "accelerate", "qwen_vl_utils", "scipy", "flash_attn"]:
    try:
        m = importlib.import_module(name)
        print(f"  {name:16s} {getattr(m, '__version__', '?')}  {getattr(m, '__file__', '')}")
    except Exception as e:
        print(f"  {name:16s} FAILED {type(e).__name__}: {str(e)[:160]}")
import torch
print("torch cuda build", torch.version.cuda)
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B")
print("config", type(cfg).__name__, cfg.model_type)
EOF

step "sizes"
du -sh "$ENV" "$Q/envs/uv-python" "$UV_CACHE_DIR" 2>/dev/null || true
step "done"
