#!/bin/bash
# Install dependencies for local Gaussian Splatting rendering
#
# This script installs ALL rendering dependencies needed for active_spatial:
#   - ninja      (build backend for gsplat CUDA JIT compilation)
#   - gsplat     (3D Gaussian Splatting renderer)
#   - plyfile    (PLY file parser, required by ply_gaussian_loader)
#   - ply_gaussian_loader (single-file module copied from ViewSuite)
#
# Usage:
#   bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh
#   # Or specify a custom ViewSuite path:
#   VIEWSUITE_PATH=/path/to/ViewSuite bash scripts/examples/vagen_base/active_spatial/install_render_deps.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "==================================="
echo "Installing Rendering Dependencies"
echo "==================================="

# 1. Install ninja (required for gsplat CUDA JIT compilation)
echo "[1/4] Installing ninja..."
pip install ninja
# Ensure ninja is on PATH (needed by torch cpp_extension)
NINJA_BIN="$(python -c 'import ninja; import os; print(os.path.join(os.path.dirname(ninja.__file__), "..", "..", "..", "bin"))'  2>/dev/null || true)"
if ! command -v ninja &>/dev/null; then
    echo "  ⚠ ninja binary not on PATH. Adding conda/venv bin dir..."
    SITE_BIN="$(python -c 'import sys; print(sys.prefix)')/bin"
    export PATH="$SITE_BIN:$PATH"
fi
echo "  ninja location: $(which ninja)"

# 2. Install gsplat
echo "[2/4] Installing gsplat..."
pip install gsplat

# 3. Install plyfile (required by ply_gaussian_loader)
echo "[3/4] Installing plyfile..."
pip install plyfile

# 4. Copy ply_gaussian_loader from ViewSuite
#    This is NOT a pip package — it's a single .py file from the ViewSuite repo.
echo "[4/4] Installing ply_gaussian_loader..."
VIEWSUITE_PATH="${VIEWSUITE_PATH:-$(dirname "$PROJECT_ROOT")/ViewSuite}"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

if [ -f "$VIEWSUITE_PATH/ply_gaussian_loader.py" ]; then
    cp "$VIEWSUITE_PATH/ply_gaussian_loader.py" "$SITE_PACKAGES/ply_gaussian_loader.py"
    echo "  ✓ ply_gaussian_loader copied from $VIEWSUITE_PATH"
else
    echo "  ✗ ply_gaussian_loader.py not found at: $VIEWSUITE_PATH"
    echo "    Please set VIEWSUITE_PATH, e.g.:"
    echo "      VIEWSUITE_PATH=/path/to/ViewSuite bash $0"
    exit 1
fi

# 5. Pre-compile gsplat CUDA extensions (avoids JIT delay on first run)
echo ""
echo "Pre-compiling gsplat CUDA extensions (this may take ~2 minutes)..."
python -c "from gsplat.cuda._backend import _C; print('  ✓ gsplat CUDA extensions compiled')" 2>&1 || {
    echo "  ⚠ gsplat CUDA pre-compilation failed (will JIT-compile on first use)"
    echo "    Make sure CUDA toolkit and ninja are available."
}

# 6. Verify all installations
echo ""
echo "Verifying installation..."
python -c "import ninja; print('✓ ninja installed')"
python -c "import gsplat; print('✓ gsplat installed:', gsplat.__version__)"
python -c "import plyfile; print('✓ plyfile installed')"
python -c "from ply_gaussian_loader import PLYGaussianLoader; print('✓ ply_gaussian_loader installed')"

echo ""
echo "==================================="
echo "Rendering Dependencies Installed! ✓"
echo "==================================="
