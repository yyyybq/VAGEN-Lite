#!/bin/bash
# =============================================================================
# Gaussian Splatting 渲染服务器启动脚本（渲染机上运行）
# =============================================================================
#
# 用法：
#   bash start_gs_render_server.sh \
#       --gs-root /path/to/InteriorGS \
#       [--port 8777] \
#       [--gpus 0] \
#       [--num-shards 8] \
#       [--max-renderers 8]
#
# 环境变量方式（等价）：
#   GS_ROOT=/path/to/InteriorGS PORT=8777 CUDA_VISIBLE_DEVICES=0 \
#       bash start_gs_render_server.sh
#
# 训练机配置示例（experiments/vXX_xxx.sh）：
#   RENDER_MODE=remote
#   RENDER_HOST=<此机器的 IP 或主机名>
#   RENDER_PORT=8777
#   ENV_CONFIG=env_config_remote_render.yaml
#
# 依赖：
#   - conda 环境 vagen 需已安装 view_suite（ViewSuite 包）
#   - 渲染机能被训练机通过 TCP 8777 访问（检查防火墙/安全组）
# =============================================================================

set -e

# ========================= 解析参数 =========================
GS_ROOT="${GS_ROOT:-}"
PORT="${PORT:-8777}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_RENDERERS="${MAX_RENDERERS:-8}"
CONDA_ENV="${CONDA_ENV:-vagen}"
VIEWSUITE_PATH="${VIEWSUITE_PATH:-/scratch/by2593/project/Active_Spatial/ViewSuite}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gs-root)    GS_ROOT="$2";       shift 2 ;;
        --port)       PORT="$2";          shift 2 ;;
        --gpus)       GPUS="$2";          shift 2 ;;
        --num-shards) NUM_SHARDS="$2";    shift 2 ;;
        --max-renderers) MAX_RENDERERS="$2"; shift 2 ;;
        --conda-env)  CONDA_ENV="$2";     shift 2 ;;
        --viewsuite)  VIEWSUITE_PATH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ========================= 参数校验 =========================
if [ -z "$GS_ROOT" ]; then
    echo "ERROR: 必须指定 --gs-root（InteriorGS 数据根目录）"
    echo "用法: bash $0 --gs-root /path/to/InteriorGS"
    exit 1
fi

if [ ! -d "$GS_ROOT" ]; then
    echo "ERROR: GS_ROOT 目录不存在: $GS_ROOT"
    exit 1
fi

# ========================= 激活 Conda 环境 =========================
# shellcheck disable=SC1090
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || \
    source "${CONDA_BASE}/bin/activate" 2>/dev/null || true
conda activate "${CONDA_ENV}" 2>/dev/null || true

# 把 ViewSuite 加到 PYTHONPATH，不需要 pip install
export PYTHONPATH="${VIEWSUITE_PATH}:${PYTHONPATH:-}"

# 验证 view_suite 可用
if ! python -c "import view_suite" 2>/dev/null; then
    echo "ERROR: 在 ${VIEWSUITE_PATH} 下找不到 view_suite 包"
    echo "请确认 ViewSuite 路径正确，或用 --viewsuite /path/to/ViewSuite 指定"
    exit 1
fi

# ========================= 设置 CUDA =========================
export CUDA_VISIBLE_DEVICES="${GPUS}"

# ========================= 打印配置 =========================
PLY_COUNT=$(find "${GS_ROOT}" -name "*.ply" 2>/dev/null | wc -l)
echo "=============================================="
echo "  Gaussian Splatting 渲染服务器"
echo "=============================================="
echo "  GS_ROOT:       ${GS_ROOT}"
echo "  PLY 文件数:    ${PLY_COUNT}"
echo "  PORT:          ${PORT}"
echo "  GPUS:          ${GPUS}"
echo "  NUM_SHARDS:    ${NUM_SHARDS}"
echo "  MAX_RENDERERS: ${MAX_RENDERERS}"
echo "  CONDA_ENV:     ${CONDA_ENV}"
echo "  HOSTNAME:      $(hostname)"
echo "----------------------------------------------"
echo "  训练机 client_url:"
echo "    ws://$(hostname):${PORT}/render/interiorgs"
echo "  （若训练机只能访问 IP，请替换 hostname 为实际 IP）"
echo "=============================================="

if [ "$PLY_COUNT" -eq 0 ]; then
    echo "WARNING: GS_ROOT 下未发现 .ply 文件，请检查路径是否正确"
fi

# ========================= 启动渲染服务 =========================
echo "启动渲染服务，按 Ctrl+C 停止..."
python -m view_suite.interiorGS.service.gs_render_service \
    --interiorgs_root="${GS_ROOT}" \
    --num_shards="${NUM_SHARDS}" \
    --max_renderers_per_worker="${MAX_RENDERERS}" \
    --host="0.0.0.0" \
    --port="${PORT}" \
    --log_level="info"
