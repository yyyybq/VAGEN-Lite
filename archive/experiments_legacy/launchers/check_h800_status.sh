#!/bin/bash
# =============================================================================
# VAGEN-Lite H800 多节点实验环境状态检查脚本
# 运行：bash check_h800_status.sh
# =============================================================================

SSH_KEY="/mnt/umm/users/yinbaiqiao/id_rsa"
SSH_OPTS="-i ${SSH_KEY} -o UserKnownHostsFile=/tmp/known_hosts -o StrictHostKeyChecking=no -n"
VAGEN_ROOT="/mnt/umm/users/yinbaiqiao/VAGEN-Lite"

echo "=========================================="
echo "VAGEN-Lite H800 Multi-Node Status Check"
echo "$(date)"
echo "=========================================="

echo ""
echo "=== 1. Conda 环境 ==="
if [ -f "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python" ]; then
    PY="/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
    echo "✓ Python: $($PY --version 2>&1)"
    $PY -c "import torch; print('✓ torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>/dev/null || echo "✗ torch not installed yet"
    $PY -c "import sglang; print('✓ sglang:', sglang.__version__)" 2>/dev/null || echo "✗ sglang not installed yet"
    $PY -c "import verl; print('✓ verl: installed')" 2>/dev/null || echo "✗ verl not installed yet"
    $PY -c "import vagen; print('✓ vagen: installed')" 2>/dev/null || echo "✗ vagen not installed yet"
    $PY -c "from gsplat.rendering import rasterization; print('✓ gsplat: installed')" 2>/dev/null || echo "✗ gsplat not installed yet"
else
    echo "✗ vagen-lite conda env not found (setup still running?)"
    echo "  Setup log: tail -f /mnt/umm/users/yinbaiqiao/setup_vagen_lite_env.log"
fi

echo ""
echo "=== 2. InteriorGS 数据集 ==="
INTERIORGS_DIR="/mnt/umm/users/yinbaiqiao/InteriorGS"
if [ -d "${INTERIORGS_DIR}" ]; then
    SCENE_COUNT=$(ls -d ${INTERIORGS_DIR}/*/  2>/dev/null | grep -v "^\." | wc -l)
    echo "✓ ${INTERIORGS_DIR}"
    echo "  场景数: ${SCENE_COUNT} (需要 ~103 个)"
    if [ ${SCENE_COUNT} -gt 90 ]; then
        echo "  状态: 充足"
    else
        echo "  状态: 不足！需要运行 download_interiorgs_scenes.py"
    fi
else
    echo "✗ InteriorGS 数据集未找到"
    echo "  请运行: HF_TOKEN=<your_token> python3 /mnt/umm/users/yinbaiqiao/download_interiorgs_scenes.py"
fi

echo ""
echo "=== 3. 训练数据 (JSONL) ==="
for f in \
  "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_6types.jsonl" \
  "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/data_gen/active_spatial_pipeline/output_v2/val_ood_v2_centering.jsonl"; do
    [ -f "$f" ] && echo "✓ $(basename $f)" || echo "✗ $(basename $f) MISSING"
done

echo ""
echo "=== 4. 节点 GPU 状态 ==="
for NODE in 10.119.21.155 10.119.21.185 10.119.21.237 10.119.31.56; do
    echo -n "  ${NODE}: "
    GPUS=$(ssh ${SSH_OPTS} root@${NODE} \
        "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader 2>/dev/null | awk -F', ' '\$2<5{printf \"GPU%s \", \$1}'" 2>/dev/null)
    echo "Free GPUs (util<5%): ${GPUS:-N/A}"
done

echo ""
echo "=== 5. 正在运行的实验 ==="
for f in ${VAGEN_ROOT}/exps/vagen_active_spatial/*/; do
    exp_name=$(basename "$f")
    prog="${f}progress.jsonl"
    if [ -f "${prog}" ]; then
        last=$(tail -1 "${prog}" 2>/dev/null)
        event=$(echo "${last}" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('event','?'))" 2>/dev/null)
        step=$(echo "${last}" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('global_step','?'))" 2>/dev/null)
        echo "  ${exp_name}: event=${event}, step=${step}"
    fi
done

echo ""
echo "=== 6. 启动命令 ==="
echo "  完整启动 (所有节点):"
echo "    bash ${VAGEN_ROOT}/examples/train/active_spatial/launch_h800_experiments.sh"
echo ""
echo "  单节点启动:"
echo "    ssh ${SSH_OPTS/.../...} root@10.119.21.155"
echo "    '< /dev/null CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup bash ${VAGEN_ROOT}/examples/train/active_spatial/run_experiment.sh"
echo "      ${VAGEN_ROOT}/examples/train/active_spatial/experiments/v46_7b_nodelta_w3.sh"
echo "      > v46.log 2>&1 &'"
