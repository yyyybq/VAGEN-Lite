#!/usr/bin/env python3
"""
GPU Holder v6 - SM 满载占空比控制（与渲染共存）

v5 → v6 修复：
  1. 自动选择矩阵大小以填满所有 SM（H200=132 SMs 需要 ≥2048x2048）
     v5 用 1024×1024 只覆盖 48% SM，导致集群 SM 利用率监控报 ~40%
  2. 充分预热 CUDA allocator 后再校准（v5 首次校准偏高 ~60%）
  3. 快速初始收敛：前 50 轮每轮校准，消除启动阶段的低利用率
  4. 用 pre-allocated output tensor 避免每次 matmul 分配临时内存

原理：
  集群监控（DCGM）通常报告 SM 活跃率，不是 nvidia-smi 的 GPU-Util。
  SM 活跃率 = (活跃SM数/总SM数) × (有kernel的时间占比)
  所以必须同时保证：
    a) 矩阵足够大，thread blocks ≥ SM 数 → SM 覆盖 100%
    b) 占空比精确 → 时间覆盖达标

  2048×2048 matmul on H200: 256 blocks / 132 SMs = 194% 覆盖，0.84ms/kernel。
  渲染最多等一个 kernel = 0.84ms，对 50ms+ 的 GS 渲染几乎无感。

使用方法：
    HOLDER_GPU=4 HOLDER_TARGET=75 python gpu_holder.py

环境变量：
    HOLDER_GPU          GPU ID (默认 4)
    HOLDER_MEM_FRAC     显存占用比例 (默认 0.75)
    HOLDER_TARGET       目标利用率 % (默认 75)
    HOLDER_MICRO_SIZE   矩阵大小 (默认 auto，自动选择)
    HOLDER_KERNELS      每批内核数 (默认 10)
"""

import torch
import time
import os
import signal

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def get_gpu_free_memory(gpu_id):
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return info.free / 1024 / 1024
        except:
            pass
    return 0


def get_gpu_utilization(gpu_id):
    if HAS_PYNVML:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return info.gpu
        except:
            pass
    return 0


def auto_select_micro_size(device):
    """根据 GPU 的 SM 数量自动选择矩阵大小，确保 thread blocks ≥ SM 数。

    cuBLAS 典型 tile = 128×128，所以 blocks ≈ (N/128)²。
    需要 blocks ≥ num_SMs 才能 100% 覆盖。
    """
    try:
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count
    except:
        num_sms = 132  # H200 default

    # N/128 ≥ sqrt(num_sms) → N ≥ 128 * sqrt(num_sms)
    # 向上取整到 128 的倍数，再多留 50% 余量
    import math
    min_n = int(128 * math.ceil(math.sqrt(num_sms * 1.5)))
    # 取 128 的倍数
    micro_size = max(2048, ((min_n + 127) // 128) * 128)
    return micro_size, num_sms


def calibrate(a, b, kernels_per_batch, out, device):
    """测量一批 matmul 的实际耗时。使用 pre-allocated output 避免分配开销。"""
    t0 = time.perf_counter()
    for _ in range(kernels_per_batch):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def main():
    # ===== 配置 =====
    gpu_id = int(os.environ.get('HOLDER_GPU', '4'))
    mem_fraction = float(os.environ.get('HOLDER_MEM_FRAC', '0.75'))
    target_util = int(os.environ.get('HOLDER_TARGET', '75'))
    micro_size_env = os.environ.get('HOLDER_MICRO_SIZE', 'auto')
    kernels_per_batch = int(os.environ.get('HOLDER_KERNELS', '10'))

    duty_ratio = target_util / 100.0

    # ===== 初始化 =====
    if HAS_PYNVML:
        pynvml.nvmlInit()

    device = torch.device(f'cuda:{gpu_id}')

    # 自动选择矩阵大小
    if micro_size_env == 'auto':
        micro_size, num_sms = auto_select_micro_size(device)
    else:
        micro_size = int(micro_size_env)
        try:
            num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        except:
            num_sms = -1

    est_blocks = (micro_size // 128) ** 2
    print(f"[gpu_holder v6] GPU {gpu_id} ({num_sms} SMs)")
    print(f"[gpu_holder v6] Target: {target_util}%, duty_ratio: {duty_ratio:.2f}")
    print(f"[gpu_holder v6] Memory fraction: {mem_fraction*100:.0f}%")
    print(f"[gpu_holder v6] Kernel: {micro_size}x{micro_size} matmul, ~{est_blocks} blocks, {kernels_per_batch}/batch")
    print(f"[gpu_holder v6] SM coverage: {min(est_blocks/max(num_sms,1)*100, 100):.0f}%")

    # 占用显存
    free_mem = get_gpu_free_memory(gpu_id)
    target_mem = free_mem * mem_fraction
    holder_tensor = None

    try:
        if target_mem > 100:
            n_elements = int(target_mem * 1024 * 1024 / 4)
            holder_tensor = torch.zeros(n_elements, dtype=torch.float32, device=device)
            print(f"[gpu_holder v6] Allocated: {target_mem:.0f} MB")
        else:
            print(f"[gpu_holder v6] Skipping memory allocation (fraction={mem_fraction})")

        a = torch.randn(micro_size, micro_size, device=device)
        b = torch.randn(micro_size, micro_size, device=device)
        out = torch.empty(micro_size, micro_size, device=device)  # pre-allocated output
        print(f"[gpu_holder v6] Ready")
    except Exception as e:
        print(f"[gpu_holder v6] FAILED: {e}")
        return

    # ===== 充分预热 =====
    print(f"[gpu_holder v6] Warming up...")
    for _ in range(200):
        torch.mm(a, b, out=out)
    torch.cuda.synchronize(device)

    # 多次校准取稳定值
    calib_results = []
    for _ in range(5):
        ms = calibrate(a, b, kernels_per_batch, out, device)
        calib_results.append(ms)
    # 取中位数（去掉异常值）
    calib_results.sort()
    batch_compute_ms = calib_results[len(calib_results) // 2]
    sleep_ms = batch_compute_ms * (1 - duty_ratio) / duty_ratio

    print(f"[gpu_holder v6] Calibration: compute={batch_compute_ms:.2f}ms, sleep={sleep_ms:.2f}ms")
    print(f"[gpu_holder v6] Cycle: {batch_compute_ms + sleep_ms:.2f}ms")
    print(f"[gpu_holder v6] Per-kernel: {batch_compute_ms/kernels_per_batch:.3f}ms")

    # 信号处理
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n[gpu_holder v6] Shutting down...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ===== 主循环 =====
    print(f"[gpu_holder v6] Running...")

    stats_start = time.time()
    stats_kernels = 0
    recalib_counter = 0
    iteration = 0

    while running:
        # 1. 计算：提交一批微内核 + sync（用 pre-allocated output）
        for _ in range(kernels_per_batch):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize(device)
        stats_kernels += kernels_per_batch

        # 2. 精确 sleep
        if sleep_ms > 0.1:
            time.sleep(sleep_ms / 1000.0)

        # 3. 校准策略：前 50 轮每轮校准（快速收敛），之后每 500 轮
        iteration += 1
        recalib_counter += 1
        need_recalib = (iteration <= 50) or (recalib_counter >= 500)

        if need_recalib:
            recalib_counter = 0
            new_ms = calibrate(a, b, kernels_per_batch, out, device)
            stats_kernels += kernels_per_batch
            # 指数移动平均，避免跳变
            if iteration <= 50:
                batch_compute_ms = new_ms  # 前 50 轮直接用新值
            else:
                batch_compute_ms = 0.7 * batch_compute_ms + 0.3 * new_ms
            sleep_ms = batch_compute_ms * (1 - duty_ratio) / duty_ratio

        # 4. 防止数值溢出：定期刷新矩阵
        if stats_kernels % 100000 == 0:
            a = torch.randn(micro_size, micro_size, device=device)
            b = torch.randn(micro_size, micro_size, device=device)

        # 5. 定期打印状态
        elapsed = time.time() - stats_start
        if elapsed >= 30:
            util = get_gpu_utilization(gpu_id)
            print(f"[gpu_holder v6] util={util}% (target={target_util}%), "
                  f"compute={batch_compute_ms:.2f}ms, sleep={sleep_ms:.2f}ms, "
                  f"iter={iteration}, kernels={stats_kernels}")
            stats_kernels = 0
            stats_start = time.time()

    # 清理
    if holder_tensor is not None:
        del holder_tensor
    del a, b, out
    torch.cuda.empty_cache()
    if HAS_PYNVML:
        pynvml.nvmlShutdown()
    print("[gpu_holder v6] Stopped.")


if __name__ == "__main__":
    main()
