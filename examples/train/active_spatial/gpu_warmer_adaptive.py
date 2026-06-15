#!/usr/bin/env python3
"""
Adaptive GPU Warmer - 自适应 GPU 利用率保持器

核心逻辑：
- 每隔一小段时间检测各 GPU 的利用率
- 利用率 < LOW_THRESHOLD 时：全力做矩阵乘法，快速拉高利用率
- 利用率在 LOW_THRESHOLD ~ HIGH_THRESHOLD 之间：轻度计算维持
- 利用率 > HIGH_THRESHOLD 时：完全让出 GPU，不做任何计算
- 使用低优先级 CUDA stream，尽量不影响训练/推理

这样训练和推理跑满 GPU 时 warmer 自动退避，
空闲时（rollout↔training 切换间隙）warmer 填充利用率。
"""
import torch
import time
import os
import signal
import sys

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def get_gpu_utilizations_pynvml(gpu_ids):
    """用 pynvml 获取 GPU 利用率（快，无子进程开销）"""
    utils = {}
    for gid in gpu_ids:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gid)
            info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utils[gid] = info.gpu
        except Exception:
            utils[gid] = 100  # 出错时假设满了，不做计算
    return utils


def get_gpu_utilizations_smi(gpu_ids):
    """用 nvidia-smi 获取 GPU 利用率（备用方案）"""
    import subprocess
    utils = {}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            if len(parts) == 2:
                idx = int(parts[0].strip())
                util = int(parts[1].strip())
                if idx in gpu_ids:
                    utils[idx] = util
    except Exception:
        for gid in gpu_ids:
            utils[gid] = 100
    return utils


def main():
    # ===== 配置 =====
    # 利用率阈值
    LOW_THRESHOLD = int(os.environ.get('WARMER_LOW', '50'))    # 低于此值：全力计算
    HIGH_THRESHOLD = int(os.environ.get('WARMER_HIGH', '65'))  # 高于此值：完全停止
    
    # 矩阵大小（影响单次计算量和显存占用）
    matrix_size = int(os.environ.get('WARMER_SIZE', '3072'))
    
    # 检测频率（秒）
    poll_interval = float(os.environ.get('WARMER_POLL_S', '0.3'))
    
    # 全力计算时的 burst 次数
    burst_count = int(os.environ.get('WARMER_BURST', '10'))
    
    # 指定 GPU（逗号分隔），不设置则使用所有可见 GPU
    gpu_env = os.environ.get('WARMER_GPUS', None)
    
    if gpu_env:
        gpu_ids = [int(x) for x in gpu_env.split(',')]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    
    if not gpu_ids:
        print("[adaptive_warmer] No GPUs available")
        return
    
    # ===== 初始化 pynvml =====
    if HAS_PYNVML:
        pynvml.nvmlInit()
        get_utils = get_gpu_utilizations_pynvml
        print("[adaptive_warmer] Using pynvml for GPU monitoring")
    else:
        get_utils = get_gpu_utilizations_smi
        print("[adaptive_warmer] Using nvidia-smi for GPU monitoring (install pynvml for better perf)")
    
    # ===== 为每个 GPU 创建低优先级 stream 和 tensor =====
    streams = {}
    tensors_a = {}
    tensors_b = {}
    
    for gid in gpu_ids:
        try:
            device = torch.device(f'cuda:{gid}')
            # 低优先级 stream（0 = default, 负数 = 更低优先级）
            stream = torch.cuda.Stream(device=device, priority=-1)  # lowest priority
            a = torch.randn(matrix_size, matrix_size, device=device)
            b = torch.randn(matrix_size, matrix_size, device=device)
            streams[gid] = stream
            tensors_a[gid] = a
            tensors_b[gid] = b
            print(f"[adaptive_warmer] GPU {gid}: ready (matrix {matrix_size}x{matrix_size})")
        except Exception as e:
            print(f"[adaptive_warmer] GPU {gid}: failed - {e}")
    
    if not streams:
        print("[adaptive_warmer] No streams created, exiting")
        return
    
    # ===== 信号处理 =====
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n[adaptive_warmer] Shutting down...")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # ===== 主循环 =====
    print(f"[adaptive_warmer] Running on GPUs {list(streams.keys())}")
    print(f"[adaptive_warmer] Thresholds: sleep>{HIGH_THRESHOLD}%, active<{LOW_THRESHOLD}%")
    print(f"[adaptive_warmer] Poll interval: {poll_interval}s, burst: {burst_count}")
    
    iteration = 0
    while running:
        # 获取当前 GPU 利用率
        utils = get_utils(list(streams.keys()))
        
        for gid in streams:
            util = utils.get(gid, 100)
            
            if util >= HIGH_THRESHOLD:
                # GPU 很忙，完全不做计算
                continue
            elif util < LOW_THRESHOLD:
                # GPU 空闲，全力做计算
                n_bursts = burst_count
            else:
                # 中间地带，轻度计算
                n_bursts = max(1, burst_count // 4)
            
            stream = streams[gid]
            a = tensors_a[gid]
            b = tensors_b[gid]
            
            with torch.cuda.stream(stream):
                for _ in range(n_bursts):
                    torch.mm(a, b)
        
        # 定期防溢出
        iteration += 1
        if iteration % 500 == 0:
            for gid in streams:
                with torch.cuda.stream(streams[gid]):
                    device = tensors_a[gid].device
                    tensors_a[gid] = torch.randn(matrix_size, matrix_size, device=device)
                    tensors_b[gid] = torch.randn(matrix_size, matrix_size, device=device)
        
        # 等待下一次检测
        time.sleep(poll_interval)
    
    # 清理
    if HAS_PYNVML:
        pynvml.nvmlShutdown()
    print("[adaptive_warmer] Stopped.")


if __name__ == "__main__":
    main()
