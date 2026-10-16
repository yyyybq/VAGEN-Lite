#!/usr/bin/env python3
"""
GPU Warmer with CUDA Streams - 可以与训练进程更好地并行

使用方法:
    python gpu_warmer_streams.py &

特点:
- 使用独立的CUDA stream，不完全阻塞主训练
- 小矩阵+高频率，在训练空闲时快速填充
- 自动检测可用GPU
"""
import torch
import time
import os
import signal
import sys

def signal_handler(sig, frame):
    print("\nGPU warmer stopped.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    # 支持 WARMER_GPU 环境变量，指定单个 GPU；如果不设置则使用所有 GPU
    warmer_gpu = os.environ.get('WARMER_GPU', None)
    
    if warmer_gpu is not None:
        gpu_ids = [int(warmer_gpu)]
        print(f"GPU Warmer starting on GPU {warmer_gpu} only...")
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPUs available")
            return
        gpu_ids = list(range(num_gpus))
        print(f"GPU Warmer starting on {num_gpus} GPUs...")
    
    # 为每个GPU创建独立的stream和tensor
    streams = []
    tensors = []
    
    # 使用较小的矩阵，减少内存占用
    matrix_size = int(os.environ.get('WARMER_SIZE', '1536'))  # ~9MB per tensor
    
    for i in gpu_ids:
        try:
            device = torch.device(f'cuda:{i}')
            # 创建非默认stream
            stream = torch.cuda.Stream(device=device)
            x = torch.randn(matrix_size, matrix_size, device=device)
            streams.append(stream)
            tensors.append(x)
            print(f"  GPU {i}: stream created, tensor {matrix_size}x{matrix_size}")
        except Exception as e:
            print(f"  GPU {i}: failed - {e}")
    
    if not tensors:
        print("No tensors created, exiting")
        return
    
    print(f"Running warmers on {len(tensors)} GPUs (Ctrl+C to stop)...")
    
    sleep_ms = float(os.environ.get('WARMER_SLEEP_MS', '5')) / 1000.0
    
    iteration = 0
    while True:
        for i, (stream, tensor) in enumerate(zip(streams, tensors)):
            # 在独立stream上执行计算
            with torch.cuda.stream(stream):
                tensors[i] = torch.mm(tensor, tensor)
                
                # 防止数值溢出
                if iteration % 100 == 0:
                    if torch.any(torch.isinf(tensors[i])) or torch.any(torch.isnan(tensors[i])):
                        tensors[i] = torch.randn(matrix_size, matrix_size, device=tensor.device)
        
        iteration += 1
        
        # 短暂sleep让训练进程有机会运行
        if sleep_ms > 0:
            time.sleep(sleep_ms)

if __name__ == "__main__":
    main()
