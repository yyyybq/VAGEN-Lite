"""Lightweight GPU warmer to keep utilization above 60%.
Runs matrix multiplications on a specified GPU with controlled memory usage,
leaving room for the actual renderer workload."""

import torch
import time
import os
import signal
import sys

def main():
    gpu_id = int(os.environ.get("WARMER_GPU", 4))
    # Use ~2GB VRAM (2048x2048 float32 = 16MB per tensor, but matmul needs workspace)
    size = int(os.environ.get("WARMER_SIZE", 3072))
    # Sleep interval (seconds) between bursts - yield to renderer
    sleep_ms = float(os.environ.get("WARMER_SLEEP_MS", 5))

    dev = torch.device(f"cuda:{gpu_id}")
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)
    print(f"[gpu_warmer] Running on GPU {gpu_id}, matrix {size}x{size}, sleep {sleep_ms}ms")

    # Graceful shutdown
    def _exit(sig, frame):
        print("[gpu_warmer] Shutting down.")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)

    while True:
        # Burst of matmuls to keep utilization up
        for _ in range(20):
            c = torch.mm(a, b)
        # Tiny sleep to let renderer get GPU cycles when it needs them
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        # Prevent overflow
        if torch.any(torch.isinf(c)):
            a = torch.randn(size, size, device=dev)
            b = torch.randn(size, size, device=dev)

if __name__ == "__main__":
    main()
