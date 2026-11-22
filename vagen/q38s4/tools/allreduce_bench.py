"""q38-s4 (C line, stage 0 item 4): all-reduce bandwidth between machines, run by torchrun on every node.

    python -m torch.distributed.run --nnodes N --nproc-per-node 8 --node-rank R --master-addr A --master-port P \
        vagen/q38s4/tools/allreduce_bench.py --out DIR

Rank 0 writes DIR/allreduce_<world>.json: for each message size the median time of ITERS all-reduces (after WARMUP)
and the bus bandwidth nccl-tests reports (algbw * 2 (n-1) / n, n = world size).  Two groups are measured: the whole
world (all nodes) and, when there is more than one node, each node's own 8 ranks (intra-node, NVLink) for the
comparison.  bf16 tensors; nothing is trained or saved.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist

SIZES_MB = (8, 64, 256, 1024)


def bench(group, size_mb: int, warmup: int, iters: int) -> dict:
    n = dist.get_world_size(group)
    numel = size_mb * 1024 * 1024 // 2
    x = torch.ones(numel, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(x, group=group)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    t = statistics.median(times)
    algbw = size_mb * 1024 * 1024 / t / 1e9
    return {"size_mb": size_mb, "ranks": n, "median_s": t, "algbw_GBps": algbw,
            "busbw_GBps": algbw * 2 * (n - 1) / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    nodes = world // per_node
    hosts = [None] * world
    dist.all_gather_object(hosts, socket.gethostname())
    res = {"world": world, "nodes": nodes, "hosts": sorted(set(hosts)), "torch": torch.__version__,
           "nccl": ".".join(map(str, torch.cuda.nccl.version())), "world_group": [], "intra_node": []}
    for s in SIZES_MB:
        res["world_group"].append(bench(None, s, a.warmup, a.iters))
    if nodes > 1:
        groups = [dist.new_group(list(range(k * per_node, (k + 1) * per_node))) for k in range(nodes)]
        mine = groups[rank // per_node]
        for s in SIZES_MB:
            res["intra_node"].append(bench(mine, s, a.warmup, a.iters))
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"allreduce_{world}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
