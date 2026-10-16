#!/usr/bin/env python3
"""Two-rank CUDA regression test for mixed BF16/FP32 FSDP1 grad clipping."""
from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.fsdp_utils import (
    align_optimizer_grad_dtype_device_,
    fsdp1_clip_grad_norm_,
    repair_unreduced_fsdp1_grad_shards_,
)


class MixedDtypeModel(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.bf16 = nn.Linear(16, 16, device=device, dtype=torch.bfloat16)
        self.fp32 = nn.Linear(16, 16, device=device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bf16(x.to(torch.bfloat16)).float() + self.fp32(x.float())


def global_sharded_grad_norm(module: FSDP) -> torch.Tensor:
    local_sq = torch.zeros((), device=torch.cuda.current_device(), dtype=torch.float32)
    for param in module.parameters():
        if param.grad is not None:
            local_sq += param.grad.detach().float().square().sum()
    dist.all_reduce(local_sq)
    return local_sq.sqrt()


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20260830)

    raw = MixedDtypeModel(device)
    raw.bf16 = FSDP(raw.bf16, device_id=device)
    raw.fp32 = FSDP(raw.fp32, device_id=device)
    model = FSDP(raw, device_id=device)
    x = torch.randn(8, 16, device=device)
    model(x).square().mean().backward()
    dtypes = sorted({str(param.grad.dtype) for param in model.parameters() if param.grad is not None})
    assert dtypes == ["torch.bfloat16", "torch.float32"], dtypes

    public_error = None
    try:
        model.clip_grad_norm_(0.25)
    except ValueError as exc:
        public_error = str(exc)
    assert public_error and "Requires uniform dtype" in public_error, public_error
    dist.barrier()

    before = global_sharded_grad_norm(model)
    returned = fsdp1_clip_grad_norm_(model, 0.25)
    after = global_sharded_grad_norm(model)
    assert torch.isfinite(returned), returned
    assert torch.allclose(returned.float(), before.float(), rtol=3e-3, atol=3e-3), (returned, before)
    assert after <= 0.2505, after

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    fp32_param = next(param for param in model.parameters() if param.grad is not None and param.dtype == torch.float32)
    local_numel = fp32_param.numel()
    synthetic_full_grad = torch.arange(
        local_numel * dist.get_world_size(), device=device, dtype=torch.bfloat16
    ) + rank
    expected_full_grad = synthetic_full_grad.float()
    expected_shard = torch.empty(local_numel, device=device, dtype=torch.float32)
    dist.reduce_scatter_tensor(expected_shard, expected_full_grad)
    expected_shard.div_(dist.get_world_size())
    expected_shard = expected_shard.to(fp32_param.dtype)
    fp32_param.grad.data = synthetic_full_grad
    fsdp_grad_repair = repair_unreduced_fsdp1_grad_shards_(model)
    assert fsdp_grad_repair["repaired"] == 1, fsdp_grad_repair
    assert fp32_param.grad.shape == fp32_param.shape
    assert torch.equal(fp32_param.grad, expected_shard), (fp32_param.grad, expected_shard)

    fp32_param.grad.data = fp32_param.grad.data.to(torch.bfloat16)
    optimizer_error = None
    try:
        optimizer.step()
    except RuntimeError as exc:
        optimizer_error = str(exc)
    assert optimizer_error is not None, "foreach AdamW unexpectedly accepted a mismatched param/grad dtype"
    alignment = align_optimizer_grad_dtype_device_(optimizer)
    assert alignment["aligned"] == 1, alignment
    assert fp32_param.grad.dtype == fp32_param.dtype
    optimizer.step()
    optimizer_step_after_alignment = True

    if rank == 0:
        print(json.dumps({
            "status": "PASS",
            "world_size": dist.get_world_size(),
            "gradient_dtypes": dtypes,
            "public_fsdp_error_reproduced": True,
            "returned_preclip_norm": float(returned),
            "independent_preclip_norm": float(before),
            "postclip_norm": float(after),
            "max_norm": 0.25,
            "optimizer_mismatch_error_reproduced": optimizer_error,
            "unreduced_fsdp_grad_repair": fsdp_grad_repair,
            "optimizer_grad_alignment": alignment,
            "optimizer_step_after_alignment": optimizer_step_after_alignment,
        }, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
