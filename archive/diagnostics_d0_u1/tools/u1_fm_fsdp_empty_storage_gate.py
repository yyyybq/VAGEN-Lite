#!/usr/bin/env python3
"""Real two-rank FSDP gate for U1's pending FM second pass.

Runs exactly one fixed transition through:
FSDP forward -> pending-only FSDP forward -> backward -> SGD step ->
sharded parameter checksum/state_dict inspection.  It does not run rollout,
overfit, generation visualization, checkpoint save/reload, or Phase 4.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(ROOT / "verl"))

from tools.u1_fm_backprop_ab_gate import (  # noqa: E402
    DEFAULT_FINAL,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    jdump,
    load_model_processor,
    load_rows,
    prepare_sample,
    seed_everything,
)
from verl.utils.fsdp_utils import get_fsdp_wrap_policy  # noqa: E402


def append_jsonl(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def scalar_all_reduce(value: float, device: torch.device) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def local_parameter_audit(model: FSDP) -> dict[str, Any]:
    count = 0
    numel = 0
    empty = []
    nonfinite = []
    norm_sq = 0.0
    for name, param in model.named_parameters():
        count += 1
        numel += param.numel()
        if param.numel() == 0 or param.untyped_storage().nbytes() == 0:
            empty.append(name)
            continue
        value = param.detach().float()
        if not torch.isfinite(value).all().item():
            nonfinite.append(name)
        norm_sq += float(value.square().sum().item())
    return {
        "tensor_count": count,
        "local_numel": numel,
        "empty_storage": empty,
        "nonfinite": nonfinite,
        "local_l2_norm_sq": norm_sq,
    }


def snapshot_local(model: FSDP) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu().clone() for name, param in model.named_parameters()}


def local_delta_sq(model: FSDP, before: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        old = before[name]
        current = param.detach().cpu()
        if current.shape != old.shape:
            raise RuntimeError(f"local shard shape changed for {name}: {old.shape} -> {current.shape}")
        total += float((current.float() - old.float()).square().sum().item())
    return total


def local_grad_sq(model: FSDP) -> tuple[float, bool, int]:
    total = 0.0
    finite = True
    present = 0
    for param in model.parameters():
        if param.grad is None:
            continue
        present += 1
        grad = param.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total += float(grad.square().sum().item())
    return total, finite, present


def sharded_state_dict_audit(model: FSDP) -> dict[str, Any]:
    cfg = ShardedStateDictConfig(offload_to_cpu=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, cfg):
        state = model.state_dict()
    empty = []
    nonfinite = []
    local_tensor_count = 0
    for name, value in state.items():
        shards = value.local_shards() if hasattr(value, "local_shards") else []
        if shards:
            for shard in shards:
                tensor = shard.tensor
                local_tensor_count += 1
                if tensor.numel() == 0 or tensor.untyped_storage().nbytes() == 0:
                    empty.append(name)
                elif not torch.isfinite(tensor.float()).all().item():
                    nonfinite.append(name)
        elif torch.is_tensor(value):
            local_tensor_count += 1
            if value.numel() == 0 or value.untyped_storage().nbytes() == 0:
                empty.append(name)
            elif not torch.isfinite(value.float()).all().item():
                nonfinite.append(name)
    return {
        "key_count": len(state),
        "local_tensor_count": local_tensor_count,
        "empty_storage": sorted(set(empty)),
        "nonfinite": sorted(set(nonfinite)),
    }


def update_final(path: Path, report: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["phase3_fsdp_empty_storage"] = {
        "status": report["status"],
        "report": "../fm_only/phase3_fsdp_empty_storage_report.json",
        "world_size": report.get("world_size"),
        "fm_loss": report.get("fm_loss"),
        "grad_norm": report.get("grad_norm"),
        "parameter_delta_l2": report.get("parameter_delta_l2"),
        "root_cause": report.get("root_cause"),
        "minimal_fix": report.get("minimal_fix"),
    }
    data["phase3_fm_only"] = {
        "status": "NOT_EXECUTED",
        "reason": "Data, forward-only, BACKPROP A/B, and FSDP one-step gates were executed; overfit, visualization, and save/reload gates remain pending.",
    }
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["next_gate"] = (
        "phase3_fm_fixed_transition_overfit"
        if report["status"] == "PASS"
        else "blocked_on_phase3_fsdp_empty_storage"
    )
    jdump(data, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fixed-transitions", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    args.out.mkdir(parents=True, exist_ok=True)
    timeline = args.out / f"phase3_fsdp_timeline_rank{rank}.jsonl"
    timeline.write_text("", encoding="utf-8")
    report_path = args.out / "phase3_fsdp_empty_storage_report.json"
    started = time.time()
    stage = "init"

    def mark(new_stage: str, **extra: Any) -> None:
        nonlocal stage
        stage = new_stage
        append_jsonl(
            {"rank": rank, "stage": stage, "time": time.time(), "gpu_allocated": torch.cuda.memory_allocated(), **extra},
            timeline,
        )

    base = {
        "phase": "phase3_fsdp_empty_storage",
        "status": "RUNNING",
        "scope": "real 2-rank FSDP one-step; no rollout/overfit/visualization/save-reload/Phase4",
        "world_size": world_size,
        "model": str(args.model),
        "model_substitution": "Original U1 FM initialization; prior step32_merged_hf path is absent and prior FM_BACKPROP was 0.",
        "fsdp": {
            "strategy": "FULL_SHARD",
            "use_orig_params": False,
            "wrap_policy": ["Qwen3DecoderLayer", "NEOVisionModel"],
            "mixed_precision": {"param": "bfloat16", "reduce": "float32", "buffer": "float32"},
        },
        "optimizer": {"class": "torch.optim.SGD", "lr": args.lr},
        "root_cause": (
            "The old dp_actor path unwrapped the root FSDP module before compute_pending_fm_aux_loss(), "
            "bypassing root pre-forward all-gather and exposing resharded zero-byte storage in timestep_embedder."
        ),
        "minimal_fix": (
            "Dispatch the pending-only computation through root FSDP.forward via _u1_compute_pending_only=True; "
            "non-FSDP keeps the direct method call."
        ),
    }
    if rank == 0:
        jdump(base, report_path)

    local_result: dict[str, Any]
    try:
        if world_size < 2:
            raise RuntimeError("FSDP gate requires at least two ranks")
        transitions = args.fixed_transitions or (args.out / "fixed_transitions.jsonl")
        row = load_rows(transitions)[args.sample_index]
        seed_everything(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

        mark("model_load_start")
        model, processor, dtype = load_model_processor(args.model, str(device))
        model.train()
        sample = prepare_sample(processor, row, str(device), dtype)
        mark("model_load_complete")

        wrap_policy = get_fsdp_wrap_policy(
            model,
            {"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer", "NEOVisionModel"]},
        )
        mixed = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        mark("fsdp_wrap_start")
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed,
            sync_module_states=False,
            use_orig_params=False,
            forward_prefetch=False,
        )
        optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=args.lr, momentum=0.0, weight_decay=0.0)
        mark("fsdp_wrap_complete")

        access_before = local_parameter_audit(fsdp_model)
        before = snapshot_local(fsdp_model)
        mark("parameter_access_before", empty_storage=len(access_before["empty_storage"]))

        os.environ["U1_FM_USE_UND_KV"] = "0"
        os.environ["U1_FM_BACKPROP"] = "1"
        optimizer.zero_grad(set_to_none=True)
        seed_everything(args.seed)
        mark("forward_start")
        output = fsdp_model(**sample, labels=None, return_dict=True, use_cache=False)
        pending_after_forward = bool(getattr(fsdp_model.module, "_u1_pending_fm", None))
        mark("forward_complete", pending=pending_after_forward, forward_loss_is_none=output.loss is None)

        mark("pending_loss_start")
        fm_loss = fsdp_model(_u1_compute_pending_only=True)
        pending_after_compute = bool(getattr(fsdp_model.module, "_u1_pending_fm", None))
        if fm_loss is None:
            raise RuntimeError("FSDP pending-only forward returned None")
        mark("pending_loss_complete", loss=float(fm_loss.detach().float().item()))

        mark("backward_start")
        fm_loss.backward()
        grad_sq_local, grad_finite_local, grad_present_local = local_grad_sq(fsdp_model)
        grad_sq = scalar_all_reduce(grad_sq_local, device)
        finite_tensor = torch.tensor([int(grad_finite_local)], dtype=torch.int32, device=device)
        dist.all_reduce(finite_tensor, op=dist.ReduceOp.MIN)
        mark("backward_complete", grad_norm=math.sqrt(grad_sq), grad_present=grad_present_local)

        mark("optimizer_step_start")
        optimizer.step()
        mark("optimizer_step_complete")

        delta_sq = scalar_all_reduce(local_delta_sq(fsdp_model, before), device)
        access_after = local_parameter_audit(fsdp_model)
        mark("checksum_complete", delta_l2=math.sqrt(delta_sq), empty_storage=len(access_after["empty_storage"]))

        mark("state_dict_start")
        state_audit = sharded_state_dict_audit(fsdp_model)
        mark("state_dict_complete", empty_storage=len(state_audit["empty_storage"]))

        local_result = {
            "rank": rank,
            "status": "PASS",
            "last_stage": stage,
            "pending_after_forward": pending_after_forward,
            "pending_after_compute": pending_after_compute,
            "forward_loss_is_none": output.loss is None,
            "fm_loss": float(fm_loss.detach().float().item()),
            "grad_present_local": grad_present_local,
            "grad_finite_global": bool(finite_tensor.item()),
            "grad_norm_global": math.sqrt(grad_sq),
            "parameter_delta_l2_global": math.sqrt(delta_sq),
            "parameter_access_before": access_before,
            "parameter_access_after": access_after,
            "state_dict": state_audit,
            "gpu_memory": {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            },
        }
    except Exception as exc:
        local_result = {
            "rank": rank,
            "status": "FAIL",
            "last_stage": stage,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "gpu_memory": {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            },
        }

    jdump(local_result, args.out / f"phase3_fsdp_rank{rank}.json")
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        all_pass = all(item["status"] == "PASS" for item in gathered)
        first = gathered[0]
        no_empty = all(
            not item.get("parameter_access_before", {}).get("empty_storage")
            and not item.get("parameter_access_after", {}).get("empty_storage")
            and not item.get("state_dict", {}).get("empty_storage")
            for item in gathered
            if item["status"] == "PASS"
        )
        passed = (
            all_pass
            and no_empty
            and first.get("pending_after_forward")
            and not first.get("pending_after_compute")
            and first.get("forward_loss_is_none")
            and first.get("grad_finite_global")
            and float(first.get("grad_norm_global", 0.0)) > 0.0
            and float(first.get("parameter_delta_l2_global", 0.0)) > 0.0
        )
        report = {
            **base,
            "status": "PASS" if passed else "FAIL",
            "ranks": gathered,
            "fm_loss": first.get("fm_loss"),
            "grad_norm": first.get("grad_norm_global"),
            "parameter_delta_l2": first.get("parameter_delta_l2_global"),
            "empty_storage_unresolved": not no_empty,
            "elapsed_seconds": time.time() - started,
            "overall_status": "NOT_READY_FOR_TRAINING",
            "next_gate": (
                "phase3_fm_fixed_transition_overfit"
                if passed
                else "blocked_on_phase3_fsdp_empty_storage"
            ),
        }
        jdump(report, report_path)
        update_final(args.final_delivery, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    dist.barrier()
    exit_code = 0 if all(item["status"] == "PASS" for item in gathered) else 2
    dist.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
