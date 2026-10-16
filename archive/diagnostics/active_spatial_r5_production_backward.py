#!/usr/bin/env python3
"""Optimizer-free, production-forward backward diagnosis for the real r5 batch.

Run this with torchrun using the checkpoint's original FSDP world size (four).
It deliberately instantiates the production actor and critic workers, reuses their
private micro-batch forward methods and the production PPO loss functions, and
never calls an update method, optimizer.step, scheduler.step, or checkpoint save.

The diagnostic objective is the token-mean loss over the immutable complete
snapshot.  This is a paired fixed-experience intervention; it is not a simulated
sequence of PPO minibatch optimizer updates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from omegaconf import OmegaConf, open_dict

from vagen.utils.active_spatial_ppo_replay import IGNORE_VALUE, recompute_branches
from vagen.utils.active_spatial_ppo_snapshot import SnapshotValidationError, load_snapshot


BRANCHES = ("S0", "S1", "S5")
REPORT_SCHEMA = "active_spatial_r5_production_backward_v1"


def _rank() -> int:
    return dist.get_rank()


def _world() -> int:
    return dist.get_world_size()


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.to_local() if hasattr(value, "to_local") else value


def _recursive_to(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _recursive_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_recursive_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_recursive_to(item, device) for item in value)
    return value


def _micro_batch(payload: Mapping[str, Any], row: int, device: torch.device) -> dict[str, Any]:
    tensors = payload["tensor_batch"]
    result = {
        key: tensors[key][row : row + 1].to(device=device, non_blocking=True)
        for key in ("input_ids", "responses", "attention_mask", "response_mask", "position_ids")
    }
    result["multi_modal_inputs"] = [_recursive_to(payload["multi_modal_inputs"][row], device)]
    return result


def _module_key(name: str, parameter: torch.nn.Parameter) -> str:
    fqns = list(getattr(parameter, "_fqns", ()) or ())
    candidates = fqns if fqns else [name]
    groups: set[str] = set()
    for candidate in candidates:
        text = str(candidate).replace("_fsdp_wrapped_module.", "")
        vision = re.search(r"(?:visual|vision_model|vision_tower)\.blocks?\.(\d+)", text)
        language = re.search(r"(?:language_model\.)?(?:model\.)?layers\.(\d+)", text)
        if vision:
            groups.add(f"vision.block.{int(vision.group(1)):02d}")
        elif language:
            groups.add(f"language.layer.{int(language.group(1)):02d}")
        elif "merger" in text or "projector" in text or "mm_projector" in text:
            groups.add("vision.projector")
        elif "patch_embed" in text:
            groups.add("vision.patch_embed")
        elif "lm_head" in text:
            groups.add("language.lm_head")
        elif "embed_tokens" in text:
            groups.add("language.embed_tokens")
        elif "v_head" in text or "value_head" in text or ".score" in text:
            groups.add("critic.value_head")
        elif "norm" in text:
            groups.add("other.norm")
        else:
            groups.add("other")
    return next(iter(groups)) if len(groups) == 1 else "mixed_flat:" + "+".join(sorted(groups))


def _all_gather_dict(local: Mapping[str, Mapping[str, float | int]]) -> dict[str, dict[str, float | int]]:
    gathered: list[dict[str, dict[str, float | int]] | None] = [None] * _world()
    dist.all_gather_object(gathered, dict(local))
    merged: dict[str, dict[str, float | int]] = defaultdict(lambda: {"grad_norm_sq": 0.0, "numel": 0, "params": 0})
    for item in gathered:
        assert item is not None
        for group, values in item.items():
            merged[group]["grad_norm_sq"] += float(values["grad_norm_sq"])
            merged[group]["numel"] += int(values["numel"])
            merged[group]["params"] += int(values["params"])
    return {
        key: {
            "status": "TRAINABLE",
            "grad_norm": math.sqrt(max(0.0, float(values["grad_norm_sq"]))),
            "local_shard_numel_sum": int(values["numel"]),
            "local_parameter_shards": int(values["params"]),
        }
        for key, values in sorted(merged.items())
    }


def _gradient_report(model: torch.nn.Module) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    groups: dict[str, dict[str, float | int]] = defaultdict(lambda: {"grad_norm_sq": 0.0, "numel": 0, "params": 0})
    frozen: set[str] = set()
    saved: dict[str, torch.Tensor] = {}
    total_sq = 0.0
    missing = 0
    for name, parameter in model.named_parameters():
        group = _module_key(name, parameter)
        if not parameter.requires_grad:
            frozen.add(group)
            continue
        groups[group]["numel"] += int(parameter.numel())
        groups[group]["params"] += 1
        if parameter.grad is None:
            missing += 1
            continue
        gradient = _local_tensor(parameter.grad.detach()).float()
        square = float(torch.sum(gradient * gradient).cpu())
        groups[group]["grad_norm_sq"] += square
        total_sq += square
        saved[name] = gradient.cpu().clone()
    total = torch.tensor([total_sq, float(missing)], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    report: dict[str, Any] = {
        "total_grad_norm": math.sqrt(max(0.0, float(total[0].item()))),
        "trainable_parameter_shards_without_grad": int(total[1].item()),
        "modules": _all_gather_dict(groups),
    }
    all_frozen: list[list[str] | None] = [None] * _world()
    dist.all_gather_object(all_frozen, sorted(frozen))
    for group in sorted({group for items in all_frozen if items for group in items}):
        report["modules"].setdefault(group, {"status": "FROZEN", "grad_norm": None})
    return report, saved


def _compare_gradients(model: torch.nn.Module, reference: Mapping[str, torch.Tensor]) -> dict[str, float | None]:
    dot = left_sq = right_sq = diff_sq = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        current = (
            _local_tensor(parameter.grad.detach()).float().cpu()
            if parameter.grad is not None
            else torch.zeros_like(reference[name])
        )
        previous = reference.get(name)
        if previous is None:
            raise SnapshotValidationError(f"S0 gradient reference missing parameter shard: {name}")
        dot += float(torch.sum(previous * current))
        left_sq += float(torch.sum(previous * previous))
        right_sq += float(torch.sum(current * current))
        delta = current - previous
        diff_sq += float(torch.sum(delta * delta))
    values = torch.tensor([dot, left_sq, right_sq, diff_sq], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    dot, left_sq, right_sq, diff_sq = map(float, values.cpu().tolist())
    denominator = math.sqrt(left_sq * right_sq)
    return {
        "cosine": dot / denominator if denominator else None,
        "difference_norm": math.sqrt(max(0.0, diff_sq)),
        "s0_norm": math.sqrt(max(0.0, left_sq)),
        "s1_norm": math.sqrt(max(0.0, right_sq)),
    }


def _parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        tensor = _local_tensor(parameter.detach()).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    local = digest.hexdigest()
    values: list[str | None] = [None] * _world()
    dist.all_gather_object(values, local)
    return hashlib.sha256("|".join(str(item) for item in values).encode("ascii")).hexdigest()


def _clear_and_verify(model: torch.nn.Module) -> bool:
    model.zero_grad(set_to_none=True)
    return all(parameter.grad is None for parameter in model.parameters())


def _reduce_metrics(local: Mapping[str, float]) -> dict[str, float]:
    keys = sorted(local)
    tensor = torch.tensor([float(local[key]) for key in keys], dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist(), strict=True)}


class _ErrorAccumulator:
    def __init__(self) -> None:
        self.sum_abs = 0.0
        self.count = 0
        self.maximum = 0.0

    def add(self, actual: torch.Tensor, expected: torch.Tensor, mask: torch.Tensor) -> None:
        selected = (actual.detach().float() - expected.detach().float()).abs()[mask.bool()]
        if selected.numel():
            self.sum_abs += float(selected.sum().cpu())
            self.count += int(selected.numel())
            self.maximum = max(self.maximum, float(selected.max().cpu()))

    def distributed(self) -> dict[str, float | int]:
        summed = torch.tensor([self.sum_abs, float(self.count)], dtype=torch.float64, device=torch.cuda.current_device())
        maximum = torch.tensor([self.maximum], dtype=torch.float64, device=torch.cuda.current_device())
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        count = int(summed[1].item())
        return {"count": count, "mean_abs_error": float(summed[0].item()) / max(1, count), "max_abs_error": float(maximum.item())}


def _forbid_optimizer_step(optimizer: torch.optim.Optimizer, label: str, calls: dict[str, int]) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        calls[label] += 1
        raise RuntimeError(f"forbidden optimizer.step reached in {label} diagnostic")

    optimizer.step = forbidden  # type: ignore[method-assign]


def _actor_diagnosis(payload: Mapping[str, Any], config: Mapping[str, Any], replay: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss_vanilla, kl_penalty
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    actor_cfg = OmegaConf.create(config["actor_rollout_ref"])
    worker = ActorRolloutRefWorker(actor_cfg, role="actor")
    worker.init_model()
    load_fsdp_model_to_gpu(worker.actor_module_fsdp)
    worker.actor_module_fsdp.train()
    step_calls = {"actor": 0}
    _forbid_optimizer_step(worker.actor_optimizer, "actor", step_calls)
    before = _parameter_digest(worker.actor_module_fsdp)

    tensors = payload["tensor_batch"]
    indices = list(range(_rank(), int(tensors["input_ids"].shape[0]), _world()))
    total_tokens = int(tensors["response_mask"].sum().item())
    parity = _ErrorAccumulator()
    reports: dict[str, Any] = {}
    s0_gradients: dict[str, torch.Tensor] | None = None
    zero_grad_checks: list[bool] = []
    device = torch.device("cuda", torch.cuda.current_device())
    entropy_coeff = float(worker.actor.config.entropy_coeff)
    kl_coeff = float(worker.actor.config.kl_loss_coef)
    loss_mode = str(worker.actor.config.loss_agg_mode)

    for branch_name in BRANCHES:
        zero_grad_checks.append(_clear_and_verify(worker.actor_module_fsdp))
        local_metrics = {"policy_loss": 0.0, "entropy": 0.0, "explicit_kl_loss": 0.0, "actor_total_loss": 0.0}
        advantages = replay["branches"][branch_name]["independent_advantages"]
        for row in indices:
            micro = _micro_batch(payload, row, device)
            entropy, log_prob, aux_loss = worker.actor._forward_micro_batch(
                micro, temperature=float(args.logprob_temperature), calculate_entropy=True, return_aux_loss=True
            )
            if aux_loss is not None:
                raise SnapshotValidationError("r5 actor unexpectedly produced an auxiliary loss")
            mask = tensors["response_mask"][row : row + 1].to(device=device)
            old = tensors["old_log_probs"][row : row + 1].to(device=device)
            ref = tensors["ref_log_prob"][row : row + 1].to(device=device)
            advantage = advantages[row : row + 1].to(device=device, dtype=log_prob.dtype)
            pg_loss, _ = compute_policy_loss_vanilla(
                old_log_prob=old,
                log_prob=log_prob,
                advantages=advantage,
                response_mask=mask,
                loss_agg_mode=loss_mode,
                config=worker.actor.config,
            )
            entropy_loss = agg_loss(entropy, mask, loss_mode)
            kl_loss = agg_loss(kl_penalty(log_prob, ref, worker.actor.config.kl_loss_type), mask, loss_mode)
            weight = float(mask.sum().item()) / float(total_tokens)
            total_loss = (pg_loss - entropy_coeff * entropy_loss + kl_coeff * kl_loss) * weight
            # FSDP averages reduced gradients across ranks.  Undo that average so
            # rank-sharded rows yield the complete-batch token-mean objective.
            (total_loss * _world()).backward()
            local_metrics["policy_loss"] += float(pg_loss.detach()) * weight
            local_metrics["entropy"] += float(entropy_loss.detach()) * weight
            local_metrics["explicit_kl_loss"] += float(kl_loss.detach()) * weight
            local_metrics["actor_total_loss"] += float(total_loss.detach())
            if branch_name == "S0":
                parity.add(log_prob, old, mask)
            del micro, entropy, log_prob, pg_loss, entropy_loss, kl_loss, total_loss
        metrics = _reduce_metrics(local_metrics)
        gradients, saved = _gradient_report(worker.actor_module_fsdp)
        reports[branch_name] = {**metrics, "gradients": gradients}
        if branch_name == "S0":
            s0_gradients = saved
        elif branch_name == "S1":
            assert s0_gradients is not None
            reports["S1_vs_S0"] = _compare_gradients(worker.actor_module_fsdp, s0_gradients)
            del s0_gradients
        del saved
        gc.collect()

    actor_parity = parity.distributed()
    after = _parameter_digest(worker.actor_module_fsdp)
    result = {
        "checkpoint_identity": config["actor_rollout_ref"]["model"]["path"],
        "parameter_digest_before": before,
        "parameter_digest_after": after,
        "parameters_unchanged": before == after,
        "optimizer_step_calls": step_calls["actor"],
        "independent_zero_grad_checks": zero_grad_checks,
        "old_logprob_parity": actor_parity,
        "branches": reports,
    }
    result["parity_pass"] = bool(
        actor_parity["max_abs_error"] <= args.actor_parity_max_atol
        and actor_parity["mean_abs_error"] <= args.actor_parity_mean_atol
    )
    _clear_and_verify(worker.actor_module_fsdp)
    del worker
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def _critic_diagnosis(payload: Mapping[str, Any], config: Mapping[str, Any], replay: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    from verl.trainer.ppo.core_algos import compute_value_loss
    from verl.utils.fsdp_utils import load_fsdp_model_to_gpu
    from verl.workers.fsdp_workers import CriticWorker

    critic_cfg = OmegaConf.create(config["critic"])
    with open_dict(critic_cfg):
        critic_cfg.nccl_timeout = int(config.get("actor_rollout_ref", {}).get("nccl_timeout", 600))
        critic_cfg.checkpoint.load_contents = ["model"]
        critic_cfg.checkpoint.save_contents = []
    worker = CriticWorker(critic_cfg)
    worker.init_model()
    checkpoint_path = str(args.critic_checkpoint)
    worker.load_checkpoint(checkpoint_path, del_local_after_load=False)
    load_fsdp_model_to_gpu(worker.critic_module)
    worker.critic_module.train()
    step_calls = {"critic": 0}
    _forbid_optimizer_step(worker.critic_optimizer, "critic", step_calls)
    before = _parameter_digest(worker.critic_module)

    tensors = payload["tensor_batch"]
    indices = list(range(_rank(), int(tensors["input_ids"].shape[0]), _world()))
    value_mask_full = tensors["value_mask"].bool() & tensors["response_mask"].bool()
    total_targets_by_branch = {
        name: int((value_mask_full & ((replay["branches"][name]["returns"].float() - IGNORE_VALUE).abs() >= 1e-6)).sum().item())
        for name in BRANCHES
    }
    parity = _ErrorAccumulator()
    reports: dict[str, Any] = {}
    s0_gradients: dict[str, torch.Tensor] | None = None
    zero_grad_checks: list[bool] = []
    device = torch.device("cuda", torch.cuda.current_device())
    loss_mode = str(worker.critic.config.loss_agg_mode)
    cliprange = float(worker.critic.config.cliprange_value)

    for branch_name in BRANCHES:
        zero_grad_checks.append(_clear_and_verify(worker.critic_module))
        local_metrics = {"value_loss": 0.0}
        returns_full = replay["branches"][branch_name]["returns"]
        total_targets = total_targets_by_branch[branch_name]
        if total_targets <= 0:
            raise SnapshotValidationError(f"{branch_name} has no critic targets")
        for row in indices:
            micro = _micro_batch(payload, row, device)
            vpred = worker.critic._forward_micro_batch(micro)
            old_values = tensors["values"][row : row + 1].to(device=device, dtype=vpred.dtype)
            returns = returns_full[row : row + 1].to(device=device, dtype=vpred.dtype)
            raw_mask = tensors["response_mask"][row : row + 1].to(device=device).bool()
            value_mask = tensors["value_mask"][row : row + 1].to(device=device).bool()
            valid = raw_mask & value_mask & ((returns.float() - IGNORE_VALUE).abs() >= 1e-6)
            if branch_name == "S0":
                parity.add(vpred, old_values, raw_mask)
            if bool(valid.any()):
                vf_loss, _ = compute_value_loss(
                    vpreds=vpred,
                    values=old_values,
                    returns=returns,
                    response_mask=valid.to(dtype=tensors["response_mask"].dtype),
                    cliprange_value=cliprange,
                    loss_agg_mode=loss_mode,
                )
                weight = float(valid.sum().item()) / float(total_targets)
                weighted = vf_loss * weight
                (weighted * _world()).backward()
                local_metrics["value_loss"] += float(vf_loss.detach()) * weight
                del vf_loss, weighted
            del micro, vpred, old_values, returns, raw_mask, value_mask, valid
        metrics = _reduce_metrics(local_metrics)
        gradients, saved = _gradient_report(worker.critic_module)
        reports[branch_name] = {**metrics, "gradients": gradients}
        if branch_name == "S0":
            s0_gradients = saved
        elif branch_name == "S1":
            assert s0_gradients is not None
            reports["S1_vs_S0"] = _compare_gradients(worker.critic_module, s0_gradients)
            del s0_gradients
        del saved
        gc.collect()

    critic_parity = parity.distributed()
    after = _parameter_digest(worker.critic_module)
    result = {
        "checkpoint_identity": checkpoint_path,
        "parameter_digest_before": before,
        "parameter_digest_after": after,
        "parameters_unchanged": before == after,
        "optimizer_step_calls": step_calls["critic"],
        "independent_zero_grad_checks": zero_grad_checks,
        "saved_value_parity": critic_parity,
        "branches": reports,
    }
    result["parity_pass"] = bool(
        critic_parity["max_abs_error"] <= args.critic_parity_max_atol
        and critic_parity["mean_abs_error"] <= args.critic_parity_mean_atol
    )
    _clear_and_verify(worker.critic_module)
    del worker
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    return result


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--critic-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--logprob-temperature", type=float, default=1.0)
    parser.add_argument("--actor-parity-max-atol", type=float, default=0.05)
    parser.add_argument("--actor-parity-mean-atol", type=float, default=0.005)
    parser.add_argument("--critic-parity-max-atol", type=float, default=0.05)
    parser.add_argument("--critic-parity-mean-atol", type=float, default=0.005)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SnapshotValidationError("production backward requires CUDA")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if _world() != args.expected_world_size:
        raise SnapshotValidationError(f"expected FSDP world size {args.expected_world_size}, got {_world()}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    started = time.time()
    payload, manifest, config = load_snapshot(args.snapshot)
    replay = recompute_branches(payload, config)
    historical_error = float((replay["historical"]["advantages"] - payload["tensor_batch"]["advantages"]).abs().max())
    actor = _actor_diagnosis(payload, config, replay, args)
    critic = _critic_diagnosis(payload, config, replay, args)
    passed = bool(
        historical_error <= 1e-6
        and actor["parity_pass"]
        and critic["parity_pass"]
        and actor["parameters_unchanged"]
        and critic["parameters_unchanged"]
        and actor["optimizer_step_calls"] == 0
        and critic["optimizer_step_calls"] == 0
        and all(actor["independent_zero_grad_checks"])
        and all(critic["independent_zero_grad_checks"])
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS" if passed else "BLOCKED",
        "evidence_scope": "REAL_R5_FIXED_BATCH_OPTIMIZER_FREE_BACKWARD",
        "claim_limit": "Fixed-experience update-signal difference only; not evidence of spatial capability improvement.",
        "objective": "complete-snapshot token-mean; production forward and production PPO loss operators",
        "snapshot_manifest_sha256": hashlib.sha256(Path(args.snapshot, "manifest.json").read_bytes()).hexdigest(),
        "snapshot_payload_sha256": manifest["payload_sha256"],
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "historical_no_concat_gae_max_abs_error": historical_error,
        "host": socket.gethostname(),
        "world_size": _world(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "elapsed_seconds": time.time() - started,
        "safety": {
            "optimizer_step_called": False,
            "checkpoint_written": False,
            "parameter_persistence": False,
            "actor_optimizer_step_guard_calls": actor["optimizer_step_calls"],
            "critic_optimizer_step_guard_calls": critic["optimizer_step_calls"],
        },
        "actor": actor,
        "critic": critic,
    }
    if _rank() == 0:
        _write_json_atomic(Path(args.output), report)
    dist.barrier()
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SnapshotValidationError as error:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2)
