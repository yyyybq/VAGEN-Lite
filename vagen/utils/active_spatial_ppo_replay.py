"""Offline no-concat PPO reward/GAE replay with an optional gradient adapter.

The module never owns an optimizer and contains no ``optimizer.step`` call.  A
caller supplies lightweight actor/critic forward functions for a fixed snapshot;
the production trainer is not imported or mutated by this diagnostic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .active_spatial_ppo_snapshot import SnapshotValidationError


IGNORE_VALUE = -100.0


def _to_numpy_int64(value: Any, *, factorize_if_non_numeric: bool) -> np.ndarray:
    """Match production no-concat GAE identity normalization exactly."""
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.int64, copy=False)
    if array.dtype == np.object_ or np.issubdtype(array.dtype, np.str_):
        try:
            return array.astype(np.int64)
        except (ValueError, TypeError):
            if not factorize_if_non_numeric:
                raise SnapshotValidationError(
                    f"cannot cast trajectory/turn identity to int64: {array.reshape(-1)[:5].tolist()}"
                )
            _, inverse = np.unique(array, return_inverse=True)
            return inverse.astype(np.int64, copy=False)
    return array.astype(np.int64, copy=False)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def _masked_whiten(values: torch.Tensor, mask: torch.Tensor, reference: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Numerically mirrors ``verl.utils.torch_functional.masked_whiten``."""
    if reference is None:
        mean = _masked_mean(values, mask)
        variance = _masked_mean((values - mean) ** 2, mask)
        count = mask.sum()
        if int(count.item()) <= 1:
            raise SnapshotValidationError("historical masked_whiten requires at least two valid tokens")
        variance = variance * count / (count - 1)
    else:
        mean, variance = reference
    return (values - mean) * torch.rsqrt(variance + 1e-8), (mean, variance)


def no_concat_gae(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    group_idx: list[Any],
    traj_idx: list[Any],
    turn_idx: list[Any],
    *,
    gamma: float,
    lam: float,
    whiten_reference: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    """Exact historical no-concat GAE: each trajectory end starts with next_value=0."""
    device = token_level_rewards.device
    mask_f_full = response_mask.to(dtype=token_level_rewards.dtype, device=device)
    mask_b_full = response_mask.bool().to(device=device)
    # Do not stack raw string group IDs with numeric trajectory/turn IDs: numpy
    # would promote every column to string and sort turns lexicographically
    # (0, 1, 10, 11, 2, ...).  Production factorizes only group IDs and keeps
    # trajectory/turn indices numeric.
    keys = np.stack(
        [
            _to_numpy_int64(group_idx, factorize_if_non_numeric=True),
            _to_numpy_int64(traj_idx, factorize_if_non_numeric=False),
            _to_numpy_int64(turn_idx, factorize_if_non_numeric=False),
        ],
        axis=1,
    )
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    first_t = torch.as_tensor(first.astype(np.int64), device=device)
    scores, vals = token_level_rewards.index_select(0, first_t), values.index_select(0, first_t)
    mask_f, mask_b = mask_f_full.index_select(0, first_t), mask_b_full.index_select(0, first_t)
    unique_keys = keys[first]
    count, width = scores.shape
    arange = torch.arange(width, device=device).view(1, width).expand(count, width)
    first_pos = torch.where(mask_b, arange, torch.full_like(arange, width)).min(dim=1).values
    first_pos = torch.where(mask_b.any(dim=1), first_pos, torch.full_like(first_pos, -1))
    turn_rewards = (scores * mask_f).sum(dim=1)
    gathered = vals.gather(1, first_pos.clamp(min=0).view(count, 1)).squeeze(1)
    turn_values = torch.where(first_pos >= 0, gathered, torch.zeros_like(gathered))
    raw_u = torch.zeros_like(vals)
    returns_u = torch.full_like(vals, IGNORE_VALUE)
    raw_turn_u = torch.zeros_like(turn_values)
    deltas_u = torch.zeros_like(turn_values)
    by_traj: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for row, (group, traj, _turn) in enumerate(unique_keys.tolist()):
        by_traj[(group, traj)].append(row)
    for rows in by_traj.values():
        rows.sort(key=lambda row: unique_keys[row, 2])
        next_value, lastgaelam = 0.0, 0.0
        for row in reversed(rows):
            reward, value = float(turn_rewards[row]), float(turn_values[row])
            delta = reward + float(gamma) * next_value - value
            lastgaelam = delta + float(gamma) * float(lam) * lastgaelam
            deltas_u[row] = delta
            raw_turn_u[row] = lastgaelam
            pos = int(first_pos[row])
            if pos >= 0:
                raw_u[row, mask_b[row]] = lastgaelam
                returns_u[row, pos] = lastgaelam + value
            next_value = value
    whitened_u, stats = _masked_whiten(raw_u, mask_f, reference=whiten_reference)
    inverse_t = torch.as_tensor(inverse, dtype=torch.long, device=device)
    return {
        "raw_advantages": raw_u.index_select(0, inverse_t),
        "advantages": whitened_u.index_select(0, inverse_t),
        "returns": returns_u.index_select(0, inverse_t),
        "whitening_stats": stats,
        "turn_rewards": turn_rewards,
        "turn_values": turn_values,
        "turn_deltas": deltas_u,
        "raw_turn_advantages": raw_turn_u,
        "unique_keys": torch.as_tensor(unique_keys.astype(np.int64), dtype=torch.long),
        "unique_first_indices": first_t.detach().cpu(),
        "inverse_indices": inverse_t.detach().cpu(),
    }


def _term(trace: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return (trace.get("components") or {}).get(name) or {}


def _applied(trace: Mapping[str, Any], name: str) -> float:
    return float(_term(trace, name).get("applied", 0.0) or 0.0)


def _potential(trace: Mapping[str, Any], gamma: float | None) -> float:
    term = _term(trace, "potential")
    if gamma is None:
        return float(term.get("applied", 0.0) or 0.0)
    # ``events=0`` is an explicit reward-trace fact: the environment skipped
    # potential shaping (for example at a primitive-limit terminal turn).  It
    # is not a missing scale to be inferred from phi, so every ablation keeps
    # its applied value at zero.
    if term.get("events") == 0:
        return 0.0
    score = trace.get("score") or {}
    phi, phi_prev = score.get("phi"), score.get("phi_prev")
    scale = term.get("scale")
    if phi is None or phi_prev is None or scale is None:
        raise SnapshotValidationError("cannot retarget potential gamma without phi_prev, phi, and potential scale")
    return float((float(gamma) * float(phi) - float(phi_prev)) * float(scale))


def reward_from_trace(trace: Mapping[str, Any], variant: str, *, ppo_gamma: float) -> float:
    """Recompute only reward composition; scorer/success/termination are untouched."""
    if not trace or trace.get("schema_version") != "active_spatial_reward_trace_v1":
        raise SnapshotValidationError("missing active_spatial_reward_trace_v1")
    if variant == "historical":
        total = (trace.get("accounting") or {}).get("actual_total_reward")
        if total is None:
            raise SnapshotValidationError("reward trace has no actual_total_reward")
        return float(total)
    names = (trace.get("components") or {}).keys()
    total = math.fsum(_applied(trace, name) for name in names if name not in {"potential", "near", "visibility"})
    if variant == "S0":
        return float(total)
    if variant == "S1":
        return float(total + _potential(trace, ppo_gamma))
    if variant == "S5":
        return float(total + _potential(trace, 0.99) + _applied(trace, "near") + _applied(trace, "visibility"))
    raise SnapshotValidationError(f"unknown reward branch: {variant}")


def _as_float(mapping: Mapping[str, Any], dotted: str, default: float) -> float:
    current: Any = mapping
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return float(default)
        current = current[part]
    return float(current)


def recompute_branches(payload: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    tensors, metadata = payload["tensor_batch"], payload["metadata"]
    gamma = _as_float(resolved_config, "algorithm.gamma", 1.0)
    lam = _as_float(resolved_config, "algorithm.lam", 1.0)
    historical = no_concat_gae(
        tensors["token_level_scores"], tensors["values"], tensors["response_mask"],
        metadata["group_idx"], metadata["traj_idx"], metadata["turn_idx"], gamma=gamma, lam=lam,
    )
    result: dict[str, Any] = {"historical": historical, "branches": {}}
    reference = historical["whitening_stats"]
    for variant in ("S0", "S1", "S5"):
        rewards = torch.zeros_like(tensors["token_level_scores"])
        totals: list[float] = []
        for row, trace in enumerate(metadata["reward_trace"]):
            total = reward_from_trace(trace, variant, ppo_gamma=gamma)
            totals.append(total)
            positions = torch.nonzero(tensors["response_mask"][row].bool(), as_tuple=False).view(-1)
            if positions.numel() == 0:
                raise SnapshotValidationError(f"row {row} has no valid response token")
            rewards[row, int(positions[-1])] = total
        independent = no_concat_gae(
            rewards, tensors["values"], tensors["response_mask"], metadata["group_idx"], metadata["traj_idx"],
            metadata["turn_idx"], gamma=gamma, lam=lam,
        )
        fixed = no_concat_gae(
            rewards, tensors["values"], tensors["response_mask"], metadata["group_idx"], metadata["traj_idx"],
            metadata["turn_idx"], gamma=gamma, lam=lam, whiten_reference=reference,
        )
        result["branches"][variant] = {
            "token_level_rewards": rewards, "turn_totals": totals,
            "raw_advantages": independent["raw_advantages"], "independent_advantages": independent["advantages"],
            "fixed_reference_advantages": fixed["advantages"], "returns": independent["returns"],
            "whitening_stats": independent["whitening_stats"],
        }
    return result


def _aggregate(loss_matrix: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "token-mean":
        return _masked_mean(loss_matrix, mask)
    if mode == "seq-mean-token-sum":
        sums, active = (loss_matrix * mask).sum(-1), (mask.sum(-1) > 0).float()
        return _masked_mean(sums, active)
    if mode == "seq-mean-token-mean":
        counts = mask.sum(-1)
        return _masked_mean((loss_matrix * mask).sum(-1) / (counts + 1e-8), (counts > 0).float())
    if mode == "seq-mean-token-sum-norm":
        return (loss_matrix * mask).sum() / mask.shape[-1]
    raise SnapshotValidationError(f"unsupported loss aggregation mode: {mode}")


def _kl(log_prob: torch.Tensor, ref_log_prob: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in ("kl", "k1"):
        return log_prob - ref_log_prob
    if kind == "abs":
        return (log_prob - ref_log_prob).abs()
    if kind in ("mse", "k2"):
        return 0.5 * (log_prob - ref_log_prob).square()
    if kind in ("low_var_kl", "k3"):
        value = torch.clamp(ref_log_prob - log_prob, min=-20, max=20)
        return torch.clamp(torch.exp(value) - value - 1, min=-10, max=10)
    raise SnapshotValidationError(f"unsupported KL type: {kind}")


def _module_gradients(model: torch.nn.Module) -> tuple[dict[str, dict[str, Any]], torch.Tensor]:
    by_module: dict[str, list[torch.Tensor]] = defaultdict(list)
    frozen: set[str] = set()
    for name, parameter in model.named_parameters():
        module = name.rsplit(".", 1)[0] if "." in name else "<root>"
        if not parameter.requires_grad:
            frozen.add(module)
        elif parameter.grad is not None:
            by_module[module].append(parameter.grad.detach().float().reshape(-1).cpu())
        else:
            by_module[module].append(torch.zeros_like(parameter.detach().float()).reshape(-1).cpu())
    report = {module: {"status": "FROZEN" if module in frozen and module not in by_module else "TRAINABLE", "grad_norm": float(torch.linalg.vector_norm(torch.cat(parts)))} for module, parts in by_module.items()}
    report.update({module: {"status": "FROZEN", "grad_norm": None} for module in frozen if module not in report})
    flat = torch.cat([part for parts in by_module.values() for part in parts]) if by_module else torch.empty(0)
    return report, flat


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() == 0 or right.numel() == 0 or left.numel() != right.numel():
        return None
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    return None if float(denom) == 0.0 else float(torch.dot(left, right) / denom)


def gradient_diagnosis(
    payload: Mapping[str, Any], resolved_config: Mapping[str, Any], *, actor: torch.nn.Module, critic: torch.nn.Module,
    actor_logprob: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
    critic_value: Callable[[torch.nn.Module, Mapping[str, Any]], torch.Tensor],
) -> dict[str, Any]:
    """Backward each branch independently.  This function never steps or saves a model."""
    replay = recompute_branches(payload, resolved_config)
    tensors = payload["tensor_batch"]
    mask = tensors["response_mask"].to(dtype=torch.float32)
    actor_cfg = ((resolved_config.get("actor_rollout_ref") or {}).get("actor") or {})
    loss_mode = str(actor_cfg.get("loss_agg_mode", "token-mean"))
    clip_ratio = float(actor_cfg.get("clip_ratio", 0.2))
    clip_low, clip_high = float(actor_cfg.get("clip_ratio_low") or clip_ratio), float(actor_cfg.get("clip_ratio_high") or clip_ratio)
    kl_enabled = bool(actor_cfg.get("use_kl_loss", False))
    if kl_enabled and "ref_log_prob" not in tensors:
        raise SnapshotValidationError("explicit KL replay requires ref_log_prob")
    output: dict[str, Any] = {"branches": {}, "optimizer_step_called": False}
    for variant, branch in replay["branches"].items():
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        log_prob = actor_logprob(actor, payload)
        advantage = branch["independent_advantages"].to(device=log_prob.device, dtype=log_prob.dtype)
        local_mask = mask.to(log_prob.device)
        delta = torch.clamp(log_prob - tensors["old_log_probs"].to(log_prob.device), min=-20.0, max=20.0)
        ratio = torch.exp(delta)
        pg = torch.maximum(-advantage * ratio, -advantage * torch.clamp(ratio, 1 - clip_low, 1 + clip_high))
        policy_loss = _aggregate(pg, local_mask, loss_mode)
        kl_loss = torch.zeros((), dtype=policy_loss.dtype, device=policy_loss.device)
        if kl_enabled:
            kind = str(actor_cfg.get("kl_loss_type", "low_var_kl"))
            kl_loss = _aggregate(_kl(log_prob, tensors["ref_log_prob"].to(log_prob.device), kind), local_mask, loss_mode)
        actor_total = policy_loss + kl_loss * float(actor_cfg.get("kl_loss_coef", 0.0))
        actor_total.backward()
        actor_report, actor_flat = _module_gradients(actor)
        critic.zero_grad(set_to_none=True)
        vpred = critic_value(critic, payload)
        returns = branch["returns"].to(vpred.device, dtype=vpred.dtype)
        values = tensors["values"].to(vpred.device, dtype=vpred.dtype)
        value_mask = tensors["value_mask"].to(vpred.device).bool() & tensors["response_mask"].to(vpred.device).bool()
        value_mask &= ~((returns.float() - IGNORE_VALUE).abs() < 1e-6)
        if not bool(value_mask.any()):
            raise SnapshotValidationError("critic replay has no valid return targets")
        cliprange = float(((resolved_config.get("critic") or {}).get("cliprange_value", 0.5)))
        clipped = torch.clamp(vpred, values - cliprange, values + cliprange)
        vf_loss = 0.5 * _aggregate(torch.maximum((vpred - returns) ** 2, (clipped - returns) ** 2), value_mask.float(), str((resolved_config.get("critic") or {}).get("loss_agg_mode", "token-mean")))
        vf_loss.backward()
        critic_report, critic_flat = _module_gradients(critic)
        output["branches"][variant] = {
            "policy_loss": float(policy_loss.detach()), "explicit_kl_loss": float(kl_loss.detach()),
            "value_loss": float(vf_loss.detach()), "actor_gradients": actor_report, "critic_gradients": critic_report,
            "_actor_flat": actor_flat, "_critic_flat": critic_flat,
        }
    s0, s1 = output["branches"]["S0"], output["branches"]["S1"]
    delta = s1["_actor_flat"] - s0["_actor_flat"]
    output["s1_vs_s0"] = {
        "actor_gradient_cosine": _cosine(s1["_actor_flat"], s0["_actor_flat"]),
        "actor_gradient_difference_norm": float(torch.linalg.vector_norm(delta)),
    }
    for branch in output["branches"].values():
        branch.pop("_actor_flat")
        branch.pop("_critic_flat")
    return output
