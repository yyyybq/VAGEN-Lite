"""Round 12 freeze, dense-reward design variant V8 (the user's 10th supplement of 2026-10-23): the group comparison
normalises the success reward and the progress reward separately, then adds them, the progress part weighted.

GRPO compares a task's episodes by their returns: (return - group mean) / (group std + eps).  With a dense reward the
return mixes the success reward (+5 on success) with the progress reward (scale x the potential difference, or V6's
milestones) and the penalties, and the success reward dominates the spread whenever the group has a success.  With the
switch (PPO config ``algorithm.progress_adv_weight``, a weight w; absent or None: off) the agent loop records each
episode's progress return (the env's applied ``potential`` and ``milestone`` reward terms), and an episode's advantage
is z(return - progress) + w * z(progress), each z normalised within the task's group as GRPO does; the remainder
(return - progress) holds the success reward and the penalties.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

SWITCH = "progress_adv_weight"
PROGRESS_TERMS = ("potential", "milestone")


def weight(config: Any) -> Optional[float]:
    """The progress part's weight in an algorithm config (DictConfig, dict or dataclass); None: the switch is off."""
    if config is None:
        return None
    getter = getattr(config, "get", None)
    value = getter(SWITCH, None) if callable(getter) else getattr(config, SWITCH, None)
    return None if value is None else float(value)


def progress_of_turn(info: Any) -> float:
    """The progress reward the env applied in one turn (its reward trace's potential and milestone terms)."""
    comps = ((info or {}).get("reward_trace") or {}).get("components") or {}
    return float(sum(float((comps.get(name) or {}).get("applied") or 0.0) for name in PROGRESS_TERMS))


def _group_z(values: torch.Tensor, uids: Sequence[Any], eps: float, norm_by_std: bool) -> torch.Tensor:
    out = torch.zeros_like(values)
    groups: Dict[Any, list] = {}
    for i, u in enumerate(uids):
        groups.setdefault(u, []).append(i)
    for idx in groups.values():
        v = values[idx]
        if len(idx) == 1:  # as verl's GRPO: a lone episode compares with nothing
            mean, std = torch.tensor(0.0, dtype=values.dtype), torch.tensor(1.0, dtype=values.dtype)
        else:
            mean, std = v.mean(), v.std()
        out[idx] = (v - mean) / (std + eps) if norm_by_std else v - mean
    return out


def split_grpo_advantages(total: torch.Tensor, progress: torch.Tensor, uids: Sequence[Any],
                          response_mask: torch.Tensor, w: float, eps: float = 1e-6,
                          norm_by_std: bool = True) -> torch.Tensor:
    """(batch, length) advantages: z(total - progress) + w * z(progress) per episode, spread over its response tokens."""
    total = total.detach().float().cpu()
    progress = progress.detach().float().cpu()
    adv = _group_z(total - progress, uids, eps, norm_by_std) + float(w) * _group_z(progress, uids, eps, norm_by_std)
    return adv.to(response_mask.device).unsqueeze(-1) * response_mask


def progress_tensor(parts: Sequence[Any]) -> torch.Tensor:
    """The episodes' progress returns from the agent loop's per-sample records ({"progress": value})."""
    return torch.tensor([float((p or {}).get("progress", 0.0)) if isinstance(p, dict) else 0.0 for p in parts])
