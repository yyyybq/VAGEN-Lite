"""Round 12 freeze (the user's 3.2 of 2026-10-23): unreadable turns get no positive update.

With GRPO every token of an episode shares the episode's advantage.  An answer the env could not read (no executable
action; it costs only the invalid-answer penalty and the episode goes on) inside a successful episode then gets a
positive advantage and is strengthened like the moves that solved the task: in round 12 the A group's runs amplified a
made-up action name this way.  With the switch (PPO config ``algorithm.invalid_turn_nonpositive_adv``, default off) the
agent loop records the token span of every turn the env judged unreadable, and those tokens keep their advantage only
where it is not above 0 (a failed episode's unreadable turns are still pushed down).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch

SWITCH = "invalid_turn_nonpositive_adv"


def spans_mask(spans_list: Sequence[Optional[Sequence[Sequence[int]]]], length: int) -> torch.Tensor:
    """A (batch, length) bool mask: True on the response positions inside a sample's [start, end) spans (a list, or the
    agent loop's {"spans": [...]}; None or empty: no position); spans are clipped to the response length."""
    mask = torch.zeros((len(spans_list), length), dtype=torch.bool)
    for i, spans in enumerate(spans_list):
        if isinstance(spans, dict):  # the agent loop's per-sample record {"spans": [[start, end], ...]}
            spans = spans.get("spans")
        for span in spans or ():
            start, end = max(0, int(span[0])), min(length, int(span[1]))
            if end > start:
                mask[i, start:end] = True
    return mask


def clamp_invalid_turn_advantages(advantages: torch.Tensor, response_mask: torch.Tensor,
                                  spans_list: Sequence[Optional[Sequence[Sequence[int]]]]
                                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``advantages`` with every positive value on an unreadable turn's tokens set to 0, and metrics: the share of
    the response tokens on unreadable turns, the share of those that were clamped, the episodes with such a turn."""
    mask = spans_mask(spans_list, advantages.shape[1]).to(advantages.device) & response_mask.bool()
    positive = mask & (advantages > 0)
    out = advantages.masked_fill(positive, 0.0)
    tokens = int(response_mask.bool().sum().item())
    on_invalid = int(mask.sum().item())
    metrics = {"actor/invalid_turn_token_share": on_invalid / max(1, tokens),
               "actor/invalid_turn_clamped_share": int(positive.sum().item()) / max(1, on_invalid),
               "actor/invalid_turn_episodes": float(mask.any(dim=1).sum().item())}
    return out, metrics


def enabled(config: Any) -> bool:
    """Whether the switch is on in an algorithm config (DictConfig, dict or dataclass; missing: off)."""
    if config is None:
        return False
    getter = getattr(config, "get", None)
    return bool(getter(SWITCH, False) if callable(getter) else getattr(config, SWITCH, False))
