"""Group G of matrix v3 (the user's 甲 of 2026-11-06, PREREG_G_vs_D section 9): the RL actor's entropy over the
answers' action segment.  With a targets format an answer is "<targets>...</targets><action>...</action>"; the stop
line on the entropy (guards entropy_max, 0.3 over 5 updates) reads the action segment only, as the sampling check
does, since the coordinate digits would dominate the whole answer's mean.

With the switch (PPO config ``algorithm.entropy_by_segment``, set by rl-config for a targets format; default off)
the agent loop records per turn the response positions of the action segment (from the token holding the "<" of
"<action>" to the turn's end) and of the targets segment before it; the trainer then logs ``actor/entropy_action``
and ``actor/entropy_targets`` beside ``actor/entropy``, with the same aggregation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

SWITCH = "entropy_by_segment"


def enabled(config: Any) -> bool:
    """Whether the switch is on in an algorithm config (DictConfig, dict or dataclass; missing: off)."""
    if config is None:
        return False
    getter = getattr(config, "get", None)
    return bool(getter(SWITCH, False) if callable(getter) else getattr(config, SWITCH, False))


def action_token_start(ids: Sequence[int], decode: Callable[[Sequence[int]], str]) -> Optional[int]:
    """The index in a turn's generated ``ids`` of the token that holds the "<" of "<action>" in the decoded text;
    None when the turn has no "<action>"."""
    at = decode(list(ids)).find("<action>")
    if at < 0:
        return None
    for i in range(len(ids)):
        if len(decode(list(ids[: i + 1]))) > at:
            return i
    return None


def segment_metrics(entropys: Any, response_mask: Any, non_tensor: Dict[str, Any], loss_agg_mode: str,
                    agg_loss: Callable[..., Any]) -> Dict[str, float]:
    """``actor/entropy_action`` / ``actor/entropy_targets``: the entropy aggregated as ``actor/entropy`` is, over the
    response positions of the recorded spans (non_tensor "action_spans" / "targets_spans"); a key without any
    position is left out."""
    from vagen.invalid_turns import spans_mask

    out: Dict[str, float] = {}
    for name, key in (("action", "action_spans"), ("targets", "targets_spans")):
        if key not in non_tensor:
            continue
        mask = spans_mask(list(non_tensor[key]), entropys.shape[1]).to(entropys.device) & response_mask.bool()
        if bool(mask.any()):
            out[f"actor/entropy_{name}"] = float(
                agg_loss(loss_mat=entropys, loss_mask=mask.to(response_mask.dtype), loss_agg_mode=loss_agg_mode)
                .detach().item())
    return out
