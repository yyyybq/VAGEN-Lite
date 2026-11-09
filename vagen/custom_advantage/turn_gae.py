"""verl registration of the turn-level GAE estimator (algorithm.adv_estimator "turn_gae"; vagen/turn_gae.py)."""

import torch

from verl.trainer.ppo.core_algos import register_adv_est

from vagen import turn_gae as TG


@register_adv_est(TG.ESTIMATOR)
def compute_turn_gae_advantage_return(data, gamma, lam, config=None, ignore_value: float = TG.IGNORE_VALUE, **kwargs):
    """Raw token advantages (each turn's GAE advantage on its response tokens; the trainer whitens them after the
    group filter, vagen.turn_gae.whiten) and returns (each turn's return at its first response token, the sentinel
    elsewhere)."""
    TG.check_config(config)
    if TG.RECORD_KEY not in data.non_tensor_batch:
        raise KeyError(f"turn_gae needs the agent loop's {TG.RECORD_KEY!r} records (non_tensor_batch)")
    with torch.no_grad():
        advantages, returns, metrics = TG.compute(
            data.batch["token_level_rewards"], data.batch["values"], data.batch["response_mask"],
            list(data.non_tensor_batch[TG.RECORD_KEY]), float(gamma), float(lam), ignore_value=ignore_value)
    data.meta_info["turn_gae_metrics"] = metrics
    return advantages, returns.to(device=advantages.device)
