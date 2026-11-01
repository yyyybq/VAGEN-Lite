"""Reward line (2026-11-01): the per-turn credit estimator, ``algorithm.adv_estimator: turn_credit``.

The trainer's compute_advantage calls a registered estimator with the batch; this one reads the agent loop's
per-turn records (``non_tensor_batch["turn_records"]``, recorded when the estimator is selected) and the episode
returns (the summed token-level rewards), and leaves its metrics in ``meta_info["turn_credit_metrics"]``.  The
arithmetic is in vagen/turn_credit.py.
"""

from verl.trainer.ppo.core_algos import register_adv_est

from vagen.turn_credit import ESTIMATOR, settings, token_advantages


@register_adv_est(ESTIMATOR)
def compute_turn_credit_advantage(data, config=None, **kwargs):
    if "turn_records" not in data.non_tensor_batch:
        raise KeyError("turn_credit needs the agent loop's per-turn records (non_tensor_batch['turn_records']): "
                       "the agent loop records them only when algorithm.adv_estimator is turn_credit")
    cfg = settings(config)
    records = list(data.non_tensor_batch["turn_records"])
    uids = list(data.non_tensor_batch["uid"])
    returns = data.batch["token_level_rewards"].sum(dim=-1).detach().float().cpu().tolist()
    adv, metrics = token_advantages(records, uids, data.batch["response_mask"], returns, cfg)
    data.meta_info["turn_credit_metrics"] = metrics
    return adv, adv
