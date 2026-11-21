"""Reward line (2026-11-01): the per-turn credit estimator, ``algorithm.adv_estimator: turn_credit``.

The trainer's compute_advantage calls a registered estimator with the batch; this one reads the agent loop's
per-turn records (``non_tensor_batch["turn_records"]``, recorded when the estimator is selected) and the episode
returns (the summed token-level rewards), and leaves its metrics in ``meta_info["turn_credit_metrics"]``.  The
arithmetic is in vagen/turn_credit.py.

With the settings' ``dump_dir`` (off by default; the smoke's check that spans cover the answers and nothing else) the
first ``dump_updates`` calls also write DUMP_DIR/update_<k>.jsonl: per sample its uid, record, return, response token
ids, response mask and token advantages (up to the mask's last 1).
"""

import json
from pathlib import Path

from verl.trainer.ppo.core_algos import register_adv_est

from vagen.turn_credit import ESTIMATOR, settings, token_advantages

_CALLS = [0]


def _dump(data, records, uids, returns, adv, cfg) -> None:
    out = Path(cfg["dump_dir"])
    out.mkdir(parents=True, exist_ok=True)
    mask = data.batch["response_mask"].detach().cpu()
    ids = data.batch["responses"].detach().cpu() if "responses" in data.batch.keys() else None
    a = adv.detach().float().cpu()
    with open(out / f"update_{_CALLS[0]:03d}.jsonl", "w") as f:
        for i, (rec, uid, ret) in enumerate(zip(records, uids, returns)):
            on = mask[i].nonzero()
            end = int(on.max()) + 1 if on.numel() else 0
            f.write(json.dumps({"i": i, "uid": str(uid), "return": ret, "record": rec,
                                "ids": ids[i, :end].tolist() if ids is not None else None,
                                "mask": mask[i, :end].int().tolist(),
                                "adv": [round(float(x), 5) for x in a[i, :end]]}) + "\n")


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
    if cfg["dump_dir"] and _CALLS[0] < cfg["dump_updates"]:
        _dump(data, records, uids, returns, adv, cfg)
    _CALLS[0] += 1
    data.meta_info["turn_credit_metrics"] = metrics
    return adv, adv
