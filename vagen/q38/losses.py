"""q38 (the Qwen3.8-27B rerun line), phase 3 (DESIGN_ASYNC 2.3 row 235, 6 W12; PEAR_ASSESSMENT B5): the value loss of
the per-turn PPO (algorithm.adv_estimator turn_gae, package p1tgae: the 8B main line's qw3tgd, Q5 S-PPO1 / S-PPO3).

verl V1's own ``value_loss`` (verl/verl/workers/utils/losses.py:158-216, VS) masks the loss with ``response_mask``
only.  turn_gae (vagen/turn_gae.py) puts a turn's return on the turn's first answer token (its anchor) and the sentinel
-100 on every other position, so that loss would regress every non-anchor prediction to -100.  ``vagen_value_loss`` is
the stock loss with the 8B's valid mask, ported from verl 0.6.1 (F061 = VAGEN-Lite/verl/verl):

- the mask is F061 workers/critic/dp_critic.py:227-234: ``response_mask & value_mask & ~sentinel``, where
  ``value_mask`` is vagen/ray_trainer.py compute_value_mask (RT:1126-1138, |returns - ignore| >= eps, eps 1e-6 for
  float32, 1e-2 for half precision; the 8B put it in the batch at RT:3095) and ``sentinel`` is dp_critic's own
  ``|returns + 100| < 1e-2`` (dp_critic.py:232).  The effective mask is response_mask & |returns + 100| >= 1e-2;
- the loss is verl's compute_value_loss (core_algos.py:2125-2185: 0.5 x max((v - R)^2, (clip(v, V +- c) - R)^2),
  the same formula as F061 core_algos.py:1370-1409) aggregated seq-mean-token-mean with the global batch size the
  trainer passes (TB:1880-1890: critic.ppo_mini_batch_size x rollout.n).  That is the 8B's objective: F061 averaged
  each one-episode micro-batch's token mean over its anchors (loss_agg_mode token-mean, micro 1 a card) and scaled it
  by 1 / gradient_accumulation (dp_critic.py:239-253), and FSDP averaged the ranks, i.e. the sum over the mini-batch's
  episodes of their anchor token means / (mini x n).  Padding rows (q38's balance rows, returns all -100) have no
  anchor: they add 0 to the sum and count in the denominator, like the actor's loss (DESIGN 3.1);
- a micro-batch without any anchor (only padding rows) gives the loss 0 (it still takes part in the backward) where
  F061 raised (dp_critic.py:237-238: the 8B had no padding rows).  A real row without an anchor is the driver's error
  (vagen/q38/trainer.py, the turn_gae branch), as it was F061's;
- ``max_rows_per_micro``: a micro-batch of more rows raises.  The q38 critic runs one episode a micro-batch (the 8B's
  ppo_micro_batch_size_per_gpu 1): verl's FSDP engine passes the packed sequences' cu_seqlens to a model only when the
  model's forward has a cu_seqlens parameter (transformer_impl.py:610), Qwen3_5ForTokenClassification's forward
  (transformers GenericForTokenClassification) has none, and without cu_seqlens the qwen3_5 gated-delta-net layers
  carry their state from one packed episode into the next (verl/models/transformers/qwen3_5.py:183-216;
  tests/test_critic_model_cpu.py shows it).  V1 does not hand critic.ppo_micro_batch_size_per_gpu /
  critic.use_dynamic_bsz to the critic's engine (TB:289-312; the engine's own default is dynamic batching of 32,768
  tokens), so the trainer passes them with each call (vagen/q38/trainer.py _critic_engine_keys) and this guard checks
  that the training micro-batches arrived that way.

Metrics (verl's Metric; the worker sums SUM-aggregated values over micro-batches and averages them over the data
ranks, Metric.aggregate_dp, so a per-micro-batch raw count is reported x dp_size to come out as the batch's sum):
``critic/vf_loss`` (as the stock loss: the global loss of the mini-batch), the raw sums ``critic/raw/<name>``
(RAW_SUMS: anchors, clipped anchors, sums of predictions, targets, squares, errors ...) from which the trainer computes
the pooled in-update readings (critic_update_readings), and the extremes ``critic/update/{pred,target}_valid_{min,max}``.
Names avoid "min" / "max" unless they are extremes: TB's reduce_metrics picks np.min / np.max by the key's name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

IGNORE_VALUE = -100.0  # vagen/turn_gae.py IGNORE_VALUE: the returns off each turn's first token
SENTINEL_EPS = 1e-2  # F061 dp_critic.py:232
RAW_PREFIX = "critic/raw/"
# the per-micro-batch sums of vagen_value_loss (each x dp_size, SUM): float64 on the detached tensors
RAW_SUMS = ("anchors", "clipped", "pred", "target", "target_sq", "err", "err_sq", "err_abs", "rows", "rows_with_anchor",
            "sentinel", "sentinel_pred")
EXTREMES = ("pred_valid_min", "pred_valid_max", "target_valid_min", "target_valid_max")
UPDATE_PREFIX = "critic/update/"
# compute_data_metrics' critic readings over every response token: with turn_gae nearly all of them are the sentinel
# (the 8B pops the same keys, RT:3384-3403, and logs compute_valid_return_metrics instead)
SENTINEL_DOMINATED = ("critic/returns/mean", "critic/returns/max", "critic/returns/min", "critic/values/mean",
                      "critic/values/max", "critic/values/min", "critic/vf_explained_var")


@dataclass(frozen=True)
class ValueLossConfig:
    """What vagen_value_loss reads of the critic config (picklable: set_loss_fn sends the bound loss to every worker).
    The trainer builds it from critic.* (vagen/q38/trainer.py _value_loss_fn)."""

    cliprange_value: float
    loss_agg_mode: str = "seq-mean-token-mean"
    loss_scale_factor: Optional[float] = None


def _default_eps(x: torch.Tensor) -> float:
    """vagen/ray_trainer.py _default_eps (RT:1113-1123)."""
    return 1e-2 if x.dtype in (torch.float16, torch.bfloat16) else 1e-6


def value_valid_mask(returns: torch.Tensor, response_mask: torch.Tensor, ignore_value: float = IGNORE_VALUE
                     ) -> torch.Tensor:
    """The positions whose return supervises the value network: F061 dp_critic.py:231-234 with the 8B's value_mask
    (RT:1126-1138 compute_value_mask).  Bool, the shape of ``returns``."""
    value_mask = (returns - ignore_value).abs() >= _default_eps(returns)
    sentinel = (returns.detach().float() - ignore_value).abs() < SENTINEL_EPS
    return response_mask.bool() & value_mask & ~sentinel


def _f(x: Any) -> float:
    return float(x.detach().item()) if isinstance(x, torch.Tensor) else float(x)


def value_loss_sums(vpreds: torch.Tensor, values: torch.Tensor, returns: torch.Tensor, response_mask: torch.Tensor,
                    cliprange_value: float, ignore_value: float = IGNORE_VALUE) -> dict:
    """The raw sums of one micro-batch (RAW_SUMS, float64) and its extremes (EXTREMES; +-inf without an anchor)."""
    from verl.utils import torch_functional as verl_F

    valid = value_valid_mask(returns, response_mask, ignore_value)
    with torch.no_grad():
        clipped_pred = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
        loss1 = (vpreds - returns) ** 2
        loss2 = (clipped_pred - returns) ** 2  # the same expressions as compute_value_loss (core_algos.py:2171-2174)
        clipped = (loss2 > loss1) & valid
        p = vpreds.detach().double()[valid]
        t = returns.detach().double()[valid]
        e = p - t
        has_anchor = valid.any(dim=-1)
        sentinel = response_mask.bool() & ~valid & has_anchor.unsqueeze(-1)  # real rows' non-anchor answer tokens
        out = {"anchors": float(valid.sum()), "clipped": float(clipped.sum()), "pred": _f(p.sum()),
               "target": _f(t.sum()), "target_sq": _f((t * t).sum()), "err": _f(e.sum()), "err_sq": _f((e * e).sum()),
               "err_abs": _f(e.abs().sum()), "rows": float(valid.shape[0]), "rows_with_anchor": float(has_anchor.sum()),
               "sentinel": float(sentinel.sum()), "sentinel_pred": _f(vpreds.detach().double()[sentinel].sum())}
        n = int(valid.sum())
        out["pred_valid_min"] = _f(p.min()) if n else math.inf
        out["pred_valid_max"] = _f(p.max()) if n else -math.inf
        out["target_valid_min"] = _f(t.min()) if n else math.inf
        out["target_valid_max"] = _f(t.max()) if n else -math.inf
    return out


def vagen_value_loss(config: Any, model_output: Mapping[str, Any], data: Any, dp_group: Any = None, *,
                     global_batch_size: int, max_rows_per_micro: Optional[int] = None,
                     ignore_value: float = IGNORE_VALUE):
    """The critic's loss function for ``critic_wg.set_loss_fn`` (module docstring); the signature of verl's
    value_loss (VS:158) plus keyword arguments the trainer binds with functools.partial.  Returns (loss, metrics)."""
    from verl.trainer.ppo.core_algos import compute_value_loss
    from verl.utils.metric import AggregationType, Metric
    from verl.workers.utils.padding import no_padding_2_padding

    if getattr(config, "loss_scale_factor", None) is not None:
        raise ValueError("vagen_value_loss: critic.loss_scale_factor must be null (seq-mean-token-mean over mini x n)")
    if getattr(config, "loss_agg_mode", "seq-mean-token-mean") != "seq-mean-token-mean":
        raise ValueError(f"vagen_value_loss: critic.loss_agg_mode {config.loss_agg_mode!r}; the loss is "
                         "seq-mean-token-mean (the 8B objective, module docstring)")
    vpreds = no_padding_2_padding(model_output["values"], data)  # (rows, response length), VS:170
    rows = int(vpreds.shape[0])
    if max_rows_per_micro is not None and rows > int(max_rows_per_micro):
        raise RuntimeError(
            f"vagen_value_loss: a critic micro-batch of {rows} sequences (allowed {max_rows_per_micro}): the qwen3_5 "
            "value model gets no cu_seqlens (transformer_impl.py:610), so packed episodes would share the gated-delta "
            "state; the trainer must pass use_dynamic_bsz false and micro_batch_size_per_gpu 1 to the critic "
            "(vagen/q38/trainer.py _critic_engine_keys)")
    dp_size = int(data["dp_size"])
    gbs = data["global_batch_size"]
    if gbs is None or int(gbs) != int(global_batch_size):
        raise ValueError(f"vagen_value_loss: global_batch_size {gbs!r} from the trainer, expected critic "
                         f"ppo_mini_batch_size x rollout.n = {global_batch_size} (TB:1880-1890)")
    batch_num_tokens = data["batch_num_tokens"]
    padded = data.select("values", "returns", "response_mask").to_padded_tensor()
    values, returns, response_mask = padded["values"], padded["returns"], padded["response_mask"]
    if tuple(values.shape) != tuple(vpreds.shape) or tuple(returns.shape) != tuple(vpreds.shape):
        raise RuntimeError(f"vagen_value_loss: predictions {tuple(vpreds.shape)}, old values {tuple(values.shape)}, "
                           f"returns {tuple(returns.shape)}")
    valid = value_valid_mask(returns, response_mask, ignore_value)
    vf_loss, _ = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=valid.to(dtype=response_mask.dtype),
        cliprange_value=config.cliprange_value,
        loss_agg_mode="seq-mean-token-mean",
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=int(gbs),
        loss_scale_factor=None,
    )
    sums = value_loss_sums(vpreds, values, returns, response_mask, config.cliprange_value, ignore_value)
    metrics = {"critic/vf_loss": Metric(value=vf_loss, aggregation=AggregationType.SUM)}
    for name in RAW_SUMS:
        metrics[RAW_PREFIX + name] = Metric(value=sums[name] * dp_size, aggregation=AggregationType.SUM)
    metrics[UPDATE_PREFIX + "pred_valid_min"] = Metric(value=sums["pred_valid_min"], aggregation=AggregationType.MIN)
    metrics[UPDATE_PREFIX + "pred_valid_max"] = Metric(value=sums["pred_valid_max"], aggregation=AggregationType.MAX)
    metrics[UPDATE_PREFIX + "target_valid_min"] = Metric(value=sums["target_valid_min"], aggregation=AggregationType.MIN)
    metrics[UPDATE_PREFIX + "target_valid_max"] = Metric(value=sums["target_valid_max"], aggregation=AggregationType.MAX)
    return vf_loss, metrics


# --------------------------------------------------------------------------------------------------------------
# readings (the trainer's; pure functions, tests/test_value_loss.py)
# --------------------------------------------------------------------------------------------------------------


def critic_update_readings(metrics: dict) -> dict:
    """The in-update readings of the value network from the raw sums the trainer's _update_critic got back (TB:1892-1897:
    rename_dict "critic/" then reduce_metrics: each raw sum is the batch's sum, averaged over the mini-batch iterations
    of all passes, so the ratios below pool every pass, as F061 dp_critic.py:277-299 pooled its predictions).  Pops
    the raw sums from ``metrics``; returns {name: float}.

    Pooled over the batch's anchors (F061 computed them per rank and the 8B log shows the mean over the 8 ranks, e.g.
    critic/valid_count 112.25 = 898 / 8 at qw3tgd step 101): critic/update/{valid_count, rows_with_anchor, sentinel,
    pred_valid_mean, target_valid_mean, mae_valid, rmse_valid, explained_var_valid (1 - Var(err) / (Var(target) +
    1e-5), population variances like dp_critic.py:281-283), sentinel_pred_mean, clipfrac}; the extremes are already
    there.  critic/vpred_mean and critic/vf_clipfrac are the pooled anchor values too (the stock loss: the mean over
    micro-batches of every answer token's prediction)."""
    raw = {}
    for name in RAW_SUMS:
        key = RAW_PREFIX + name
        if key not in metrics:
            raise KeyError(f"q38 trainer: the critic update returned no {key} (is the loss vagen_value_loss?)")
        raw[name] = float(metrics.pop(key))
    out = {UPDATE_PREFIX + "valid_count": raw["anchors"], UPDATE_PREFIX + "rows": raw["rows"],
           UPDATE_PREFIX + "rows_with_anchor": raw["rows_with_anchor"], UPDATE_PREFIX + "sentinel": raw["sentinel"]}
    n = raw["anchors"]
    if n <= 0:
        raise RuntimeError("q38 trainer: the critic update saw no value target (no turn anchor in the batch)")
    mean_t = raw["target"] / n
    mean_e = raw["err"] / n
    var_t = max(raw["target_sq"] / n - mean_t * mean_t, 0.0)
    var_e = max(raw["err_sq"] / n - mean_e * mean_e, 0.0)
    out.update({
        UPDATE_PREFIX + "pred_valid_mean": raw["pred"] / n,
        UPDATE_PREFIX + "target_valid_mean": mean_t,
        UPDATE_PREFIX + "mae_valid": raw["err_abs"] / n,
        UPDATE_PREFIX + "rmse_valid": math.sqrt(max(raw["err_sq"] / n, 0.0)),
        UPDATE_PREFIX + "explained_var_valid": 1.0 - var_e / (var_t + 1e-5),
        UPDATE_PREFIX + "sentinel_pred_mean": raw["sentinel_pred"] / raw["sentinel"] if raw["sentinel"] else 0.0,
        UPDATE_PREFIX + "clipfrac": raw["clipped"] / n,
        "critic/vpred_mean": raw["pred"] / n,
        "critic/vf_clipfrac": raw["clipped"] / n,
    })
    return out


def _bucket_readings(values: torch.Tensor, returns: torch.Tensor, valid: torch.Tensor) -> dict:
    """EV (RT:1239-1241: 1 - var(err) / (var(R) + 1e-5), torch.var's unbiased default) and the start-of-update value loss
    (0.5 x the mean over rows of each row's anchor mean of (V - R)^2: the loss at the update's first pass, where the
    prediction is the old value and the clip is inactive) over some rows."""
    v = values.detach().double()
    r = returns.detach().double()
    m = valid.bool()
    out = {"anchors": int(m.sum()), "rows": int(m.shape[0])}
    if out["anchors"] == 0:
        return out
    vv, rr = v[m], r[m]
    err = vv - rr
    if out["anchors"] >= 2:
        out["ev"] = float(1.0 - torch.var(err) / (torch.var(rr) + 1e-5))
    per_row = ((err.new_zeros(m.shape).masked_scatter(m, err * err)).sum(dim=-1) / m.sum(dim=-1).clamp(min=1))
    out["vf_loss"] = float(0.5 * per_row[m.any(dim=-1)].mean())
    return out


def critic_batch_readings(values: torch.Tensor, returns: torch.Tensor, response_mask: torch.Tensor,
                          advantages: torch.Tensor, lags: Sequence[Optional[int]], buckets: Sequence[int] = (0, 1),
                          ignore_value: float = IGNORE_VALUE) -> dict:
    """The driver's readings of one turn_gae batch's real rows (padded (rows, length) tensors; ``values`` are the
    critic's predictions the advantages used, ``advantages`` the whitened ones):

    - the 8B's compute_valid_return_metrics (RT:1191-1243) under its names: critic/returns/{valid_count, sentinel_ratio,
      valid_mean, valid_std, valid_min, valid_max}, critic/advantages/{valid_mean, valid_std} (over the answer tokens),
      critic/values/{valid_mean, valid_min, valid_max, valid_std, mae_valid, rmse_valid, explained_var_valid};
    - by trajectory staleness (DESIGN 7.1 question 9: the critic's targets get no off-policy correction, so its fit is
      reported per staleness): vagen/critic/{rows, anchors, ev, vf_loss}_s{b} for b in ``buckets`` (ev needs >= 2
      anchors, vf_loss >= 1; a bucket without rows has neither) and vagen/critic/rows_s_other."""
    valid = value_valid_mask(returns, response_mask, ignore_value)
    resp = response_mask.bool()
    n_resp, n_valid = int(resp.sum()), int(valid.sum())
    out = {"critic/returns/valid_count": n_valid,
           "critic/returns/sentinel_ratio": float(max(n_resp - n_valid, 0) / max(n_resp, 1))}
    if n_resp:
        a = advantages.detach().float()[resp]
        out["critic/advantages/valid_mean"] = float(a.mean())
        out["critic/advantages/valid_std"] = float(torch.std(a, unbiased=False))
    if n_valid:
        r = returns.detach()[valid]
        v = values.detach()[valid]
        err = v.float() - r.float()
        out.update({
            "critic/returns/valid_mean": float(r.float().mean()),
            "critic/returns/valid_std": float(torch.std(r.float(), unbiased=False)),
            "critic/returns/valid_min": float(r.min()),
            "critic/returns/valid_max": float(r.max()),
            "critic/values/valid_mean": float(v.float().mean()),
            "critic/values/valid_min": float(v.min()),
            "critic/values/valid_max": float(v.max()),
            "critic/values/valid_std": float(torch.std(v.float(), unbiased=False)),
            "critic/values/mae_valid": float(err.abs().mean()),
            "critic/values/rmse_valid": float(torch.sqrt((err * err).mean())),
            "critic/values/explained_var_valid": float(1.0 - torch.var(err) / (torch.var(r.float()) + 1e-5)),
        })
    if len(lags) != valid.shape[0]:
        raise ValueError(f"critic_batch_readings: {len(lags)} staleness values for {valid.shape[0]} rows")
    for b in buckets:
        rows = torch.tensor([lag == b for lag in lags], dtype=torch.bool)
        out[f"vagen/critic/rows_s{b}"] = int(rows.sum())
        if not rows.any():
            continue
        r = _bucket_readings(values[rows], returns[rows], valid[rows])
        out[f"vagen/critic/anchors_s{b}"] = r["anchors"]
        for name in ("ev", "vf_loss"):
            if name in r:
                out[f"vagen/critic/{name}_s{b}"] = r[name]
    out["vagen/critic/rows_s_other"] = int(sum(1 for lag in lags if lag not in tuple(buckets)))
    return out
