"""q38 (the Qwen3.8-27B rerun line): the trainer mixin of the V1 PPO trainers (DESIGN_ASYNC 2.5.2).

``VagenTrainerMixin`` sits in front of verl's own V1 trainer of a mode (``VagenSyncTrainer``,
``VagenColocateAsyncTrainer``; vagen/q38/ppo_v1.py selects them by ``trainer.v1.trainer_mode``).  Phase 1 overrides
five methods of verl/verl/trainer/ppo/v1/trainer_base.py (TB) and changes nothing else of the step:

- ``_prompt_tags`` (TB:1584): every dispatched prompt also gets ``draw_batch`` (the step that dispatched it, minus 1
  for the warm-up batch of colocate_async) and ``draw_index`` (its position in the dispatch, the 8B
  ``uid_draw_order``, RT:2731-2732).  The step-batch sampler (vagen/q38/sampler.py) waits for and selects from whole
  draw batches with them;
- ``on_train_begin``: the parent's warm-up prefetch (trainer_colocate_async.py:34-35, TB:651-678) runs with the
  offset -1, so the warm-up batch W is draw batch 0 and step s dispatches draw batch s.  A resume must have restored
  either no prompt group or one whole draw batch (a colocate checkpoint holds only B_s, TB:118-123);
- ``_compute_old_log_prob`` (TB:1708-1767): on the calibration steps (``vagen.calib_entropy_t1_steps``, default [1])
  one extra forward at T=1 before the parent's T=rollout.temperature forward, for the entropy calibration
  ``vagen/guard/entropy_t1_calib`` / ``entropy_t08_calib`` (user question 2).  The order matters: the parent's forward
  must come last, it writes the old_log_probs the TIS weights and the loss use;
- ``_compute_advantage`` (TB:1817-1876): the parent first (GRPO, token TIS weights, write-back), then every real
  (non-padding) row's advantages are multiplied by the factor the sampler gave its group, looked up strictly
  (``lookup_adv_scale``: a missing or unused factor raises, MF2), in float32 as the 8B filter does (FI:276-283).
  Then the entropy guard value (user question 2: T = rollout.temperature 0.8, token mean, kept episodes) under the
  name ``actor/entropy`` (verl's own value, seq-mean-token-mean over all rows, moves to ``actor/entropy_v1``) and the
  asynchronous diagnostics split by staleness 0 / 1 (``vagen/async/*_s0`` / ``*_s1``);
- ``_update_actor`` (TB:1901-1944): the parent, then ``actor/kl_loss`` in the 8B unit (user question 3, MF1):
  kl_v1 x rows / (vagen.ga_8b x n_real) (``kl_to_8b_unit``), verl's own value as ``actor/kl_loss_v1``; the batch's
  rows and real rows (``vagen/actor/rows``, ``vagen/actor/real_rows``).  With ppo_epochs > 1 (Q5 PT, smoke S4) also
  the value of each pass (``per_epoch_metrics``): verl reduces the per-mini-batch lists of the worker to their mean
  over every mini-batch of every pass (engine_workers.py:323-341, TB:1939-1941), so the parent's reduce_metrics is
  wrapped for the duration of the call to keep a copy of the lists (``capture_reduced_lists``; the values it returns
  are unchanged).  Pass 1 runs at the old weights: its KL is the one the driver can recompute from old_log_probs and
  ref_log_prob, and its ppo_kl is 0.

Two more overrides change only the type of logged values, not the step: ``_compute_metrics`` (TB:1946-2074, the last
step before ``self.logger.log``, TB:528-544) and ``_validate`` (TB:1092; its metrics are logged directly before
training, TB:470) turn numpy integer / floating scalars into Python floats (``plain_metric_scalars``), so the console
log has ``training/num_turns/max:12.0`` where numpy 2 printed ``np.int64(12)``, which the training chain's metric
regex does not read (data_gen/interiorgs_v10/train_chain.py _METRIC).

Metric names this module writes (all int or float; MetricsAggregator drops bools, V1/utils.py:57-71):
actor/entropy, actor/entropy_v1, actor/kl_loss, actor/kl_loss_v1, vagen/guard/entropy_t1_calib,
vagen/guard/entropy_t08_calib, vagen/guard/entropy_tokens, vagen/guard/calib_tokens, vagen/adv/real_rows,
vagen/adv/scaled_groups, vagen/adv/zero_groups, vagen/async/rows_s0, vagen/async/rows_s1, vagen/async/rows_s_other,
vagen/async/tis_w_mean_s{0,1}, vagen/async/tis_clip_frac_s{0,1}, vagen/async/logp_diff_abs_mean_s{0,1},
vagen/actor/rows, vagen/actor/real_rows, and with ppo_epochs > 1 vagen/epoch/{kl_loss,kl_loss_v1,ppo_kl,pg_clipfrac,
pg_clipfrac_lower,pg_loss,grad_norm}_e{1..epochs} (kl_loss in the 8B unit).
"""

from __future__ import annotations

import contextlib
import logging
import math
import numbers
import os
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import TensorDict
from transfer_queue import KVBatchMeta

from verl.trainer.ppo.v1.trainer_colocate_async import PPOTrainerColocateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.workers.utils.padding import response_from_nested, response_to_nested

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# the temperature of the calibration forward: the 8B entropy guard was computed at T=1 (DESIGN 7.1 question 2)
CALIB_TEMPERATURE = 1.0
DEFAULT_CALIB_STEPS = (1,)
STALENESS_BUCKETS = (0, 1)
# the worker's per-mini-batch actor metrics logged per pass when ppo_epochs > 1 (losses.py:57-155 Metric values,
# one float per mini-batch after Metric.aggregate_dp; grad_norm one per optimizer step, engine_workers.py:202-215)
EPOCH_METRICS = ("actor/kl_loss", "actor/ppo_kl", "actor/pg_clipfrac", "actor/pg_clipfrac_lower", "actor/pg_loss",
                 "actor/grad_norm")


# --------------------------------------------------------------------------------------------------------------
# pure helpers (tests/test_trainer.py)
# --------------------------------------------------------------------------------------------------------------


def kl_to_8b_unit(kl_v1: float, rows: int, ga_8b: int, n_real: int) -> float:
    """``actor/kl_loss`` in the unit of the 8B runs (MF1; DESIGN 2.5.2 ``_update_actor``).

    V1, with loss_agg_mode seq-mean-token-mean, reports per mini-batch sum(token-mean KL of each episode) / (mini x n)
    (losses.py:143-153, core_algos.py:1194-1202, SUM over micro-batches and the mean over dp ranks,
    utils/metric/utils.py:139-156), and the mean of that over the mini-batches and epochs (engine_workers.py:323-341,
    reduce_metrics); 8B reports the mean over its episodes of (token-mean KL of the episode) / ga_8b (one episode per
    micro-batch, loss_scale_factor 1 / gradient accumulation, F061 dp_actor.py:598-601,617-619,705-716).  ``rows`` is
    the trainer's batch after padding, (mini x n) x the number of mini-batches, so with the one mini-batch of the
    main line this is the design's kl_v1 x (mini x n) / (ga_8b x n_real) = kl_v1 x 8 / n_real (the contract fixes
    (mini x n) / ga_8b = 8)."""
    if n_real <= 0:
        raise ValueError(f"q38 trainer: no real (non-padding) row in the actor batch ({rows} rows)")
    if ga_8b <= 0 or rows <= 0:
        raise ValueError(f"q38 trainer: bad KL conversion inputs rows={rows}, ga_8b={ga_8b}")
    return float(kl_v1) * float(rows) / (float(ga_8b) * float(n_real))


def per_epoch_metrics(
    raw: Mapping[str, Any], epochs: int, kl_factor: float | None
) -> tuple[dict[str, float], list[str]]:
    """The value of each pass of the actor update; (metrics, keys of EPOCH_METRICS that could not be split).

    ``raw`` holds the worker's lists as the trainer reduces them (after rename_dict, TB:1939): one entry per
    mini-batch iteration, epoch-major (tensordict_utils.make_iterator: every mini-batch of pass 1, then pass 2, ...;
    engine_workers.py:276-336).  Pass e is the mean of its contiguous chunk, the same reduction verl applies to the
    whole list.  actor/kl_loss gives ``vagen/epoch/kl_loss_v1_e{e}`` (verl's unit) and, with ``kl_factor`` (the
    trainer's kl_to_8b_unit of 1), ``vagen/epoch/kl_loss_e{e}`` in the 8B unit; the others ``vagen/epoch/<name>_e{e}``.
    A list whose length is not a multiple of ``epochs``, or that holds a non-number, is left out (reported back)."""
    if epochs <= 0:
        raise ValueError(f"per_epoch_metrics: epochs must be positive, got {epochs}")
    out: dict[str, float] = {}
    skipped: list[str] = []
    for key in EPOCH_METRICS:
        values = raw.get(key)
        if values is None:
            continue
        if (not isinstance(values, (list, tuple)) or not values or len(values) % epochs
                or not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in values)):
            skipped.append(key)
            continue
        per = len(values) // epochs
        name = key.split("/", 1)[1]
        for e in range(epochs):
            chunk = [float(v) for v in values[e * per:(e + 1) * per]]
            mean = sum(chunk) / len(chunk)
            if key == "actor/kl_loss":
                out[f"vagen/epoch/kl_loss_v1_e{e + 1}"] = mean
                if kl_factor is not None:
                    out[f"vagen/epoch/kl_loss_e{e + 1}"] = mean * float(kl_factor)
            else:
                out[f"vagen/epoch/{name}_e{e + 1}"] = mean
    return out, skipped


@contextlib.contextmanager
def capture_reduced_lists(sink: dict) -> Iterator[dict]:
    """While open, every dict trainer_base's reduce_metrics reduces leaves a copy of its list values in ``sink``
    (key -> list); the reduction itself is verl's, unchanged.  TB looks the function up in its module globals at
    call time (TB:86, 1941), so the module attribute is swapped for the duration and always restored."""
    from verl.trainer.ppo.v1 import trainer_base

    original = trainer_base.reduce_metrics

    def capturing(metrics, *args, **kwargs):
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                sink[key] = list(value)
        return original(metrics, *args, **kwargs)

    trainer_base.reduce_metrics = capturing
    try:
        yield sink
    finally:
        trainer_base.reduce_metrics = original


def row_uid(key: str) -> str:
    """The prompt uid of a trajectory key {uid}_{session}_{index} (RB:69-73; the sampler and TB:961 split the same
    way; dataset uids have no "_", vagen/q38/dataset.py)."""
    return str(key).split("_", 1)[0]


def is_padding(tag: dict) -> bool:
    """Rows added by _balance_batch to fill the batch (padding_utils.py: is_padding=True in the tag)."""
    return bool(tag.get("is_padding", False))


def trajectory_staleness(tag: dict, global_steps: int) -> int | None:
    """V1's trajectory_staleness of a row, (global_steps - 1) - max_global_steps (TB:2040-2066), or None without
    the version tag."""
    hi = tag.get("max_global_steps")
    if hi is None or tag.get("min_global_steps") is None:
        return None
    return int(global_steps) - 1 - int(hi)


def _rows(t: torch.Tensor) -> list[torch.Tensor]:
    """The rows of a jagged nested tensor (or of a padded 2-D tensor)."""
    if getattr(t, "is_nested", False):
        return list(t.unbind())
    return [t[i] for i in range(t.shape[0])]


def token_mean(values: torch.Tensor, mask: torch.Tensor, real: Sequence[bool]) -> tuple[float, int]:
    """Token mean of ``values`` over the response tokens (mask 1) of the real rows, in float64; (mean, tokens).
    ``values`` and ``mask`` are nested (one row per trajectory, the response length) or padded alike."""
    total, count = 0.0, 0
    for v, m, r in zip(_rows(values), _rows(mask), real, strict=True):
        if not r:
            continue
        m = m.to(torch.bool)
        count += int(m.sum())
        total += float(v.to(torch.float64)[m].sum())
    if count == 0:
        raise RuntimeError("q38 trainer: no response token in the real rows of the batch")
    return total / count, count


def plain_metric_scalars(metrics: dict) -> int:
    """Turn every numpy integer / floating scalar value of ``metrics`` into a Python float, in place; returns how many.

    Only the type changes (``float(np.int64(12)) == 12.0``).  The console logger prints a value with its repr, and
    numpy >= 2 writes ``np.int64(12)`` / ``np.float64(8.07)``, which the training chain's metric regex
    (data_gen/interiorgs_v10/train_chain.py _METRIC) does not read; verl's _compute_metrics leaves such values for
    num_turns/{min,max,mean} (compute_data_metrics), training/num_turns/* and training/off_policy/* (TB:2041-2070).
    numpy bools and everything else stay as they are."""
    count = 0
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics[key] = float(value)
            count += 1
    return count


def _upper_threshold(spec: Any) -> float:
    """The upper TIS threshold of algorithm.rollout_correction.rollout_is_threshold (rollout_corr_helper.py:93-118)."""
    try:
        from verl.trainer.ppo.rollout_corr_helper import _parse_rollout_is_threshold

        return float(_parse_rollout_is_threshold(spec)[0])
    except ImportError:  # pragma: no cover - the helper is private; the plain float form is the q38 one
        return float(str(spec).split("_")[-1])


# --------------------------------------------------------------------------------------------------------------
# the mixin
# --------------------------------------------------------------------------------------------------------------


class VagenTrainerMixin:
    """The q38 overrides of verl's V1 PPOTrainer (module docstring).  Put it before the verl class in the bases."""

    def __init__(self, config):
        self._offset = 0
        super().__init__(config)
        for name in ("lookup_adv_scale", "clear_adv_scale"):
            if not callable(getattr(self.replay_buffer, name, None)):
                raise TypeError(
                    f"q38 trainer: the replay buffer {type(self.replay_buffer).__name__} has no {name}; the vagen trainers "
                    "need trainer.v1.sampler.custom_sampler = vagen.q38.sampler.VagenStepBatchSampler"
                )
        self._ga_8b()  # fail at start-up, not at the first update

    # ---- configuration -------------------------------------------------------------------------------------------

    def _vagen(self, key: str, default: Any = None) -> Any:
        return OmegaConf.select(self.config, f"vagen.{key}", default=default)

    def _ga_8b(self) -> int:
        ga = self._vagen("ga_8b")
        if ga is None or int(ga) <= 0:
            raise ValueError(f"q38 trainer: vagen.ga_8b must be a positive integer (main line 12), got {ga!r}")
        return int(ga)

    def _calib_steps(self) -> set[int]:
        steps = self._vagen("calib_entropy_t1_steps", default=list(DEFAULT_CALIB_STEPS))
        return {int(s) for s in (steps or [])}

    def _bypass_mode(self) -> bool:
        rc = self.config.algorithm.get("rollout_correction", None)
        return bool(rc and rc.get("bypass_mode", False))

    def _tis_threshold(self) -> float | None:
        """The upper threshold when the trainer writes token TIS weights (TB:1838-1846, 1866-1869), else None."""
        rc = self.config.algorithm.get("rollout_correction", None)
        if rc is None or rc.get("bypass_mode", False) or not rc.get("rollout_is", None):
            return None
        if not self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False):
            return None
        return _upper_threshold(rc.get("rollout_is_threshold", 2.0))

    # ---- 1. draw tags -----------------------------------------------------------------------------------------------

    def _prompt_tags(self, batch) -> list[dict]:
        """The parent's tags plus draw_batch = global_steps + offset and draw_index = the position in this dispatch."""
        tags = super()._prompt_tags(batch)
        draw_batch = int(self.global_steps) + int(getattr(self, "_offset", 0))
        for i, tag in enumerate(tags):
            tag["draw_batch"] = draw_batch
            tag["draw_index"] = i
        return tags

    # ---- 2. warm-up ---------------------------------------------------------------------------------------------------

    def on_train_begin(self):
        restored = int(getattr(self, "_restored_tq_prompt_count", 0) or 0)
        candidates = int(self.config.data.train_batch_size)
        if restored not in (0, candidates):
            raise RuntimeError(
                f"q38 trainer: the checkpoint restored {restored} prompt groups; a q38 checkpoint holds none or one whole "
                f"draw batch of {candidates} (DESIGN 2.5.2 on_train_begin)"
            )
        self._offset = -1
        try:
            super().on_train_begin()
        finally:
            self._offset = 0

    # ---- 3. old log-probs, with the T=1 entropy calibration -------------------------------------------------------------

    def _real_rows(self, batch: KVBatchMeta) -> list[bool]:
        return [not is_padding(tag) for tag in batch.tags]

    def _entropy_token_mean(self, batch: KVBatchMeta, real: Sequence[bool], response_form: bool) -> tuple[float, int]:
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["entropy", "response_mask"])
        entropy = data["entropy"]
        if not response_form:  # straight from the worker: (prompt + response) per row, TB:1742-1744
            entropy = response_from_nested(entropy, data["response_mask"])
        return token_mean(entropy, data["response_mask"], real)

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        if int(self.global_steps) not in self._calib_steps() or self._bypass_mode():
            return super()._compute_old_log_prob(batch, metrics)
        real = self._real_rows(batch)
        # the calibration forward at T=1 (the arguments of the parent's own call, TB:1724-1731, but the temperature)
        batch.extra_info.update({"calculate_entropy": True, "compute_loss": False, "temperature": CALIB_TEMPERATURE})
        output = self.actor_rollout_wg.compute_log_prob(batch)
        assert len(output) == len(batch)
        t1, tokens = self._entropy_token_mean(batch, real, response_form=False)
        # the parent's forward at T = rollout.temperature overwrites log_probs / entropy and writes old_log_probs
        batch = super()._compute_old_log_prob(batch, metrics)
        t08, tokens_08 = self._entropy_token_mean(batch, real, response_form=True)
        if tokens_08 != tokens:
            raise RuntimeError(f"q38 trainer: the calibration counted {tokens} tokens at T=1 and {tokens_08} after")
        metrics["vagen/guard/entropy_t1_calib"] = float(t1)
        metrics["vagen/guard/entropy_t08_calib"] = float(t08)
        metrics["vagen/guard/calib_tokens"] = int(tokens)
        logger.info(
            f"q38 trainer: step {self.global_steps}: entropy calibration T=1 {t1:.6g}, "
            f"T={self.config.actor_rollout_ref.rollout.temperature} {t08:.6g} over {tokens} tokens"
        )
        return batch

    # ---- 4. advantages x factor, entropy guard, asynchronous diagnostics ---------------------------------------------

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        keys = list(batch.keys)
        real = self._real_rows(batch)
        tags = [dict(t) for t in batch.tags]
        batch = super()._compute_advantage(batch, metrics)
        partition_id = batch.partition_id

        # MF2: strict lookup of every real row's factor
        real_uids = [row_uid(k) for k, r in zip(keys, real, strict=True) if r]
        if not real_uids:
            raise RuntimeError("q38 trainer: the batch has no real (non-padding) row")
        factors = self.replay_buffer.lookup_adv_scale(real_uids)
        scale = any(f != 1.0 for f in factors)
        group_factor = dict(zip(real_uids, factors, strict=True))

        tis_threshold = self._tis_threshold()
        calc = bool(self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False))
        guard = not self._bypass_mode()
        fields = ["response_mask"]
        if scale:
            fields.append("advantages")
        if guard:
            fields.append("entropy")
        if calc:
            fields += ["old_log_probs", "rollout_log_probs"]
        if tis_threshold is not None:
            fields.append("rollout_is_weights")
        data = tq.kv_batch_get(keys=keys, partition_id=partition_id, select_fields=fields)
        response_mask = data["response_mask"]
        padded = data.to_padded_tensor()
        mask = padded["response_mask"].to(torch.bool)
        real_t = torch.tensor(real, dtype=torch.bool)

        if scale:
            adv = padded["advantages"]
            factor = torch.ones(len(keys), dtype=adv.dtype)
            factor[real_t] = torch.tensor(factors, dtype=adv.dtype)
            adv = adv * factor.to(adv.device).unsqueeze(-1)  # the 8B filter's operation (FI:276-283)
            tq.kv_batch_put(
                keys=keys,
                partition_id=partition_id,
                fields=TensorDict({"advantages": response_to_nested(adv, response_mask)}, batch_size=len(keys)),
            )
        self.replay_buffer.clear_adv_scale()
        metrics["vagen/adv/real_rows"] = int(real_t.sum())
        metrics["vagen/adv/scaled_groups"] = int(sum(1 for f in group_factor.values() if f != 1.0))
        metrics["vagen/adv/zero_groups"] = int(sum(1 for f in group_factor.values() if f == 0.0))

        # entropy guard (question 2): T = rollout.temperature, token mean over the kept episodes' response tokens
        if guard:
            tok = mask & real_t.unsqueeze(-1)
            n_tok = int(tok.sum())
            if n_tok == 0:
                raise RuntimeError("q38 trainer: no response token in the real rows of the batch")
            entropy = float((padded["entropy"].to(torch.float64) * tok).sum()) / n_tok
            if "actor/entropy" in metrics:
                metrics["actor/entropy_v1"] = metrics.pop("actor/entropy")
            metrics["actor/entropy"] = entropy
            metrics["vagen/guard/entropy_tokens"] = n_tok

        # asynchronous diagnostics by trajectory staleness (d2; DESIGN 2.5.2 item 4)
        lags = []
        for k, tag, r in zip(keys, tags, real, strict=True):
            if not r:
                lags.append(None)
                continue
            lag = trajectory_staleness(tag, self.global_steps)
            if lag is None:
                raise RuntimeError(f"q38 trainer: trajectory {k} has no min/max_global_steps tag")
            lags.append(lag)
        for b in STALENESS_BUCKETS:
            metrics[f"vagen/async/rows_s{b}"] = int(sum(1 for lag in lags if lag == b))
        metrics["vagen/async/rows_s_other"] = int(sum(1 for lag in lags if lag is not None and lag not in STALENESS_BUCKETS))
        if calc:
            diff = padded["old_log_probs"].to(torch.float64) - padded["rollout_log_probs"].to(torch.float64)
            weights = padded["rollout_is_weights"].to(torch.float64) if tis_threshold is not None else None
            for b in STALENESS_BUCKETS:
                rows_b = torch.tensor([lag == b for lag in lags], dtype=torch.bool)
                tok = mask & rows_b.unsqueeze(-1)
                n_tok = int(tok.sum())
                if n_tok == 0:
                    continue
                metrics[f"vagen/async/logp_diff_abs_mean_s{b}"] = float((diff.abs() * tok).sum()) / n_tok
                if weights is not None:
                    metrics[f"vagen/async/tis_w_mean_s{b}"] = float((weights * tok).sum()) / n_tok
                    clipped = (diff > math.log(tis_threshold)) & tok
                    metrics[f"vagen/async/tis_clip_frac_s{b}"] = float(clipped.sum()) / n_tok
        return batch

    # ---- 5. actor update, KL in the 8B unit ---------------------------------------------------------------------------

    def _update_actor(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        actor = self.config.actor_rollout_ref.actor
        epochs = int(actor.get("ppo_epochs", 1) or 1)
        raw: dict = {}
        if epochs > 1:
            with capture_reduced_lists(raw):
                batch = super()._update_actor(batch, metrics)
        else:
            batch = super()._update_actor(batch, metrics)
        rows = len(batch)
        n_real = sum(1 for r in self._real_rows(batch) if r)
        metrics["vagen/actor/rows"] = int(rows)
        metrics["vagen/actor/real_rows"] = int(n_real)
        kl_factor = None
        if "actor/kl_loss" in metrics:  # absent with use_kl_loss off
            mini_n = int(actor.ppo_mini_batch_size) * int(self.config.actor_rollout_ref.rollout.n)
            if rows % mini_n:
                raise RuntimeError(f"q38 trainer: the actor batch of {rows} rows is not whole mini-batches of {mini_n}")
            kl_v1 = float(metrics["actor/kl_loss"])
            metrics["actor/kl_loss_v1"] = kl_v1
            metrics["actor/kl_loss"] = kl_to_8b_unit(kl_v1, rows, self._ga_8b(), n_real)
            kl_factor = kl_to_8b_unit(1.0, rows, self._ga_8b(), n_real)
        if epochs > 1:
            per_epoch, skipped = per_epoch_metrics(raw, epochs, kl_factor)
            if skipped:
                logger.warning(f"q38 trainer: step {self.global_steps}: no per-pass values for {skipped} (the worker's "
                               f"lists are not {epochs} passes of numbers)")
            metrics.update(per_epoch)
        return batch

    # ---- 6. logged values as Python numbers ---------------------------------------------------------------------------

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps, epoch):
        """The parent (TB:1946-2074, the last metrics of the step: fit calls it at TB:528 and logs the dict at TB:544
        with nothing in between that adds a metric), then numpy scalars as Python floats (plain_metric_scalars)."""
        out = super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        plain_metric_scalars(metrics)
        return out

    def _validate(self, *args, **kwargs):
        """The parent; numpy scalars of the validation metrics as Python floats (the val_before_train metrics go
        to ``self.logger.log`` without passing _compute_metrics, TB:464-470)."""
        val_metrics = super()._validate(*args, **kwargs)
        if isinstance(val_metrics, dict):
            plain_metric_scalars(val_metrics)
        return val_metrics


class VagenSyncTrainer(VagenTrainerMixin, PPOTrainerSync):
    """sync: the A' control and the 8B parity run (DESIGN 3.1, 5.3)."""


class VagenColocateAsyncTrainer(VagenTrainerMixin, PPOTrainerColocateAsync):
    """colocate_async: the main line (user Q2), one-step-off."""


# separate_async (VagenSeparateAsyncTrainer) is phase 4 (DESIGN 2.3, 3.2); vagen.q38.ppo_v1 refuses the mode until then
