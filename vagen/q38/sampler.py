"""q38 (the Qwen3.8-27B rerun line): the step-batch sampler of the V1 trainer (DESIGN_ASYNC 2.5.1).

``VagenStepBatchSampler`` replaces the replay buffer's own selection.  The trainer loads it with

    trainer.v1.sampler.custom_sampler = {path: pkg://vagen.q38.sampler, name: VagenStepBatchSampler}

(verl/verl/trainer/ppo/v1/trainer_base.py ``_build_replay_buffer``, TB:155-202) and calls ``sample`` once per step,
inside the step's "gen" timer.  One call:

1. waits until every prompt group of the oldest draw batch in TransferQueue is terminal: draw batch s-1 under
   colocate_async (one-step-off; the warm-up batch W has draw batch 0) and s under sync.  The draw batch and the
   position in it come from the prompt tags ``draw_batch`` / ``draw_index`` that the q38 trainer mixin adds
   (vagen/q38/trainer.py ``_prompt_tags``).  A wait longer than max(min_s, factor x the median of the last 10 waits)
   is logged as a warning, again every repeat_s; it never stops the wait;
2. failed groups (an agent loop raised in one of its sessions): failure_policy "raise" (the default, the user's
   decision on question 1 of DESIGN_ASYNC section 7) raises as soon as any failed group is seen; "skip" leaves the
   batch's failed groups out and counts them (filter/step/failed_groups);
3. reads the batch's trajectories in one ``kv_batch_get`` and runs the 8B group filter on them (FILTER_REGISTRY of
   vagen/custom_filter/filter.py, the same call as vagen/ray_trainer.py RT:3115-3117) over a one-row-per-episode
   DataProto: token_level_rewards = the episode return, advantages = 1, the episodes' reward_extra_info as the
   non-tensor columns, the draw order as meta_info["uid_draw_order"].  The selected groups are the rows it returns,
   the factor each selected group's advantages are to be multiplied by is what it leaves in the advantages column
   (1, the keep_by_return_std scale, or 0 for the zero-advantage fallback);
4. logs the filter's own metrics (filter/mixed/*, filter/keep/*), the 8B candidate metrics over every episode of the
   batch (vagen/q38/candidate_metrics.py: episode/*, train/traj_success/*, transition/*), filter/step/*,
   vagen/env/*, vagen/gen/* and vagen/async/*;
5. writes the 8B-format rollout dump of every candidate and the per-group decision table
   (vagen/q38/rollout_dump.py, atomically);
6. clears the groups it did not select from TransferQueue (prompt and trajectory keys) and returns the selected
   groups' trajectory keys (``_materialize_batch``);
7. checks the selected trajectories' version tags (min/max_global_steps, written by the rollout): a missing tag
   raises; a tag outside the bounds of the trainer mode (``version_bounds_ok``: sync min = max = s-1;
   colocate_async 0 <= (s-1) - max, (s-1) - min <= 1, max - min + 1 <= 2) is a violation (vagen/async/
   version_violations), which raises with strict_version_check.

The factors stay in the driver process: ``adv_scale`` = {selected uid: factor} is replaced as a whole by every
``sample`` call.  The trainer mixin reads it strictly (``lookup_adv_scale``: a real row whose uid has no factor, or a
set of uids that differs from the factors', raises) and empties it after use (``clear_adv_scale``).

Configuration: ``trainer.v1.sampler.sampler_kwargs`` (DESIGN_ASYNC 3.1): filter (enable, name, filter_kwargs),
adv_estimator, rollout_n, candidates_per_step, rollout_data_dir (null: no dump), model_path (the tokenizer the dump
decodes with), replace_image_tokens_for_logging, failure_policy (raise | skip), stuck_warn {min_s, factor,
repeat_s}, strict_version_check, and optionally poll_interval (seconds, default the replay buffer's 2.0).
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import DictConfig, ListConfig, OmegaConf

from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync
from vagen.q38.candidate_metrics import candidate_metrics, columns, stats_metrics
from vagen.q38.rollout_dump import decode_rows, episode_rows, write_step_dump

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

# the trajectory fields one sample() reads (DESIGN_ASYNC 2.5.1 item 4; response_mask is not needed here, and the
# token fields only for the rollout dump)
TRAJECTORY_FIELDS = ["rm_scores", "extra_fields"]
DUMP_FIELDS = ["prompts", "responses"]
FAILURE_POLICIES = ("raise", "skip")


# --------------------------------------------------------------------------------------------------------------
# the selection, a pure function (tests/test_sampler.py T1 runs it on the 8B runs' episodes)
# --------------------------------------------------------------------------------------------------------------


@dataclass
class Selection:
    """What the group filter decided for one draw batch."""

    selected: list[str]  # the selected uids, in draw order
    adv_scale: dict[str, float]  # selected uid -> factor of its advantages (float32 value)
    metrics: dict[str, Any] = field(default_factory=dict)  # the filter's own metrics
    groups: list[dict] = field(default_factory=list)  # one row per candidate group, in draw order
    fallback: bool = False  # every selected factor is 0 (no group carried a signal)


def _object_column(values: Sequence[Any]) -> np.ndarray:
    """A 1-D object array, element by element (equal-length lists would otherwise become a 2-D array)."""
    arr = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        arr[i] = v
    return arr


def group_kind(success: np.ndarray) -> str:
    """FI:251-253 / FI:154-156: the kind of a group by its episodes' success."""
    return "mixed" if success.any() and not success.all() else ("all_success" if success.all() else "all_fail")


def select_groups(
    uids: Sequence[str],
    returns: Sequence[float],
    extra: Mapping[str, Sequence[Any]],
    draw_order: Mapping[str, int],
    filter_cfg: Mapping[str, Any] | None,
    adv_estimator: str = "grpo",
    success_key: str | None = None,
) -> Selection:
    """Run the configured 8B group filter on one draw batch.

    Args:
        uids: the group (prompt uid) of each episode row.
        returns: each episode's return (float32 in the trainer; rm_scores summed).
        extra: the episodes' reward_extra_info as columns {name: [value per row]}; must hold the success key.
        draw_order: uid -> position of the group in its draw batch (the 8B uid_draw_order, RT:2732).
        filter_cfg: {"enable": bool, "name": str, "filter_kwargs": dict} (the 8B ``filter`` section); None or
            enable false selects every group with factor 1.
        adv_estimator: put into meta_info["adv_estimator"] as RT:3116 (keep_by_return_std skips its scale under
            turn_gae).
    """
    from vagen.custom_filter.filter import FILTER_REGISTRY
    from verl import DataProto

    uids = [str(u) for u in uids]
    n = len(uids)
    if n == 0:
        raise ValueError("select_groups: no episodes")
    if len(returns) != n:
        raise ValueError(f"select_groups: {len(returns)} returns for {n} episodes")
    for k, v in extra.items():
        if len(v) != n:
            raise ValueError(f"select_groups: column {k!r} has {len(v)} values for {n} episodes")
    rows_of: dict[str, list[int]] = {}
    for i, u in enumerate(uids):
        rows_of.setdefault(u, []).append(i)
    missing = [u for u in rows_of if u not in draw_order]
    if missing:
        raise ValueError(f"select_groups: groups without a draw position: {missing[:5]}")
    order = sorted(rows_of, key=lambda u: (int(draw_order[u]), rows_of[u][0]))

    cfg = dict(filter_cfg or {})
    kwargs = dict(cfg.get("filter_kwargs") or {})
    key = success_key or kwargs.get("success_key", "traj_success")
    returns_t = torch.tensor([float(r) for r in returns], dtype=torch.float32)

    if cfg.get("enable", False):
        name = cfg.get("name")
        if name not in FILTER_REGISTRY:
            raise ValueError(f"unknown filter {name!r}; registered: {sorted(FILTER_REGISTRY)}")
        if key not in extra:
            raise ValueError(f"filter {name!r} needs the per-episode {key!r}; the episodes carry {sorted(extra)}")
        dp = DataProto.from_dict(
            tensors={"token_level_rewards": returns_t.reshape(n, 1).clone(),
                     "advantages": torch.ones(n, 1, dtype=torch.float32)},
            non_tensors={"uid": _object_column(uids), **{k: _object_column(v) for k, v in extra.items()}},
            meta_info={"uid_draw_order": {u: int(draw_order[u]) for u in order}, "adv_estimator": str(adv_estimator)},
        )
        filter_metrics: dict = {}
        filtered, out_metrics = FILTER_REGISTRY[name](dp, filter_metrics, **kwargs)  # RT:3117
        filter_metrics = dict(out_metrics if out_metrics is not None else filter_metrics)
        f_uids = [str(u) for u in filtered.non_tensor_batch["uid"]]
        adv = filtered.batch["advantages"].reshape(len(f_uids), -1)
        factors: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = defaultdict(int)
        for j, u in enumerate(f_uids):
            row = adv[j]
            if not bool((row == row[0]).all()):
                raise RuntimeError(f"filter {name!r} left a non-constant advantage factor in a row of group {u}")
            counts[u] += 1
            if u in factors and not bool(factors[u] == row[0]):
                raise RuntimeError(f"filter {name!r} gave the rows of group {u} different factors")
            factors.setdefault(u, row[0])
        for u, c in counts.items():
            if c != len(rows_of[u]):
                raise RuntimeError(f"filter {name!r} kept {c} of the {len(rows_of[u])} episodes of group {u}")
        selected = [u for u in order if u in factors]
        adv_scale = {u: float(factors[u]) for u in selected}
    else:
        filter_metrics = {}
        selected = list(order)
        adv_scale = {u: 1.0 for u in selected}

    success = np.asarray(extra[key], dtype=float) > 0.5 if key in extra else None
    groups = []
    for u in order:
        idx = rows_of[u]
        std = float(torch.std(returns_t[idx])) if len(idx) > 1 else 0.0  # FI:259-260
        groups.append({
            "uid": u, "draw_index": int(draw_order[u]),
            "kind": group_kind(success[idx]) if success is not None else None,
            "return_std": std, "selected": u in adv_scale, "adv_scale": adv_scale.get(u),
            "n_episodes": len(idx),
        })
    fallback = bool(selected) and all(f == 0.0 for f in adv_scale.values())
    return Selection(selected=selected, adv_scale=adv_scale, metrics=filter_metrics, groups=groups,
                     fallback=fallback)


# --------------------------------------------------------------------------------------------------------------
# the version-tag bounds (MF3)
# --------------------------------------------------------------------------------------------------------------
#
# The tag convention, from the verl sources (verl submodule 4bfa94d; paths under verl/verl):
# - the rollout server's tag is the global_steps passed to update_weights: vllm_rollout.py:250 / 276 / 319 call
#   set_global_steps(global_steps) after a weight update, vllm_async_server.py:939-941 stores it, and
#   vllm_async_server.py:700 puts it into every generate output (extra_fields["global_steps"]); llm_server.py:
#   161-162 / 281-334 turn it into min/max_global_steps per call (min/max over a partial-rollout resume), and
#   vagen/q38/gym_agent_loop_v1.py takes the min / max over the episode's calls;
# - the trainer passes 0 (or the resumed step R) at init (on_init_end: trainer_sync.py:31-33,
#   trainer_colocate_async.py:30-32) and s after the update of step s (on_step_end: trainer_sync.py:35-38,
#   trainer_colocate_async.py:37-42); trainer_base.py:479 / 546 count the steps from 1 (R+1 on resume).
# So the weights during the sampling of step s are w_{s-1}:
# - sync (no partial rollout; the step's own draw batch, generated after the update of step s-1): min = max = s-1;
# - colocate_async (one-step-off; draw batch s-1 starts under w_{s-2} in step s-1 and may resume under w_{s-1}):
#   0 <= (s-1) - max, (s-1) - min <= 1, max - min + 1 <= 2 (and min <= max).
# Any other mode gets the colocate_async bounds.


def version_bounds_ok(trainer_mode: str, global_steps: int, lo: int, hi: int) -> bool:
    """Whether a trajectory's min/max_global_steps (lo, hi) fit the trainer mode at step ``global_steps``."""
    cur = int(global_steps) - 1
    lo, hi = int(lo), int(hi)
    if trainer_mode == "sync":
        return lo == hi == cur
    return lo <= hi and cur - hi >= 0 and cur - lo <= 1 and hi - lo + 1 <= 2


def version_bounds_text(trainer_mode: str, global_steps: int) -> str:
    cur = int(global_steps) - 1
    if trainer_mode == "sync":
        return f"min = max = s-1 = {cur}"
    return f"{cur - 1} <= min <= max <= {cur}, i.e. 0 <= (s-1)-max, (s-1)-min <= 1, max-min+1 <= 2"


# --------------------------------------------------------------------------------------------------------------
# reading TransferQueue columns
# --------------------------------------------------------------------------------------------------------------


def _tensor_rows(col: Any) -> tuple[list[torch.Tensor], bool]:
    """The rows of a TransferQueue tensor column and whether they are padded (a dense 2-D tensor); the q38 rows
    are jagged (nested) tensors, one unpadded row per trajectory."""
    if isinstance(col, torch.Tensor):
        if col.is_nested:
            return list(col.unbind(0)), False
        return list(col), True
    return [torch.as_tensor(r) for r in col], False


def _object_rows(col: Any) -> list[Any]:
    """The rows of a TransferQueue non-tensor column (NonTensorStack), unwrapped (as RB:292,301)."""
    if hasattr(col, "tolist") and not isinstance(col, torch.Tensor):
        try:
            items = col.tolist()
        except Exception:  # noqa: BLE001 - a stack that cannot list itself is iterated instead
            items = list(col)
    else:
        items = list(col)
    return [getattr(v, "data", v) for v in items]


def _plain(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _session_index(key: str) -> tuple[int, int]:
    parts = key.split("_")
    try:
        return int(parts[1]), int(parts[2])
    except (IndexError, ValueError):
        return 0, 0


# --------------------------------------------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------------------------------------------


class VagenStepBatchSampler(ReplayBufferAsync):
    """Whole-draw-batch sampling with the 8B group filter (module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kw = dict(_plain(self.sampler_kwargs) or {})
        for required in ("rollout_n", "candidates_per_step"):
            if kw.get(required) is None:
                raise ValueError(f"trainer.v1.sampler.sampler_kwargs.{required} must be set for VagenStepBatchSampler")
        self.filter_cfg: dict = dict(kw.get("filter") or {"enable": False})
        self.adv_estimator = str(kw.get("adv_estimator") or "grpo")
        self.rollout_n = int(kw["rollout_n"])
        self.candidates_per_step = int(kw["candidates_per_step"])
        self.rollout_data_dir = kw.get("rollout_data_dir") or None
        self.model_path = kw.get("model_path")
        self.replace_image_tokens = bool(kw.get("replace_image_tokens_for_logging", True))
        self.failure_policy = str(kw.get("failure_policy") or "raise")
        if self.failure_policy not in FAILURE_POLICIES:
            raise ValueError(f"failure_policy must be one of {FAILURE_POLICIES}, got {self.failure_policy!r}")
        stuck = dict(kw.get("stuck_warn") or {})
        self.stuck_min_s = float(stuck.get("min_s", 300.0))
        self.stuck_factor = float(stuck.get("factor", 3.0))
        self.stuck_repeat_s = float(stuck.get("repeat_s", 300.0))
        self.strict_version_check = bool(kw.get("strict_version_check", True))
        if kw.get("poll_interval") is not None:
            self.poll_interval = float(kw["poll_interval"])
        if self.rollout_data_dir and not self.model_path and kw.get("tokenizer") is None:
            raise ValueError("sampler_kwargs.rollout_data_dir needs model_path (the dump decodes the tokens)")
        self._tokenizer = kw.get("tokenizer")  # tests inject one; otherwise loaded from model_path on first dump
        # selected uid -> advantage factor of the last sample(); replaced as a whole by every sample()
        self.adv_scale: dict[str, float] = {}
        # partition -> prompt uid -> (draw_batch, draw_index), refreshed with every metadata sync
        self._draw: dict[str, dict[str, tuple[Any, Any]]] = defaultdict(dict)
        self._wait_history: deque[float] = deque(maxlen=10)
        self._prev_wait_s: float | None = None
        self._clock = time.monotonic
        self.last_step: dict[str, Any] = {}

    # ---- metadata: the draw tags of every prompt group -----------------------------------------------------

    def _sync_metadata_from_transfer_queue(self):
        self._draw.clear()
        super()._sync_metadata_from_transfer_queue()

    def _track_prompt(self, partition_id: str, key: str, tag: dict) -> None:
        super()._track_prompt(partition_id, key, tag)
        self._draw[partition_id][key] = (tag.get("draw_batch"), tag.get("draw_index"))

    # ---- the advantage factors (read by the trainer mixin) ---------------------------------------------------

    def lookup_adv_scale(self, real_uids: Sequence[str]) -> list[float]:
        """The factor of each real (non-padding) row's uid, strictly (MF2): a uid without a factor raises, and so
        does a set of real uids that differs from the uids sample() selected."""
        missing = sorted({str(u) for u in real_uids} - self.adv_scale.keys())
        if missing:
            raise KeyError(f"q38 sampler: rows of groups without an advantage factor: {missing[:5]}")
        unused = sorted(self.adv_scale.keys() - {str(u) for u in real_uids})
        if unused:
            raise KeyError(f"q38 sampler: selected groups missing from the batch: {unused[:5]}")
        return [self.adv_scale[str(u)] for u in real_uids]

    def clear_adv_scale(self) -> None:
        self.adv_scale = {}

    # ---- sampling --------------------------------------------------------------------------------------------

    def _expected_draw_batches(self, global_steps: int) -> set[int]:
        if self.trainer_mode == "sync":
            return {global_steps}
        if self.trainer_mode == "colocate_async":
            return {global_steps - 1}
        return {global_steps - 1, global_steps}

    def _stuck_threshold(self) -> float:
        if not self._wait_history:
            return self.stuck_min_s
        return max(self.stuck_min_s, self.stuck_factor * statistics.median(self._wait_history))

    def _raise_failures(self, failed: set[str]) -> None:
        draws = self._draw["train"]
        detail = ", ".join(f"{u} (draw batch {draws.get(u, (None,))[0]})" for u in sorted(failed)[:10])
        raise RuntimeError(
            f"q38 sampler: agent loop raised in {len(failed)} prompt group(s): {detail}; failure_policy=raise. "
            "The root exception is logged above by the agent loop worker (Error in _run_prompt for uid=...)."
        )

    def _wait_for_draw_batch(self, global_steps: int) -> tuple[int, set[str], float, int]:
        """Poll until every prompt group of the oldest draw batch is terminal.  Returns (draw batch, its uids,
        seconds waited, warnings logged)."""
        expected = self._expected_draw_batches(global_steps)
        start = self._clock()
        last_debug = time.time()
        next_warn: float | None = None
        warnings = 0
        while True:
            self._sync_metadata_from_transfer_queue()
            draws = self._draw["train"]
            if self.failure_policy == "raise" and self.failure_keys["train"]:
                self._raise_failures(set(self.failure_keys["train"]))
            if draws:
                untagged = sorted(u for u, (b, i) in draws.items() if b is None or i is None)
                if untagged:
                    raise RuntimeError(
                        f"q38 sampler: prompt groups without draw_batch/draw_index tags: {untagged[:5]} "
                        "(the trainer must be the q38 VagenTrainerMixin)"
                    )
                target = min(int(b) for b, _ in draws.values())
                if target not in expected:
                    raise RuntimeError(
                        f"q38 sampler: step {global_steps} ({self.trainer_mode}) expects draw batch "
                        f"{sorted(expected)}, the oldest one in TransferQueue is {target}"
                    )
                members = {u for u, (b, _) in draws.items() if int(b) == target}
                terminal = self.finished_keys["train"] | self.failure_keys["train"]
                if members <= terminal:
                    if len(members) != self.candidates_per_step:
                        raise RuntimeError(
                            f"q38 sampler: draw batch {target} has {len(members)} prompt groups, expected "
                            f"{self.candidates_per_step}"
                        )
                    return target, members, self._clock() - start, warnings
                waiting = sorted(members - terminal)
            else:
                target, waiting = None, []
            elapsed = self._clock() - start
            threshold = self._stuck_threshold()
            if elapsed >= threshold and (next_warn is None or elapsed >= next_warn):
                warnings += 1
                next_warn = elapsed + self.stuck_repeat_s
                logger.warning(
                    f"q38 sampler: step {global_steps} has waited {elapsed:.0f} s (threshold {threshold:.0f} s) for "
                    f"draw batch {target}: {len(waiting)} group(s) not finished, e.g. {waiting[:10]}"
                )
            last_debug = self._wait_for_next_poll("train", last_debug)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from verl.utils import hf_tokenizer

                self._tokenizer = hf_tokenizer(self.model_path)
            except ImportError:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def sample(self, global_steps: int, partition_id: str, batch_size: int):
        """One step's batch: the selected groups of the oldest draw batch (module docstring).  ``batch_size`` (the
        trainer's train_batch_size // parameter_sync_step) is not used: the draw batch decides the candidates."""
        if partition_id != "train":
            raise ValueError(f"VagenStepBatchSampler only samples the train partition, not {partition_id!r}")
        call_start = self._clock()
        target, members, wait_s, stuck_warnings = self._wait_for_draw_batch(global_steps)
        processing_start = self._clock()
        draws = self._draw["train"]
        draw_index = {u: int(draws[u][1]) for u in members}
        failed = sorted(members & self.failure_keys["train"])  # only under failure_policy=skip
        candidates = sorted(members - set(failed), key=lambda u: draw_index[u])
        if not candidates:
            raise RuntimeError(f"q38 sampler: every group of draw batch {target} failed")
        cand = set(candidates)

        # trajectory keys {uid}_{session}_{index} (RB:69-73), in draw order then session order
        by_uid: dict[str, list[str]] = defaultdict(list)
        for k in self.partitions["train"]:
            u = k.split("_")[0]
            if u in cand:
                by_uid[u].append(k)
        short = {u: len(by_uid[u]) for u in candidates if len(by_uid[u]) != self.rollout_n}
        if short:
            raise RuntimeError(f"q38 sampler: groups without {self.rollout_n} trajectories: {dict(list(short.items())[:5])}")
        keys = [k for u in candidates for k in sorted(by_uid[u], key=_session_index)]
        tags = {k: self.partitions["train"][k] for k in keys}

        fields = TRAJECTORY_FIELDS + (DUMP_FIELDS if self.rollout_data_dir else [])
        data = tq.kv_batch_get(keys=keys, partition_id="train", select_fields=fields)
        rm_rows, _ = _tensor_rows(data["rm_scores"])
        extra_fields = _object_rows(data["extra_fields"])
        if len(rm_rows) != len(keys) or len(extra_fields) != len(keys):
            raise RuntimeError("q38 sampler: TransferQueue returned a different number of rows than keys")
        returns = [float(r.sum()) for r in rm_rows]  # reward_score sits on the last token (AL:145-150)
        infos = []
        for k, ef in zip(keys, extra_fields, strict=True):
            info = ef.get("reward_extra_info") if isinstance(ef, Mapping) else None
            if not isinstance(info, Mapping):
                raise RuntimeError(f"q38 sampler: trajectory {k} has no extra_fields.reward_extra_info")
            infos.append(dict(info))
        extra = columns(infos)
        row_uids = [k.split("_")[0] for k in keys]

        selection = select_groups(row_uids, returns, extra, draw_index, self.filter_cfg, self.adv_estimator)
        selected = set(selection.selected)

        metrics: dict[str, Any] = {}
        metrics.update(selection.metrics)
        metrics.update(candidate_metrics(extra, returns))
        stats = [ef.get("vagen_stats") if isinstance(ef, Mapping) else None for ef in extra_fields]
        metrics.update(stats_metrics(stats))

        # version tags (MF3), on the selected trajectories
        lags, best_lags, spans, violations, problems = [], [], [], 0, []
        dump_versions = []
        for k in keys:
            lo, hi = tags[k].get("min_global_steps"), tags[k].get("max_global_steps")
            dump_versions.append((lo, hi))
            if k.split("_")[0] not in selected:
                continue
            if lo is None or hi is None:
                raise RuntimeError(f"q38 sampler: trajectory {k} has no min/max_global_steps tag ({lo}, {hi})")
            lo, hi = int(lo), int(hi)
            lags.append((global_steps - 1) - lo)
            best_lags.append((global_steps - 1) - hi)
            spans.append(hi - lo + 1)
            if not version_bounds_ok(self.trainer_mode, global_steps, lo, hi):
                violations += 1
                problems.append((k, lo, hi))

        next_finished = sum(
            1 for u, (b, _) in draws.items() if b is not None and int(b) == target + 1 and u in self.finished_keys["train"]
        )
        gen_2step = wait_s if self._prev_wait_s is None else (wait_s + self._prev_wait_s) / 2.0
        kept_rows = sum(1 for k in keys if k.split("_")[0] in selected)
        metrics.update({
            "filter/step/draw_batch": int(target),
            "filter/step/wait_s": float(wait_s),
            "filter/step/gen_s_2step_mean": float(gen_2step),
            "filter/step/candidates_finished": len(candidates),
            "filter/step/failed_groups": len(failed),
            "filter/step/kept_groups": len(selection.selected),
            "filter/step/kept_rows": int(kept_rows),
            "filter/step/next_batch_finished_groups": int(next_finished),
            "filter/step/stuck_warnings": int(stuck_warnings),
            "vagen/async/version_violations": int(violations),
            "vagen/async/version_lag_worst_max": int(max(lags)) if lags else 0,
            # (s-1) - max_global_steps over the selected trajectories; < 0 is a tag ahead of the weights
            "vagen/async/version_lag_best_min": int(min(best_lags)) if best_lags else 0,
            "vagen/async/version_spans_max": int(max(spans)) if spans else 0,
        })

        if self.rollout_data_dir:
            self._dump(global_steps, target, keys, data, returns, infos, selected, draw_index, dump_versions,
                       selection, failed)

        dropped = (members - selected)  # the not-selected groups, and the failed ones under skip
        self._clear_groups("train", dropped)
        batch = self._materialize_batch("train", selection.selected, dict(self.partitions["train"]))

        if problems:
            message = (
                f"q38 sampler: step {global_steps}: {violations} trajectories outside the version bounds of "
                f"{self.trainer_mode} ({version_bounds_text(self.trainer_mode, global_steps)}), "
                f"e.g. (key, min, max) {problems[:5]}"
            )
            if self.strict_version_check:
                raise RuntimeError(message)
            logger.warning(message)

        if self.adv_scale:
            logger.warning(f"q38 sampler: {len(self.adv_scale)} advantage factors of the previous step were not used")
        self.adv_scale = dict(selection.adv_scale)
        self._wait_history.append(float(wait_s))
        self._prev_wait_s = float(wait_s)
        metrics["filter/step/process_s"] = float(self._clock() - processing_start)
        self.last_step = {
            "global_steps": int(global_steps), "draw_batch": int(target), "selected": list(selection.selected),
            "dropped": sorted(dropped), "failed": failed, "fallback": selection.fallback,
            "sample_s": float(self._clock() - call_start),
        }
        return batch, metrics

    def _dump(self, step, target, keys, data, returns, infos, selected, draw_index, versions, selection, failed):
        tok = self._get_tokenizer()
        prompt_rows, p_padded = _tensor_rows(data["prompts"])
        response_rows, r_padded = _tensor_rows(data["responses"])
        inputs = decode_rows(tok, prompt_rows, padded=p_padded, side="left",
                             replace_image_tokens=self.replace_image_tokens)
        outputs = decode_rows(tok, response_rows, padded=r_padded, side="right",
                              replace_image_tokens=self.replace_image_tokens)
        uids = [k.split("_")[0] for k in keys]
        q38 = [{"uid": u, "draw_batch": int(target), "draw_index": draw_index[u], "min_global_steps": lo,
                "max_global_steps": hi} for u, (lo, hi) in zip(uids, versions, strict=True)]
        rows = episode_rows(step, inputs, outputs, returns, infos, [u in selected for u in uids], q38)
        first_info = {}
        for u, info in zip(uids, infos, strict=True):
            first_info.setdefault(u, info)
        dispatch = self.prompt_global_steps["train"]
        groups = []
        for g in selection.groups:
            info = first_info.get(g["uid"], {})
            groups.append({
                "uid": g["uid"], "task_id": info.get("task_id"), "task_type": info.get("task_type"),
                "draw_batch": int(target), "draw_index": g["draw_index"], "dispatch_step": dispatch.get(g["uid"]),
                "kind": g["kind"], "return_std": g["return_std"], "selected": g["selected"],
                "adv_scale": g["adv_scale"], "n_episodes": g["n_episodes"], "step": int(step),
            })
        for u in failed:
            groups.append({
                "uid": u, "task_id": None, "task_type": None, "draw_batch": int(target),
                "draw_index": draw_index[u], "dispatch_step": dispatch.get(u), "kind": "failed", "return_std": None,
                "selected": False, "adv_scale": None, "n_episodes": 0, "step": int(step),
            })
        write_step_dump(self.rollout_data_dir, step, rows, groups)
