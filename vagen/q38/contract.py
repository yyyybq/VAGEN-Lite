"""Configuration contract of the q38 V1 runs (DESIGN_ASYNC 2.1, 2.5.4, 3.1; review fixes MF1, MF8, MF9, MF12).

``check(cfg, endpoint)`` runs at start-up (vagen.q38.ppo_v1.main) on the complete V1 config, before Ray starts:

- EXPECTED: config key -> expected value, each with its source (the 8B run, the asynchronous recipe, or a decision of
  the user).  Rules can depend on the run's profile: ``vagen.trainer_impl`` (``vagen``, or ``stock`` for smoke S0),
  ``vagen.parity_mode`` (the 8B parity run, DESIGN 5.3) and whether a group filter is on;
- FORBIDDEN: combinations that silently change the recipe (the table at the end of DESIGN 3.1);
- PATHS: the inputs a run reads (env yamls, the model, the filter's task rows) must exist;
- endpoint rules: the endpoint must lie within the learning-rate horizon and within total_epochs x steps_per_epoch
  (otherwise fit stops early and no endpoint checkpoint is written, trainer_base.py:492,548).

It writes ``<run>/contract_check.json`` (every rule with its expected and actual value) and
``<run>/effective_config.yaml`` (the resolved config), where ``<run>`` is the parent of ``trainer.default_local_dir``,
and raises ContractError when a rule fails that is not listed in ``vagen.contract.allow_override`` (a list of rule
names: an EXPECTED key, a FORBIDDEN name, or ``path:<key>``).  An unknown name in that list is itself an error.

    python -m vagen.q38.contract --config RL/ppo.yaml --endpoint 300          # dry run: prints, writes nothing
    python -m vagen.q38.contract --config RL/ppo.yaml --endpoint 300 --write  # also writes the two files
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf

CONTRACT_VERSION = 1

# ---------------------------------------------------------------------------------------------------------------
# Values other q38 modules assert against.  The agent loop asserts the data rows' max_turns and
# response_length_per_turn against these (MF8); 12 turns plus the initial frame is the image limit.
MAX_TURNS = 12
RESPONSE_LENGTH_PER_TURN = 384
LIMIT_IMAGES = MAX_TURNS + 1
# actor/kl_loss in the 8B unit = V1 value x rows / (vagen.ga_8b x n_real) (vagen/q38/trainer.py kl_to_8b_unit), where
# rows = len(batch), the actor batch after padding = (ppo_mini_batch_size x rollout.n) x the number of mini-batches (V1
# averages the KL over the mini-batches, engine_workers.py:323-341).  With the one mini-batch of the phase-1 runs rows
# is mini x n, and the contract fixes (mini x n) / ga_8b = 8 (MF1; main line 12 x 8 / 12, Q5 PT 24 x 8 / 24), so the
# conversion is kl_v1 x 8 / n_real; with k mini-batches it is k x that, not a constant 8
KL_8B_FACTOR = 8
LR_HORIZON = 700  # trainer.total_training_steps of the 8B main line (Y8:500); the endpoint comes from the command line
PHASE1_FILTERS = ("mixed_outcome", "keep_by_return_std")
SAMPLER = {"path": "pkg://vagen.q38.sampler", "name": "VagenStepBatchSampler"}
DATASET = {"path": "pkg://vagen.q38.dataset", "name": "AgenticDatasetV1"}
AGENT_LOOP_CONFIG = "vagen/q38/configs/agent_loop_v1.yaml"
AGENT_NAME = "gym_agent"
CHECKPOINT_CALLBACK = "vagen.q38.ckpt_callback.VagenCheckpointCallback"
SAVE_CONTENTS = ["model", "hf_model", "optimizer", "extra"]
# the 8B parity run (DESIGN 5.3, P8-2): train-side T=1 through rollout.temperature, the agent loop samples like 8B
PARITY_SAMPLING = {"temperature": 0.8, "top_p": 0.92}
TRAINER_IMPLS = ("vagen", "stock")
FAILURE_POLICIES = ("raise", "skip")

A = "actor_rollout_ref.actor"
R = "actor_rollout_ref.rollout"
SK = "trainer.v1.sampler.sampler_kwargs"

# where the 8B values come from: Y8 = R8/rl_qw3v4s3/ppo.yaml (the 8B main line V4)
Y8 = "8B Y8"
ASYNC = "async recipe (DESIGN_ASYNC 3.1)"


class ContractError(RuntimeError):
    """A rule of the contract failed and is not listed in vagen.contract.allow_override."""


class _Absent:
    """Marks a key that is not in the config (as opposed to a key whose value is null)."""

    def __repr__(self) -> str:
        return "<absent>"


ABSENT = _Absent()


def get(cfg: DictConfig, key: str) -> Any:
    """The resolved value at a dotted key (ABSENT if the key is not there); containers come back as plain Python."""
    value = OmegaConf.select(cfg, key, default=ABSENT)
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    return value


def values_equal(actual: Any, expected: Any) -> bool:
    """Equality for config values: bools only equal bools; numbers compare by value (int 2 == float 2.0)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, float) or isinstance(actual, float):
            return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=0.0)
        return actual == expected
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(values_equal(actual[k], expected[k]) for k in expected))
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(values_equal(a, e) for a, e in zip(actual, expected)))
    return actual == expected


def jsonable(value: Any) -> Any:
    """A JSON-safe copy of a config value (ABSENT becomes the string "<absent>")."""
    if value is ABSENT:
        return "<absent>"
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


# ---------------------------------------------------------------------------------------------------------------
# Profile of a run


def trainer_impl(cfg: DictConfig) -> str:
    """``vagen`` (our trainers and sampler) or ``stock`` (verl's own V1 classes, smoke S0); absent means vagen."""
    value = OmegaConf.select(cfg, "vagen.trainer_impl", default="vagen")
    return "vagen" if value is None else str(value)


def parity_mode(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "vagen.parity_mode", default=False))


def filter_on(cfg: DictConfig) -> bool:
    return bool(OmegaConf.select(cfg, "filter.enable", default=False))


def _vagen(cfg) -> bool:
    return trainer_impl(cfg) == "vagen"


def _stock(cfg) -> bool:
    return trainer_impl(cfg) == "stock"


def _vagen_filter(cfg) -> bool:
    return _vagen(cfg) and filter_on(cfg)


def _keep_by_std(cfg) -> bool:
    return _vagen_filter(cfg) and OmegaConf.select(cfg, "filter.name", default=None) == "keep_by_return_std"


def _not_parity(cfg) -> bool:
    return not parity_mode(cfg)


def _grpo(cfg) -> bool:
    return OmegaConf.select(cfg, "algorithm.adv_estimator", default=None) == "grpo"


def _always(cfg) -> bool:
    return True


# ---------------------------------------------------------------------------------------------------------------
# Derived values (from files the config names) and the run directory


@dataclass
class Context:
    endpoint: Optional[int]
    run_dir: Optional[Path]
    dataset_size: Optional[int] = None
    steps_per_epoch: Optional[int] = None
    env_specs: Dict[str, List[dict]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def run_dir_of(cfg: DictConfig) -> Optional[Path]:
    """The run directory: the parent of trainer.default_local_dir (relative paths count from the working directory,
    as V1 resolves them, trainer_base.py:889-892)."""
    local_dir = get(cfg, "trainer.default_local_dir")
    if not isinstance(local_dir, str) or not local_dir:
        return None
    return Path(os.path.abspath(os.path.expanduser(local_dir))).parent


def read_env_specs(path: str) -> List[dict]:
    """The env specs of an AgenticDataset yaml, read without resolving their configs (they hold ${oc.env:...}).

    Defaults are those of vagen/gym_agent_dataset.py EnvSpec (max_turns 1, response_length_per_turn None).  Read
    with PyYAML: OmegaConf takes about 40 s on the login node for the 8B V4 env yaml (a 10,798-entry seed_list).
    """
    import yaml

    from vagen.q38.dataset import single_data_file

    with open(single_data_file(path)) as f:
        doc = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
    specs = []
    for spec in doc.get("envs", []) or []:
        specs.append({
            "name": spec.get("name"),
            "n_envs": int(spec.get("n_envs")),
            "max_turns": spec.get("max_turns", 1),
            "response_length_per_turn": spec.get("response_length_per_turn", None),
            "agent_name": spec.get("agent_name", None),
        })
    return specs


def build_context(cfg: DictConfig, endpoint: Optional[int], read_files: bool = True) -> Context:
    ctx = Context(endpoint=endpoint, run_dir=run_dir_of(cfg))
    if not read_files:
        return ctx
    for split, key in (("train", "data.train_files"), ("val", "data.val_files")):
        path = get(cfg, key)
        try:
            ctx.env_specs[split] = read_env_specs(path)
        except Exception as e:  # reported as a violation of the rules that need the specs
            ctx.errors.append(f"{key}: cannot read the env specs of {path!r}: {type(e).__name__}: {e}")
    if "train" in ctx.env_specs:
        ctx.dataset_size = sum(s["n_envs"] for s in ctx.env_specs["train"])
        batch = get(cfg, "data.train_batch_size")
        if isinstance(batch, int) and batch > 0:
            # V1: steps_per_epoch = len(train_dataset) // train_batch_size (trainer_base.py:792)
            ctx.steps_per_epoch = ctx.dataset_size // batch
    return ctx


# ---------------------------------------------------------------------------------------------------------------
# Matchers for EXPECTED


class Matcher:
    def matches(self, actual: Any, cfg: DictConfig, ctx: Context) -> bool:
        raise NotImplementedError

    def describe(self, cfg: DictConfig, ctx: Context) -> Any:
        raise NotImplementedError


class OneOf(Matcher):
    def __init__(self, *values, allow_absent: bool = False):
        self.values = values
        self.allow_absent = allow_absent

    def matches(self, actual, cfg, ctx):
        if actual is ABSENT:
            return self.allow_absent
        return any(values_equal(actual, v) for v in self.values)

    def describe(self, cfg, ctx):
        return {"one_of": jsonable(list(self.values)), **({"or": "<absent>"} if self.allow_absent else {})}


class AbsentOr(Matcher):
    def __init__(self, value):
        self.value = value

    def matches(self, actual, cfg, ctx):
        return actual is ABSENT or values_equal(actual, self.value)

    def describe(self, cfg, ctx):
        return {"absent_or": jsonable(self.value)}


class Contains(Matcher):
    def __init__(self, item):
        self.item = item

    def matches(self, actual, cfg, ctx):
        items = [actual] if isinstance(actual, str) else actual
        return isinstance(items, (list, tuple)) and self.item in items

    def describe(self, cfg, ctx):
        return {"contains": self.item}


class SameSet(Matcher):
    def __init__(self, items):
        self.items = list(items)

    def matches(self, actual, cfg, ctx):
        return isinstance(actual, (list, tuple)) and sorted(map(str, actual)) == sorted(self.items) \
            and len(actual) == len(self.items)

    def describe(self, cfg, ctx):
        return {"same_set": self.items}


class NonEmptyStr(Matcher):
    def matches(self, actual, cfg, ctx):
        return isinstance(actual, str) and bool(actual.strip())

    def describe(self, cfg, ctx):
        return "a non-empty string"


class SameAs(Matcher):
    """Equal to the resolved value of another key (sampler_kwargs interpolate one value written once)."""

    def __init__(self, other: str):
        self.other = other

    def matches(self, actual, cfg, ctx):
        other = get(cfg, self.other)
        return actual is not ABSENT and other is not ABSENT and values_equal(actual, other)

    def describe(self, cfg, ctx):
        return {"same_as": self.other, "value": jsonable(get(cfg, self.other))}


class Computed(Matcher):
    """Equal to a value computed from the config and the derived context; None from the function = cannot tell."""

    def __init__(self, fn: Callable[[DictConfig, Context], Any], text: str):
        self.fn = fn
        self.text = text

    def matches(self, actual, cfg, ctx):
        expected = self.fn(cfg, ctx)
        return expected is not None and values_equal(actual, expected)

    def describe(self, cfg, ctx):
        return {"computed": self.text, "value": jsonable(self.fn(cfg, ctx))}


def _total_epochs_expected(cfg, ctx):
    if not ctx.steps_per_epoch:
        return None
    horizon = get(cfg, "trainer.total_training_steps")
    horizon = horizon if isinstance(horizon, int) and horizon > 0 else LR_HORIZON
    return math.ceil(horizon / ctx.steps_per_epoch) + 1


def _rollout_data_dir_expected(cfg, ctx):
    return None if ctx.run_dir is None else str(ctx.run_dir / "rollout_data")


def _env_values(split: str, item: str):
    """Getter: the set of an item over the env specs of one split (one value expected)."""

    def getter(cfg, ctx):
        if split not in ctx.env_specs:
            return ABSENT
        values = sorted({jsonable(s[item]) for s in ctx.env_specs[split]}, key=repr)
        return values[0] if len(values) == 1 else values

    return getter


@dataclass(frozen=True)
class Expect:
    key: str
    expected: Any  # a literal (compared with values_equal) or a Matcher
    source: str
    when: Callable[[DictConfig], bool] = _always
    getter: Optional[Callable[[DictConfig, Context], Any]] = None  # default: get(cfg, key)

    def actual(self, cfg: DictConfig, ctx: Context) -> Any:
        return self.getter(cfg, ctx) if self.getter is not None else get(cfg, self.key)

    def ok(self, actual: Any, cfg: DictConfig, ctx: Context) -> bool:
        if isinstance(self.expected, Matcher):
            return self.expected.matches(actual, cfg, ctx)
        return actual is not ABSENT and values_equal(actual, self.expected)

    def describe(self, cfg: DictConfig, ctx: Context) -> Any:
        if isinstance(self.expected, Matcher):
            return self.expected.describe(cfg, ctx)
        return jsonable(self.expected)


EXPECTED: List[Expect] = [
    # --- mode and sampler
    Expect("trainer.use_v1", True, "V1 trainer family (user Q1, 2026-11-17)"),
    Expect("trainer.v1.trainer_mode", OneOf("colocate_async", "sync"),
           "user Q2 (asynchronous RL): colocate_async; sync only for the A' control and the 8B parity run; "
           "separate_async is phase 4 (question 11)", _not_parity),
    Expect("trainer.v1.trainer_mode", "sync", "8B parity run P8-2 uses VagenSyncTrainer (DESIGN 5.3)",
           parity_mode),
    Expect("trainer.v1.colocate_async.num_warmup_batches", 1, ASYNC + ": prefetch one batch, one-step-off"),
    Expect("trainer.v1.sampler.max_off_policy_threshold", 2,
           ASYNC + ": at most one version behind; recorded only, the guarantee is the step-batch sampler (DESIGN 4.4)"),
    Expect("trainer.v1.sampler.max_off_policy_strategy", "wait", ASYNC),
    Expect("trainer.v1.sampler.custom_sampler", SAMPLER, "step-batch sampler (DESIGN 2.5.1; TB:155-169)", _vagen),
    Expect("trainer.v1.sampler.custom_sampler.path", AbsentOr(None),
           "trainer_impl stock: verl's own ReplayBufferAsync (smoke S0)", _stock),
    Expect("trainer.v1.sampler.sync_refill_failed_groups", False, ASYNC + ": no refill"),
    Expect("filter.enable", AbsentOr(False), "trainer_impl stock: plain GRPO, nothing applies filter.* (smoke S0)",
           _stock),
    Expect("filter.name", OneOf(*PHASE1_FILTERS),
           "8B group filters of phase 1 (Y8:579-592; S0 mixed_outcome, V4 keep_by_return_std); nominal is phase 2",
           _vagen_filter),
    Expect("filter.filter_kwargs.target_groups", 12, Y8 + ":582", _vagen_filter),
    Expect("filter.filter_kwargs.std_threshold", 0.3318, Y8 + ":583", _keep_by_std),
    Expect("filter.filter_kwargs.std_floor", 1.0, Y8 + ":584", _keep_by_std),
    Expect("filter.filter_kwargs.rows", NonEmptyStr(), Y8 + ":585 (the run's own task rows)", _keep_by_std),
    Expect("data.train_batch_size", 24, Y8 + ":291 (24 candidate tasks a step)"),
    Expect("data.gen_batch_size", 1, "async V1 forces 1 (TB:754-766)"),
    Expect("data.custom_cls", DATASET, "DESIGN 2.1"),
    Expect(R + ".n", 8, Y8 + ":196"),
    # --- loss and optimizer
    Expect(A + ".ppo_mini_batch_size", 12, Y8 + ":42 (padded to 96 rows, one optimizer step)"),
    Expect(A + ".ppo_epochs", 1, Y8 + ":68"),
    Expect(A + ".shuffle", False, Y8 + ":69"),
    Expect(A + ".loss_agg_mode", "seq-mean-token-mean",
           "the 8B's actual objective: sum of per-episode token means / nominal episodes (DESIGN 3.1)"),
    # verl default null (_generated_ppo_trainer.yaml:104); a value goes into global_batch_info and switches the metric
    # aggregation to SUM (VS/workers/utils/losses.py:70-85): the objective and the KL conversion would change
    Expect(A + ".loss_scale_factor", AbsentOr(None), "seq-mean-token-mean over mini x n; MF1 KL conversion"),
    Expect(A + ".optim.clip_grad", 0.3, "8B effective clip 0.3 (Y8:105 actor.grad_clip; the new engine reads "
                                         "optim.clip_grad only, transformer_impl.py:859-876)"),
    Expect(A + ".optim.optimizer", "AdamW", Y8 + ":5"),
    Expect(A + ".optim.betas", [0.9, 0.999], Y8 + ":12-14"),
    Expect(A + ".optim.lr", 2e-6, Y8 + ":7"),
    Expect(A + ".optim.lr_scheduler_type", "cosine", Y8 + ":18 (V1 default constant, j2 F21)"),
    Expect(A + ".optim.min_lr_ratio", 0.05, Y8 + ":16"),
    Expect(A + ".optim.num_cycles", 0.5, Y8 + ":17"),
    Expect(A + ".optim.lr_warmup_steps_ratio", 0.0, Y8 + ":8"),
    Expect(A + ".optim.lr_warmup_steps", OneOf(-1, 0, None), Y8 + ":11 (no warmup)"),
    Expect(A + ".optim.weight_decay", 0.01, Y8 + ":10 (V1 default 0.0, j2 F21)"),
    # verl default true (_generated_ppo_trainer.yaml:24); false shifts the whole schedule by one step
    Expect(A + ".optim.zero_indexed_step", True, "T11: the 8B lr at step 100 needs the 0-indexed schedule"),
    Expect("trainer.total_training_steps", LR_HORIZON,
           Y8 + ":500 (learning-rate horizon; the endpoint comes from --endpoint)"),
    Expect(A + ".clip_ratio", 0.2, Y8 + ":47"),
    Expect(A + ".clip_ratio_low", 0.2, Y8 + ":48"),
    Expect(A + ".clip_ratio_high", 0.2, Y8 + ":49"),
    Expect(A + ".clip_ratio_c", 3.0, Y8 + ":59 (dual clip)"),
    Expect(A + ".use_kl_loss", True, Y8 + ":64"),
    Expect(A + ".kl_loss_type", "low_var_kl", Y8 + ":67"),
    Expect(A + ".kl_loss_coef", 0.05, Y8 + ":66"),
    Expect(A + ".entropy_coeff", 0.01, Y8 + ":61 (acts on the T 0.8 entropy, registered R3)"),
    Expect(A + ".policy_loss.loss_mode", "vanilla", Y8 + ":53 (decoupled PPO through the native path)"),
    Expect("algorithm.adv_estimator", "grpo", Y8 + ":480 (phase 1)"),
    Expect("algorithm.norm_adv_by_std_in_grpo", True, Y8 + ":481"),
    Expect("critic.enable", OneOf(None, False, allow_absent=True), "GRPO trains no critic (utils.py:96-107)", _grpo),
    Expect("algorithm.rollout_correction.rollout_is", "token",
           "user PEAR 2.6: decoupled PPO + token-level TIS; the dataclass default is sequence (algorithm.py:193)",
           _not_parity),
    Expect("algorithm.rollout_correction.rollout_is", None, "8B parity run: no TIS (DESIGN 5.3)", parity_mode),
    Expect("algorithm.rollout_correction.rollout_is_threshold", 2.0, "user PEAR 2.6: w = min(exp(old - rollout), 2)"),
    Expect("algorithm.rollout_correction.rollout_rs", None, ASYNC + ": no rejection sampling"),
    Expect("algorithm.rollout_correction.bypass_mode", False, ASYNC + ": pi_old recomputed on the trainer"),
    Expect("algorithm.rollout_correction.rollout_is_batch_normalize", False, ASYNC),
    Expect("algorithm.rollout_correction.score_centering", AbsentOr(False), ASYNC + " (no score centering, 4.4 #3)"),
    Expect("vagen.ga_8b", 12, "8B gradient accumulation of the main line (12 kept groups); Q5 PT 24 (MF1)"),
    Expect("vagen.onpolicy_shortcut", AbsentOr(False), "on until S9 passes (question 8)"),
    # --- reward and verl's rollout skip
    # A reward model without its own resource pool has no reward-loop handles (reward_loop.py:320-322), so V1 scores
    # the batch after the sampler chose the groups and overwrites rm_scores (TB:604-606, TB:1641-1653).  The legacy
    # reward_model.enable is copied into reward.reward_model when not null (reward_loop.py:62-66), so both are checked.
    Expect("reward.reward_model.enable", AbsentOr(False),
           "the env reward is the only reward; a colocated RM would overwrite rm_scores after selection (TB:604-606)"),
    Expect("reward_model.enable", OneOf(None, False, allow_absent=True),
           "legacy key, migrated into reward.reward_model.enable when not null (reward_loop.py:62-66)"),
    # verl's rollout skip replays cached generations: rollout_tq would skip the warm-up dispatch (TB:659) and cache the
    # dispatch (TB:1593 @SkipManager.annotate_tq) while the sampler's sample() is not annotated (RB:618 is); rollout
    # is the agent-loop manager's (AL:1288), async_rollout the fully-async rollouter's
    Expect("skip.rollout_tq.enable", AbsentOr(False),
           "the step-batch sampler does not support verl's rollout skip (sampler.py sample is not "
           "SkipManager-annotated; TB:659 would drop the warm-up)"),
    Expect("skip.rollout.enable", AbsentOr(False), "no replay of cached generations (AL:1288)"),
    Expect("skip.async_rollout.enable", AbsentOr(False),
           "no replay of cached generations (fully_async_rollouter.py:306)"),
    # --- sampling and agent loop
    Expect(R + ".temperature", 0.8, "8B sampling T 0.8 (Y8:169); the train side uses the same T in V1 (TB:600,1729)",
           _not_parity),
    Expect(R + ".temperature", 1.0, "8B parity run: train-side T=1 (DESIGN 5.3)", parity_mode),
    Expect(R + ".top_p", 1.0, "user PEAR 2.6: top-p < 1 biases TIS (rollout.py:197)"),
    Expect(R + ".top_k", -1, Y8 + ":170"),
    Expect(R + ".calculate_log_probs", True, "TIS needs rollout_log_probs (8B had false)"),
    Expect(R + ".logprobs_mode", "processed_logprobs", "post-temperature logprobs (raw are pre-temperature)"),
    Expect(R + ".prompt_length", 4096, Y8 + ":289 (data.max_prompt_length)"),
    Expect(R + ".response_length", 4096, Y8 + ":290 (data.max_response_length; 384 a turn via max_tokens)"),
    Expect(R + ".limit_images", LIMIT_IMAGES, "12 turns + the initial frame (MF8); vLLM 0.28 default not checked"),
    Expect(R + ".agent.agent_loop_config_path", AGENT_LOOP_CONFIG, "DESIGN 2.1 (AL:559-561)"),
    Expect(R + ".agent.default_agent_loop", AGENT_NAME, "DESIGN 3.1"),
    # rollout.kernels and rollout.custom.lmcache are PEAR's keys (q38/src/pear: pear/rollout/config.py:21-30,
    # pear/rollout/lmcache.py:221-228), not in verl's RolloutConfig; a config copied from a PEAR recipe must not
    # bring them in.  A vLLM KV connector (LMCache's way into vLLM) is refused too.
    Expect(R + ".kernels", AbsentOr(None),
           "colocated rollout: vLLM's own kernels; PEAR's need free_cache_engine false (PEAR 2.3 A, 3.2 I8)"),
    Expect(R + ".custom.lmcache.enable", AbsentOr(False),
           "colocated rollout: no LMCache, it needs sleep mode off (PEAR 2.3 A, 3.2 I8)"),
    Expect(R + ".engine_kwargs.vllm.kv_transfer_config", AbsentOr(None),
           "colocated rollout: no KV connector (LMCache, PEAR 3.2 I8)"),
    Expect(R + ".free_cache_engine", True, "colocated rollout (PEAR 2.3 A)"),
    Expect(R + ".enable_sleep_mode", AbsentOr(True), "colocated rollout sleeps while training (PEAR 2.3 A)"),
    Expect("vagen.parity_mode", OneOf(True, False, allow_absent=True), "DESIGN 3.1 (true only for P8-2)"),
    Expect("vagen.parity_sampling_override", PARITY_SAMPLING, "8B parity run: sample like 8B (Y8:169-171)",
           parity_mode),
    Expect("data.train_files#max_turns", MAX_TURNS, "12 turns (MF8; GL:484-486)", getter=_env_values("train", "max_turns")),
    Expect("data.train_files#response_length_per_turn", RESPONSE_LENGTH_PER_TURN, "384 tokens a turn (MF8)",
           getter=_env_values("train", "response_length_per_turn")),
    Expect("data.train_files#agent_name", AbsentOr(None),
           "the rows' agent_name is gym_agent (gym_agent_dataset.py:219), not set per spec",
           getter=_env_values("train", "agent_name")),
    # --- checkpoints, validation, logging
    Expect("trainer.critic_warmup", 0, Y8 + ":518"),
    Expect("trainer.total_epochs", Computed(_total_epochs_expected, "ceil(horizon / steps_per_epoch) + 1"),
           "DESIGN 3.1 (enough epochs for the horizon)"),
    Expect("trainer.save_freq", 50, Y8 + ":511"),
    Expect("trainer.checkpoint_callback_class", CHECKPOINT_CALLBACK, "milestone retention (DESIGN 2.5.5)", _vagen),
    Expect("trainer.checkpoint_callback_class", OneOf(None, CHECKPOINT_CALLBACK, allow_absent=True),
           "trainer_impl stock (smoke S0)", _stock),
    Expect(A + ".checkpoint.save_contents", SameSet(SAVE_CONTENTS),
           Y8 + ":72-76 (without hf_model the chain would evaluate an empty folder, PEAR m7)"),
    Expect("trainer.test_freq", -1, Y8 + ":517"),
    Expect("trainer.val_before_train", False, Y8 + ":515"),
    Expect("trainer.val_only", False, Y8 + ":516"),
    Expect("trainer.rollout_data_dir", None, "the sampler writes the dump (DESIGN 2.5.5); V1's own dump off"),
    Expect("trainer.logger", Contains("console"), "train_chain parses the console lines (aggregate_logger.py:26-32)"),
    Expect("vagen.trainer_impl", OneOf(*TRAINER_IMPLS, allow_absent=True), "runner switch (DESIGN 2.1)"),
    Expect("vagen.failure_policy", "raise", "user question 1, recommended (a) (2026-11-18: all recommendations)",
           _vagen),
    # --- the sampler's keyword arguments: interpolations of values written once (DESIGN 3.1)
    Expect(SK + ".filter", SameAs("filter"), "DESIGN 3.1", _vagen),
    Expect(SK + ".adv_estimator", SameAs("algorithm.adv_estimator"), "DESIGN 3.1", _vagen),
    Expect(SK + ".rollout_n", SameAs(R + ".n"), "DESIGN 3.1", _vagen),
    Expect(SK + ".candidates_per_step", SameAs("data.train_batch_size"), "DESIGN 3.1", _vagen),
    Expect(SK + ".rollout_data_dir", Computed(_rollout_data_dir_expected, "<run>/rollout_data"), "DESIGN 2.5.5",
           _vagen),
    Expect(SK + ".model_path", SameAs("actor_rollout_ref.model.path"), "DESIGN 3.1", _vagen),
    Expect(SK + ".replace_image_tokens_for_logging", True, "8B dump format (Y8:528)", _vagen),
    Expect(SK + ".failure_policy", SameAs("vagen.failure_policy"), "DESIGN 3.1", _vagen),
    Expect(SK + ".stuck_warn", SameAs("vagen.stuck_warn"), "DESIGN 3.1 (MF5)", _vagen),
    Expect(SK + ".strict_version_check", SameAs("vagen.async.strict_version_check"), "DESIGN 3.1 (MF3)", _vagen),
]

# recorded in contract_check.json, not asserted (DESIGN 3.1 rows without a tick, and the run's own choices)
INFO_KEYS = [
    "vagen.async.strict_version_check", "vagen.stuck_warn", "vagen.calib_entropy_t1_steps", "vagen.endpoint",
    "trainer.resume_mode", "trainer.ckpt_keep_every", "trainer.ckpt_keep_latest", "trainer.ckpt_prune_to_hf",
    "trainer.default_local_dir", "trainer.experiment_name", "trainer.n_gpus_per_node", "trainer.nnodes",
    "actor_rollout_ref.model.path", "actor_rollout_ref.model.override_config", "data.train_files", "data.val_files",
    "data.seed", "data.pixel_dtype", "data.apply_chat_template_kwargs", R + ".name", R + ".scheduling_policy",
    R + ".tensor_model_parallel_size", R + ".gpu_memory_utilization", R + ".max_num_seqs", R + ".enforce_eager",
    R + ".agent.num_workers", A + ".ppo_micro_batch_size_per_gpu", A + ".fsdp_config.param_offload",
    A + ".fsdp_config.optimizer_offload", "actor_rollout_ref.ref.fsdp_config.param_offload",
    "async_training", "filter", "trainer.v1.sampler.sampler_kwargs",
]

# the inputs a run reads; checked to exist when check_paths is on
PATH_KEYS = ["data.train_files", "data.val_files", "actor_rollout_ref.model.path"]


# ---------------------------------------------------------------------------------------------------------------
# FORBIDDEN: each test returns a description of the violation, or None


def _effective_top_p(cfg):
    override = get(cfg, "vagen.parity_sampling_override")
    if parity_mode(cfg) and isinstance(override, dict) and "top_p" in override:
        return override["top_p"]
    return get(cfg, R + ".top_p")


def _f_top_p_tis(cfg, ctx):
    top_p, rollout_is = _effective_top_p(cfg), get(cfg, "algorithm.rollout_correction.rollout_is")
    if isinstance(top_p, (int, float)) and top_p < 1 and rollout_is not in (None, ABSENT):
        return f"sampling top_p={top_p} with rollout_is={rollout_is!r}"
    return None


def _f_grad_clip(cfg, ctx):
    found = []
    if get(cfg, A + ".grad_clip") is not ABSENT:
        found.append(f"{A}.grad_clip={get(cfg, A + '.grad_clip')!r}")
    if get(cfg, A + ".optim.grad_clip") not in (None, ABSENT):
        # the deprecated alias overrides optim.clip_grad (verl/verl/workers/config/optimizer.py:54-61)
        found.append(f"{A}.optim.grad_clip={get(cfg, A + '.optim.grad_clip')!r}")
    return "; ".join(found) or None


def _f_ckpt_keep(cfg, ctx):
    bad = [f"{k}={get(cfg, k)!r}" for k in ("trainer.max_actor_ckpt_to_keep", "trainer.max_critic_ckpt_to_keep")
           if get(cfg, k) not in (None, ABSENT)]
    return "; ".join(bad) or None


def _f_kl_in_reward(cfg, ctx):
    return "algorithm.use_kl_in_reward=true" if get(cfg, "algorithm.use_kl_in_reward") is True else None


def _f_dapo(cfg, ctx):
    return "algorithm.filter_groups.enable=true" if get(cfg, "algorithm.filter_groups.enable") is True else None


def _kept_groups(cfg):
    if filter_on(cfg):
        target = get(cfg, "filter.filter_kwargs.target_groups")
        if isinstance(target, int):
            return target
    return get(cfg, "data.train_batch_size")


def _f_shortcut(cfg, ctx):
    if get(cfg, "vagen.onpolicy_shortcut") is not True:
        return None
    epochs, mini, kept = get(cfg, A + ".ppo_epochs"), get(cfg, A + ".ppo_mini_batch_size"), _kept_groups(cfg)
    if epochs != 1 or mini != kept:
        return f"on-policy shortcut with ppo_epochs={epochs}, ppo_mini_batch_size={mini}, kept groups={kept}"
    return None


def _f_turn_gae_rs(cfg, ctx):
    if get(cfg, "algorithm.adv_estimator") == "turn_gae" and \
            get(cfg, "algorithm.rollout_correction.rollout_rs") not in (None, ABSENT):
        return f"turn_gae with rollout_rs={get(cfg, 'algorithm.rollout_correction.rollout_rs')!r}"
    return None


def _f_parameter_sync_step(cfg, ctx):
    """The active mode's parameter_sync_step, read as V1 reads it (absent = 1, TB:146), must be 1.

    DESIGN 3.1 names separate_async (its generated default is 4); the same holds for any mode, because a value
    above 1 splits each step's batch into local updates (TB:562-576), merges their metrics (V1/utils.py:84-102)
    and multiplies the learning-rate horizon (TB:804).
    """
    mode = get(cfg, "trainer.v1.trainer_mode")
    if not isinstance(mode, str):
        return None
    psync = get(cfg, f"trainer.v1.{mode}.parameter_sync_step")
    if psync is ABSENT:
        return None
    if not values_equal(psync, 1):
        return f"trainer_mode {mode} with trainer.v1.{mode}.parameter_sync_step={psync!r}"
    return None


def _f_epochs_short(cfg, ctx):
    if ctx.endpoint is None:
        return None
    epochs = get(cfg, "trainer.total_epochs")
    if ctx.steps_per_epoch is None or not isinstance(epochs, int):
        return f"cannot tell: total_epochs={epochs!r}, steps_per_epoch={ctx.steps_per_epoch!r}"
    if epochs * ctx.steps_per_epoch < ctx.endpoint:
        return (f"total_epochs {epochs} x steps_per_epoch {ctx.steps_per_epoch} = {epochs * ctx.steps_per_epoch}"
                f" < endpoint {ctx.endpoint}")
    return None


def _f_endpoint_horizon(cfg, ctx):
    if ctx.endpoint is None:
        return None
    horizon = get(cfg, "trainer.total_training_steps")
    if not isinstance(ctx.endpoint, int) or ctx.endpoint <= 0:
        return f"endpoint {ctx.endpoint!r} is not a positive integer"
    if not isinstance(horizon, int) or ctx.endpoint > horizon:
        return f"endpoint {ctx.endpoint} beyond the learning-rate horizon trainer.total_training_steps={horizon!r}"
    return None


def _f_partial_rollout(cfg, ctx):
    """No async_training section (verl's default, V1 never reads it), or one with partial_rollout true.

    LS:347 tests ``hasattr(config, "async_training") and not config.async_training.partial_rollout``: a section
    without the key (or with null) counts as false there, so an aborted turn is not resumed and the agent loop raises.
    """
    section = get(cfg, "async_training")
    if section is ABSENT:
        return None
    value = section.get("partial_rollout", ABSENT) if isinstance(section, dict) else ABSENT
    if value is True:
        return None
    if value is False:
        return "async_training.partial_rollout=false"
    return (f"async_training present without partial_rollout: true (partial_rollout={jsonable(value)!r}; "
            f"LS treats it as false)")


def _f_thinking(cfg, ctx):
    kwargs = get(cfg, "data.apply_chat_template_kwargs")
    if not isinstance(kwargs, dict):
        return f"data.apply_chat_template_kwargs={kwargs!r} (needs enable_thinking: false)"
    bad = []
    if kwargs.get("enable_thinking", ABSENT) is not False:
        bad.append(f"enable_thinking={jsonable(kwargs.get('enable_thinking', ABSENT))!r}")
    if "preserve_thinking" in kwargs:
        bad.append(f"preserve_thinking={kwargs['preserve_thinking']!r}")
    return "; ".join(bad) or None


def _f_kl_factor(cfg, ctx):
    mini, n, ga = get(cfg, A + ".ppo_mini_batch_size"), get(cfg, R + ".n"), get(cfg, "vagen.ga_8b")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (mini, n, ga)) or ga <= 0:
        return f"ppo_mini_batch_size={mini!r}, rollout.n={n!r}, vagen.ga_8b={ga!r}"
    if mini * n != KL_8B_FACTOR * ga:
        return f"(ppo_mini_batch_size {mini} x rollout.n {n}) / ga_8b {ga} = {mini * n / ga:g}, not {KL_8B_FACTOR}"
    return None


def _f_parity_override(cfg, ctx):
    override = get(cfg, "vagen.parity_sampling_override")
    if not parity_mode(cfg) and override not in (None, ABSENT) and override != {}:
        return f"vagen.parity_sampling_override={override!r} without vagen.parity_mode"
    return None


def _f_dynamic_bsz(cfg, ctx):
    keys = (A + ".use_dynamic_bsz", "actor_rollout_ref.ref.log_prob_use_dynamic_bsz", R + ".log_prob_use_dynamic_bsz")
    bad = [k for k in keys if get(cfg, k) is True]
    return ("dynamic batch on: " + ", ".join(bad)) if bad else None


@dataclass(frozen=True)
class Forbid:
    name: str
    test: Callable[[DictConfig, Context], Optional[str]]
    reason: str


FORBIDDEN: List[Forbid] = [
    Forbid("top_p_below_1_with_tis", _f_top_p_tis, "processed logprobs are renormalised in the nucleus: TIS biased"),
    Forbid("actor_grad_clip", _f_grad_clip, "the new engine reads actor.optim.clip_grad only"),
    Forbid("max_ckpt_to_keep", _f_ckpt_keep, "the workers would delete the milestones' HF weights (i4 2a)"),
    Forbid("use_kl_in_reward", _f_kl_in_reward, "the 8B had it off; turn_gae refuses it"),
    Forbid("filter_groups_enable", _f_dapo, "group selection belongs to the sampler"),
    Forbid("onpolicy_shortcut_shape", _f_shortcut, "the shortcut equals decoupled PPO only for one epoch, one mini-batch"),
    Forbid("turn_gae_with_rollout_rs", _f_turn_gae_rs, "rollout_rs changes response_mask"),
    Forbid("parameter_sync_step_not_1", _f_parameter_sync_step,
           "metrics would merge across local steps and the LR horizon would be multiplied"),
    Forbid("epochs_short_of_endpoint", _f_epochs_short, "fit would stop before the endpoint and save no checkpoint"),
    Forbid("endpoint_beyond_horizon", _f_endpoint_horizon, "the endpoint must lie within the LR horizon"),
    Forbid("partial_rollout_false", _f_partial_rollout,
           "an aborted turn would not be resumed (LS:347) and the agent loop raises on it (gym_agent_loop_v1.py:406-409)"),
    Forbid("thinking_template", _f_thinking, "thinking would eat the 384-token turn (TEMPLATE_CHECK 1, MF9)"),
    Forbid("kl_8b_factor", _f_kl_factor, "the KL conversion to the 8B unit assumes (mini x n) / ga_8b = 8 (MF1)"),
    Forbid("parity_override_outside_parity", _f_parity_override, "the sampling override is for the parity run only"),
    Forbid("dynamic_bsz", _f_dynamic_bsz, "aggregate_dp may fail with uneven micro-batches; off until S10 passes"),
]


# ---------------------------------------------------------------------------------------------------------------
# Evaluation


def rule_names() -> List[str]:
    """Every name vagen.contract.allow_override may list."""
    paths = {f"path:{k}" for k in PATH_KEYS} | {"path:filter.filter_kwargs.rows"}
    return sorted({e.key for e in EXPECTED} | {f.name for f in FORBIDDEN} | paths)


def _path_values(cfg: DictConfig) -> Dict[str, Any]:
    values = {k: get(cfg, k) for k in PATH_KEYS}
    if _keep_by_std(cfg):
        values["filter.filter_kwargs.rows"] = get(cfg, "filter.filter_kwargs.rows")
    return values


def evaluate(cfg: DictConfig, endpoint: Optional[int] = None, *, check_paths: bool = True,
             read_files: bool = True) -> dict:
    """Every rule's result as a JSON-safe dict; ``report["passed"]`` is False when a rule fails unoverridden.

    ``read_files=False`` skips reading the env yamls (no dataset size: the rules that need it fail with
    "cannot tell"); ``check_paths=False`` skips the existence checks (for configs built in tests).
    """
    ctx = build_context(cfg, endpoint, read_files=read_files)
    allow = get(cfg, "vagen.contract.allow_override")
    allow = [] if allow in (None, ABSENT) else list(allow) if isinstance(allow, (list, tuple)) else [allow]
    known = set(rule_names())
    # a file that cannot be read shows up in the rules that need it (they fail with "cannot tell" or <absent>);
    # the reason is kept under derived.errors
    violations, overridden, problems = [], [], []
    for name in allow:
        if name not in known:
            problems.append(f"vagen.contract.allow_override names an unknown rule: {name!r}")

    expected = []
    for rule in EXPECTED:
        applies = bool(rule.when(cfg))
        entry = {"key": rule.key, "source": rule.source, "applies": applies}
        if applies:
            try:
                actual = rule.actual(cfg, ctx)
                ok = rule.ok(actual, cfg, ctx)
                entry.update(expected=rule.describe(cfg, ctx), actual=jsonable(actual), ok=ok)
            except Exception as e:  # an interpolation that does not resolve, a file that does not load
                entry.update(expected=jsonable(None), actual=f"<error {type(e).__name__}: {e}>", ok=False)
            if not entry["ok"]:
                entry["overridden"] = rule.key in allow
                (overridden if entry["overridden"] else violations).append(rule.key)
        expected.append(entry)

    forbidden = []
    for rule in FORBIDDEN:
        try:
            detail = rule.test(cfg, ctx)
        except Exception as e:
            detail = f"<error {type(e).__name__}: {e}>"
        entry = {"name": rule.name, "reason": rule.reason, "violated": detail is not None, "detail": detail}
        if detail is not None:
            entry["overridden"] = rule.name in allow
            (overridden if entry["overridden"] else violations).append(rule.name)
        forbidden.append(entry)

    paths = []
    if check_paths:
        for key, value in _path_values(cfg).items():
            try:
                from vagen.q38.dataset import single_data_file
                path = single_data_file(value) if key.startswith("data.") else value
                exists = isinstance(path, str) and os.path.exists(os.path.expanduser(path))
            except Exception:
                path, exists = value, False
            name = f"path:{key}"
            entry = {"name": name, "path": jsonable(path), "exists": exists}
            if not exists:
                entry["overridden"] = name in allow
                (overridden if entry["overridden"] else violations).append(name)
            paths.append(entry)

    report = {
        "contract_version": CONTRACT_VERSION,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": endpoint,
        "profile": {"trainer_impl": trainer_impl(cfg), "parity_mode": parity_mode(cfg),
                    "trainer_mode": jsonable(get(cfg, "trainer.v1.trainer_mode")), "filter": filter_on(cfg)},
        "derived": {"run_dir": None if ctx.run_dir is None else str(ctx.run_dir), "dataset_size": ctx.dataset_size,
                    "steps_per_epoch": ctx.steps_per_epoch, "env_specs": ctx.env_specs, "errors": ctx.errors},
        "passed": not violations and not problems,
        "violations": violations,
        "overridden": overridden,
        "problems": problems,
        "allow_override": allow,
        "expected": expected,
        "forbidden": forbidden,
        "paths": paths,
        "info": {k: jsonable(_safe_get(cfg, k)) for k in INFO_KEYS},
    }
    return report


def _safe_get(cfg, key):
    try:
        return get(cfg, key)
    except Exception as e:
        return f"<error {type(e).__name__}: {e}>"


def summary(report: dict) -> str:
    lines = [f"[q38 contract] {'PASSED' if report['passed'] else 'FAILED'}: profile {report['profile']}, "
             f"endpoint {report['endpoint']}, dataset {report['derived']['dataset_size']}, "
             f"steps/epoch {report['derived']['steps_per_epoch']}"]
    for e in report["expected"]:
        if e["applies"] and not e["ok"]:
            tag = "overridden" if e.get("overridden") else "VIOLATION"
            lines.append(f"  {tag} {e['key']}: expected {e['expected']}, actual {e['actual']} ({e['source']})")
    for f in report["forbidden"]:
        if f["violated"]:
            tag = "overridden" if f.get("overridden") else "FORBIDDEN"
            lines.append(f"  {tag} {f['name']}: {f['detail']} ({f['reason']})")
    for p in report["paths"]:
        if not p["exists"]:
            lines.append(f"  {'overridden' if p.get('overridden') else 'MISSING'} {p['name']}: {p['path']}")
    for problem in report["problems"]:
        lines.append(f"  PROBLEM {problem}")
    for error in report["derived"]["errors"]:
        lines.append(f"  NOTE {error}")
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_outputs(cfg: DictConfig, report: dict, run_dir: Path) -> Dict[str, str]:
    """Write <run>/contract_check.json and <run>/effective_config.yaml (each atomically)."""
    run_dir = Path(run_dir)
    try:
        effective = OmegaConf.to_yaml(cfg, resolve=True)
        report["effective_config_resolved"] = True
    except Exception as e:  # keep the unresolved text rather than nothing
        effective = OmegaConf.to_yaml(cfg, resolve=False)
        report["effective_config_resolved"] = f"no: {type(e).__name__}: {e}"
    paths = {"contract_check": str(run_dir / "contract_check.json"),
             "effective_config": str(run_dir / "effective_config.yaml")}
    _atomic_write_text(run_dir / "effective_config.yaml", effective)
    _atomic_write_text(run_dir / "contract_check.json", json.dumps(report, indent=1, sort_keys=False) + "\n")
    return paths


def check(cfg: DictConfig, endpoint: Optional[int], *, check_paths: bool = True, write: bool = True,
          run_dir: Optional[Path] = None) -> dict:
    """Evaluate the contract, write the two files into the run directory, raise ContractError on a failure."""
    report = evaluate(cfg, endpoint, check_paths=check_paths)
    target = Path(run_dir) if run_dir is not None else (
        Path(report["derived"]["run_dir"]) if report["derived"]["run_dir"] else None)
    if write:
        if target is None:
            report["passed"] = False
            report["problems"].append("trainer.default_local_dir is not set: no run directory for the contract files")
        else:
            report["files"] = write_outputs(cfg, report, target)
    if not report["passed"]:
        where = f" (details: {target / 'contract_check.json'})" if write and target is not None else ""
        raise ContractError(summary(report) + where)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", type=int, default=None)
    ap.add_argument("--write", action="store_true", help="also write contract_check.json / effective_config.yaml")
    ap.add_argument("--no-paths", action="store_true", help="skip the existence checks of the input files")
    args = ap.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    try:
        report = check(cfg, args.endpoint, check_paths=not args.no_paths, write=args.write)
    except ContractError as e:
        print(e)
        return 2
    print(summary(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
