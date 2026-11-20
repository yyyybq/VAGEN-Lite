"""T11 (docs/q38/DESIGN_ASYNC.md 5.1, MF12): the complete V1 config ``training_prep rl-config --stack q38`` writes.

The q38 configs of the 8B main line's V4 and S0 runs are generated from the same inputs and options as the 8B runs
(their rl-config command lines in R8/logs/rl_config_qw3v4s3.log, rl_config_qw3s0s3.log) and compared key by key with
the 8B effective configs (Y8 = R8/rl_qw3v4s3/ppo.yaml, and R8/rl_qw3s0s3/ppo.yaml), both resolved.  Every difference
must be registered below: a recipe change R1-R13 of DESIGN_ASYNC 3.3, or the software stack (a key renamed, dropped
or new in V1, the profiler, the reward-model sections).  Also: the learning rate the V1 scheduler applies at step 100
is the 8B one (1.9077617e-06), the 8B path without --stack still writes Y8, the delta file only holds keys that
differ from verl's default, the V1 dataclasses accept the sections, and vagen.q38.contract passes the configs.

Reads the 8B run directory R8 (read only); writes the generated configs to a temporary folder (Q38_TEST_TMP, or
/mnt/umm/users/yinbaiqiao/q38/tmp_tests, else pytest's) and removes it afterwards.
"""

from __future__ import annotations

import argparse
import copy
import fnmatch
import math
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from data_gen.interiorgs_v10 import training_prep as tp

REPO = Path(__file__).resolve().parents[3]
E8 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial")
R8 = E8 / "interiorgs_v10_train_q3s3_20261027"
MODEL8 = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232"
ROWS8 = E8 / "interiorgs_v10_r11_20261016" / "rl_rows" / "rows.jsonl"
SFT8 = Path("/mnt/umm/users/yinbaiqiao/models_runs/sft_v10_newformat_20261024/sft_trajectories.jsonl")
EXCLUDE8 = E8 / "interiorgs_v10_r10_20261015" / "depth_check" / "depth_check_summary.json"
TMP_ROOT = Path("/mnt/umm/users/yinbaiqiao/q38/tmp_tests")
ENDPOINT = 300  # the main line's endpoint (R8/plan.json)

# the 8B runs' rl-config options (R8/logs/rl_config_<run>.log), without --rows / --run-root / --out
COMMON = ["--gs-root", "/mnt/umm/users/yinbaiqiao/InteriorGS_ply", "--model", MODEL8, "--val-limit", "64",
          "--seed", "20261022", "--format-example", "action_names", "--model-type", "qwen3_vl", "--task-pool", "reserve",
          "--sft-trajectories", str(SFT8), "--exclude", str(EXCLUDE8)]
RUNS = {
    "v4": {"y8": R8 / "rl_qw3v4s3", "run8": R8 / "rl_run_qw3v4s3", "name": "v10_rl_q3s3_qw3v4s3",
           "args": ["--standard", "v10_p1_Dcond", "--group-filter", "keep_by_return_std", "--keep-std-threshold",
                    "0.3318", "--keep-std-floor", "1.0", "--progress-reward-scale", "3.3333333333333335"]},
    "s0": {"y8": R8 / "rl_qw3s0s3", "run8": R8 / "rl_run_qw3s0s3", "name": "v10_rl_q3s3_qw3s0s3",
           "args": ["--standard", "v10_p1_S0"]},
}
INPUTS = [ROWS8, SFT8, EXCLUDE8, Path(MODEL8) / "config.json", tp.Q38_VERL_DEFAULT] + \
    [r["y8"] / "ppo.yaml" for r in RUNS.values()]
if not all(p.exists() for p in INPUTS):
    pytest.skip(f"T11 needs the 8B run inputs and the q38 verl tree: {[str(p) for p in INPUTS if not p.exists()]}",
                allow_module_level=True)

# ---------------------------------------------------------------------------------------------------------------
# The register of the differences to the 8B configs: (pattern on the dotted key, class, note).  Classes: R1-R13
# (DESIGN_ASYNC 3.3) or stack:* (the software stack, DESIGN_ASYNC 3.3 "另有不算配方改动"); the first match counts.
# 8B keys the q38 config does not have
DROPPED = [
    ("actor_rollout_ref.actor.grad_clip", "stack:rename", "-> actor.optim.clip_grad, the only clip the V1 engine reads "
                                                          "(transformer_impl.py:859-876)"),
    ("critic.model.fsdp_config.*", "stack:rename", "-> critic.fsdp.* (V1 critic schema)"),
    ("actor_rollout_ref.rollout.update_weights_bucket_megabytes", "stack:rename",
     "-> rollout.checkpoint_engine.update_weights_bucket_megabytes"),
    ("trainer.replace_image_tokens_for_logging", "stack:rename",
     "-> trainer.v1.sampler.sampler_kwargs.replace_image_tokens_for_logging (the sampler writes the dump)"),
    ("*.profiler.*", "stack:profiler", "profiler schema of 0.6.1 vs V1; disabled in both"),
    ("actor_rollout_ref.actor.nfp_loss_coef", "stack:drop",
     "the 8B fork's auxiliary-loss weight (Cambrian / U1); only acts with an aux loss, Qwen has none"),
    ("actor_rollout_ref.actor.image_gen_loss_coef", "stack:drop", "as nfp_loss_coef (null)"),
    ("actor_rollout_ref.ref.model", "stack:drop", "null; not a V1 key"),
    ("actor_rollout_ref.rollout.multi_turn.interaction_config_path", "stack:drop", "null; not a V1 key"),
    ("actor_rollout_ref.rollout.skip_rollout", "stack:drop", "off; V1 has the skip.* section"),
    ("actor_rollout_ref.rollout.skip_dump_dir", "stack:drop", "unused (skip_rollout off)"),
    ("algorithm.rollout_correction.rollout_rs_threshold_lower", "stack:drop", "0.6.1 rollout-correction key, null"),
    ("algorithm.rollout_correction.rollout_token_veto_threshold", "stack:drop", "0.6.1 rollout-correction key, null"),
    ("algorithm.rollout_correction.use_policy_gradient", "stack:drop", "0.6.1 key, false (V1: loss_type ppo_clip)"),
    ("data.datagen.*", "stack:drop", "null; no dataset generator in V1"),
    ("data.sampler.*", "stack:drop", "null; removed (DESIGN 2.2): the step sampler draws the tasks"),
    ("huggingface_hub.*", "stack:drop", "the 8B trainer's HF upload, unused"),
    ("reward_model.*", "stack:reward", "legacy reward-model section; no reward model in either stack (the env reward "
                                       "comes through the agent loop)"),
    ("trainer.ckpt_milestones", "stack:drop", "the 8B trainer's milestone list (RT:2289), empty; the callback keeps "
                                              "every ckpt_keep_every-th"),
    ("trainer.concat_multi_turn", "stack:drop", "the 8B trainer's switch; the V1 agent loop is one sequence an episode"),
    ("trainer.log_image.*", "stack:drop", "the 8B trainer's image logging, off"),
    ("trainer.skip_pad_tokens", "stack:drop", "the 8B trainer's dump option"),
    ("trainer.use_legacy_worker_impl", "stack:drop", "0.6.1 worker switch; V1 has the engine workers only"),
]
# keys of both configs whose values differ
CHANGED = [
    ("actor_rollout_ref.actor.loss_agg_mode", "stack:rename",
     "token-mean -> seq-mean-token-mean: the 8B update's actual objective in V1's terms (DESIGN 3.1)"),
    ("critic.loss_agg_mode", "stack:rename", "interpolates actor.loss_agg_mode; no critic under GRPO"),
    ("actor_rollout_ref.actor.optim.clip_grad", "stack:rename", "1.0 (unread by the 8B engine) -> 0.3, the 8B's "
                                                                "effective actor.grad_clip"),
    ("*.profiler.*", "stack:profiler", "disabled in both"),
    ("actor_rollout_ref.ref.fsdp_config.forward_only", "stack:change", "the V1 reference engine is forward-only"),
    ("actor_rollout_ref.rollout.agent.agent_loop_config_path", "stack:change", "GymAgentLoopV1 (DESIGN 2.5.3)"),
    ("actor_rollout_ref.rollout.agent.default_agent_loop", "stack:change",
     "gym_agent, the rows' agent_name (8B's single_turn_agent was never used)"),
    ("data.custom_cls.*", "stack:change", "AgenticDatasetV1 (DESIGN 2.1)"),
    ("critic.model._target_", "stack:change", "V1's critic model config class"),
    ("custom_reward_function.name", "stack:reward", "V1 keeps the reward function under reward.*"),
    ("reward_model.*", "stack:reward", "legacy section, all null in V1"),
    ("actor_rollout_ref.rollout.calculate_log_probs", "R2", "token TIS needs the rollout log-probs"),
    ("algorithm.rollout_correction.rollout_is", "R2", "decoupled PPO + token TIS"),
    ("actor_rollout_ref.rollout.top_p", "R3", "0.92 -> 1.0 (top-p < 1 biases TIS)"),
    ("trainer.max_actor_ckpt_to_keep", "stack:change", "2 -> null: retention in the checkpoint callback (DESIGN 2.5.5)"),
    ("trainer.max_critic_ckpt_to_keep", "stack:change", "2 -> null, as max_actor_ckpt_to_keep"),
    ("trainer.rollout_data_dir", "stack:rename", "-> sampler_kwargs.rollout_data_dir (the sampler writes the dump)"),
    ("trainer.total_epochs", "stack:change", "30 -> ceil(700 / steps_per_epoch) + 1 (DESIGN 3.1)"),
]
# keys only the q38 config has, whose (unresolved) value is not verl's default
NEW = [
    ("actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes", "stack:rename", "<- 8B rollout key"),
    ("critic.fsdp.*", "stack:rename", "<- critic.model.fsdp_config.*"),
    ("trainer.v1.sampler.sampler_kwargs.rollout_data_dir", "stack:rename", "<- trainer.rollout_data_dir"),
    ("trainer.v1.sampler.sampler_kwargs.replace_image_tokens_for_logging", "stack:rename",
     "<- trainer.replace_image_tokens_for_logging"),
    ("trainer.v1.sampler.sampler_kwargs.*", "R1", "the step sampler's arguments, interpolated (DESIGN 2.5.1)"),
    ("trainer.v1.trainer_mode", "R1", "colocate_async"),
    ("trainer.v1.sampler.max_off_policy_*", "R1", "staleness gate 2 + wait"),
    ("trainer.v1.sampler.custom_sampler.*", "R1", "the step-batch sampler"),
    ("data.gen_batch_size", "R1", "async V1 draws one task at a time (TB:754-766)"),
    ("vagen.async.strict_version_check", "R1", "version-tag assertion (MF3)"),
    ("vagen.ga_8b", "R6", "KL guard in the 8B unit (MF1)"),
    ("vagen.failure_policy", "R7", "failed groups raise (question 1)"),
    ("vagen.stuck_warn.*", "R8", "stuck warning (MF5)"),
    ("vagen.calib_entropy_t1_steps", "R5", "entropy calibration (question 2)"),
    ("vagen.host_mem_log.interval_s", "stack:new",
     "host-memory log every 10 s (vagen/q38/host_mem_log.py), read by the chain's host-memory guard (P1 decision 7)"),
    ("vagen.onpolicy_shortcut", "R11", "off until S9"),
    ("vagen.parity_mode", "stack:new", "true only for the 8B parity run"),
    ("vagen.trainer_impl", "stack:new", "runner switch"),
    ("vagen.contract.allow_override", "stack:new", "contract overrides, none"),
    ("actor_rollout_ref.rollout.limit_images", "stack:new", "13 explicit (MF8); 8B left vLLM's default"),
    ("trainer.checkpoint_callback_class", "stack:new", "retention callback (DESIGN 2.5.5)"),
    ("trainer.ckpt_keep_latest", "stack:new", "the 8B's 2 latest kept, now the callback's"),
]
# 8B key -> q38 key holding the same value
RENAMED_VALUES = {"actor_rollout_ref.actor.grad_clip": "actor_rollout_ref.actor.optim.clip_grad",
                  "actor_rollout_ref.rollout.update_weights_bucket_megabytes":
                      "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes",
                  "trainer.rollout_data_dir": "trainer.v1.sampler.sampler_kwargs.rollout_data_dir",
                  "trainer.replace_image_tokens_for_logging":
                      "trainer.v1.sampler.sampler_kwargs.replace_image_tokens_for_logging"}
# learning rates of the 8B V4 run (R8/rl_run_qw3v4s3/logs/rl-20261027T222142Z-...-worker-0.log, actor/lr at step:N)
LR8 = {1: 2e-06, 2: 1.9999904325424047e-06, 100: 1.907761700699954e-06}


def _flat(node, prefix=""):
    return tp._q38_flat(node, prefix)


def _norm(value, subs):
    if isinstance(value, str):
        for old, new in subs:
            value = value.replace(old, new)
    elif isinstance(value, list):
        value = [_norm(v, subs) for v in value]
    return value


def _resolved_flat(cfg: dict, subs) -> dict:
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True, throw_on_missing=False)
    return {k: _norm(v, subs) for k, v in _flat(data).items()}


def _match(key: str, table):
    for pattern, cls, note in table:
        if fnmatch.fnmatchcase(key, pattern):
            return pattern, cls, note
    return None


def _at_default(key: str, raw: dict, default: dict) -> bool:
    """Whether a resolved key comes unchanged from verl's default: the nearest key of the unresolved config at or
    above it (a node interpolation such as rollout.mtp resolves into sub-keys) holds the default's value."""
    parts = key.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in raw:
            return prefix in default and default[prefix] == raw[prefix]
    return False


def _differences(c: dict):
    """(only in 8B, changed, new off verl's default) keys of one run, resolved and with the run folders mapped."""
    y = _resolved_flat(c["y8"], c["subs8"])
    q = _resolved_flat(c["cfg"], c["subs"])
    default = _flat(yaml.safe_load(tp.Q38_VERL_DEFAULT.read_text()))
    raw = _flat(c["cfg"])
    dropped = sorted(set(y) - set(q))
    changed = sorted(k for k in set(y) & set(q) if y[k] != q[k])
    new = sorted(k for k in set(q) - set(y) if not _at_default(k, raw, default))
    return y, q, dropped, changed, new


def _rl_config(out: Path, run: Path, spec: dict, stack: bool) -> dict:
    argv = ["rl-config", "--rows", str(ROWS8), "--name", spec["name"], "--run-root", str(run), "--out", str(out),
            *COMMON, *spec["args"]] + (["--stack", "q38"] if stack else [])
    assert tp.main(argv) == 0
    return yaml.safe_load((out / "ppo.yaml").read_text())


@pytest.fixture(scope="module")
def work():
    base = os.environ.get("Q38_TEST_TMP") or (str(TMP_ROOT) if TMP_ROOT.parent.is_dir() else None)
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="test_config_", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def configs(work):
    """{run: {"cfg": the q38 config, "subs": path substitutions, "y8": the 8B config, "out": folder}}."""
    out = {}
    for name, spec in RUNS.items():
        rl, run = work / f"rl_{name}", work / f"run_{name}"
        cfg = _rl_config(rl, run, spec, stack=True)
        # the run's own folders: the 8B run's and this test's map to the same tokens
        subs8 = [(str(spec["run8"]), "<RUN>"), (str(spec["y8"]), "<OUT>")]
        subs = [(str(run.resolve()), "<RUN>"), (str(rl.resolve()), "<OUT>")]
        y8 = yaml.safe_load((spec["y8"] / "ppo.yaml").read_text())
        out[name] = {"cfg": cfg, "subs": subs, "y8": y8, "subs8": subs8, "out": rl, "run": run}
    return out


# ---------------------------------------------------------------------------------------------------------------


def test_8b_path_unchanged(work):
    """Without --stack the 8B path writes Y8 itself (only the run's folders differ)."""
    spec = RUNS["v4"]
    rl, run = work / "rl_8b", work / "run_8b"
    _rl_config(rl, run, spec, stack=False)
    mine = _norm((rl / "ppo.yaml").read_text(), [(str(run.resolve()), "<RUN>"), (str(rl.resolve()), "<OUT>")])
    theirs = _norm((spec["y8"] / "ppo.yaml").read_text(), [(str(spec["run8"]), "<RUN>"), (str(spec["y8"]), "<OUT>")])
    assert mine == theirs
    report = yaml.safe_load((rl / "rl_config_report.json").read_text())
    assert "stack" not in report and "q38" not in report


@pytest.mark.parametrize("name", sorted(RUNS))
def test_full_diff_registered(configs, name):
    """Every key-by-key difference to the 8B config is registered (MF12)."""
    c = configs[name]
    y, q, dropped, changed, new = _differences(c)
    problems = [f"8B key missing in q38, unregistered: {k} = {y[k]!r}" for k in dropped if not _match(k, DROPPED)]
    problems += [f"value changed, unregistered: {k}: 8B {y[k]!r} -> q38 {q[k]!r}" for k in changed
                 if not _match(k, CHANGED)]
    problems += [f"new q38 key off verl's default, unregistered: {k} = {q[k]!r}" for k in new if not _match(k, NEW)]
    assert not problems, "\n".join(problems)
    # renamed keys keep the 8B values
    for k8, kq in RENAMED_VALUES.items():
        assert q[kq] == y[k8], (k8, kq)
    for k in (k for k in y if k.startswith("critic.model.fsdp_config.")):
        assert q["critic.fsdp." + k[len("critic.model.fsdp_config."):]] == y[k], k
    # the run's own keys are the 8B run's
    for k in ("data.train_files", "data.val_files", "data.seed", "actor_rollout_ref.model.path", "filter.name",
              "trainer.experiment_name", "trainer.default_local_dir", "actor_rollout_ref.actor.optim.lr"):
        assert q[k] == y[k], k


def test_register_entries_used(configs):
    """No register entry is stale: each one explains a difference of at least one of the two runs."""
    used = set()
    for c in configs.values():
        _, _, dropped, changed, new = _differences(c)
        for tag, keys, table in (("DROPPED", dropped, DROPPED), ("CHANGED", changed, CHANGED), ("NEW", new, NEW)):
            used |= {(tag, m[0]) for m in (_match(k, table) for k in keys) if m}
    stale = [(t, p) for t, table in (("DROPPED", DROPPED), ("CHANGED", CHANGED), ("NEW", NEW))
             for p, _, _ in table if (t, p) not in used]
    assert not stale, stale


@pytest.mark.parametrize("name", sorted(RUNS))
def test_recipe_keys(configs, name):
    """DESIGN_ASYNC 2.2 / 3.1: the keys rl-config --stack q38 must get right."""
    c = configs[name]
    cfg, q = c["cfg"], _resolved_flat(c["cfg"], c["subs"])
    actor = cfg["actor_rollout_ref"]["actor"]
    assert "grad_clip" not in actor and actor["optim"]["clip_grad"] == 0.3
    assert actor["optim"].get("grad_clip") is None
    assert "fsdp_config" not in cfg["critic"]["model"] and cfg["critic"]["fsdp"]["param_offload"] is True
    assert "sampler" not in cfg["data"]
    expected = {
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "cosine", "actor_rollout_ref.actor.optim.min_lr_ratio": 0.05,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0, "actor_rollout_ref.actor.optim.lr_warmup_steps": -1,
        "actor_rollout_ref.actor.optim.weight_decay": 0.01, "actor_rollout_ref.actor.optim.lr": 2e-6,
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean", "trainer.total_training_steps": 700,
        "trainer.use_v1": True, "trainer.v1.trainer_mode": "colocate_async",
        "trainer.v1.colocate_async.num_warmup_batches": 1, "trainer.v1.sampler.max_off_policy_threshold": 2,
        "trainer.v1.sampler.max_off_policy_strategy": "wait", "trainer.v1.sampler.sync_refill_failed_groups": False,
        "trainer.v1.sampler.custom_sampler.path": "pkg://vagen.q38.sampler",
        "trainer.v1.sampler.custom_sampler.name": "VagenStepBatchSampler",
        "algorithm.rollout_correction.rollout_is": "token", "algorithm.rollout_correction.rollout_is_threshold": 2.0,
        "algorithm.rollout_correction.rollout_rs": None, "algorithm.rollout_correction.bypass_mode": False,
        "algorithm.rollout_correction.rollout_is_batch_normalize": False,
        "algorithm.rollout_correction.score_centering": False, "algorithm.filter_groups.enable": False,
        "trainer.max_actor_ckpt_to_keep": None, "trainer.max_critic_ckpt_to_keep": None,
        "trainer.rollout_data_dir": None, "trainer.test_freq": -1, "trainer.val_before_train": False,
        "actor_rollout_ref.rollout.temperature": 0.8, "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.logprobs_mode": "processed_logprobs", "actor_rollout_ref.rollout.limit_images": 13,
        "actor_rollout_ref.rollout.prompt_length": 4096, "actor_rollout_ref.rollout.response_length": 4096,
        "data.apply_chat_template_kwargs.enable_thinking": False, "data.gen_batch_size": 1,
        "vagen.ga_8b": 12, "vagen.failure_policy": "raise", "vagen.async.strict_version_check": True,
        "vagen.onpolicy_shortcut": False, "vagen.parity_mode": False, "vagen.trainer_impl": "vagen",
        "trainer.ckpt_keep_every": 50, "trainer.ckpt_keep_latest": 2, "trainer.ckpt_prune_to_hf": True,
        "trainer.save_freq": 50, "vagen.host_mem_log.interval_s": 10,
    }
    wrong = {k: (q.get(k, "<absent>"), v) for k, v in expected.items() if q.get(k, "<absent>") != v}
    assert not wrong, wrong
    assert "preserve_thinking" not in cfg["data"]["apply_chat_template_kwargs"]
    assert "async_training" not in cfg
    # the sampler's arguments interpolate the values written once (DESIGN 3.1)
    sk = cfg["trainer"]["v1"]["sampler"]["sampler_kwargs"]
    assert sk["filter"] == "${filter}" and sk["model_path"] == "${actor_rollout_ref.model.path}"
    assert q["trainer.v1.sampler.sampler_kwargs.rollout_data_dir"] == "<RUN>/rollout_data"
    for k in ("name", "enable"):
        assert q[f"trainer.v1.sampler.sampler_kwargs.filter.{k}"] == q[f"filter.{k}"]
    assert q["trainer.v1.sampler.sampler_kwargs.candidates_per_step"] == 24
    assert q["trainer.v1.sampler.sampler_kwargs.rollout_n"] == 8
    assert q["trainer.v1.sampler.sampler_kwargs.adv_estimator"] == "grpo"
    assert q["trainer.v1.sampler.sampler_kwargs.stuck_warn.min_s"] == 300
    # the epochs: 10798 train rows // 24 = 449 steps an epoch, ceil(700 / 449) + 1 = 3 (>= the endpoint)
    report = yaml.safe_load((c["out"] / "rl_config_report.json").read_text())
    assert report["stack"] == "q38" and report["train"] == 10798
    assert report["q38"]["steps_per_epoch"] == 10798 // 24 == 449
    assert cfg["trainer"]["total_epochs"] == math.ceil(700 / 449) + 1 == 3
    assert cfg["trainer"]["total_epochs"] * 449 >= ENDPOINT
    assert report["q38"]["run_keys_over_recipe"] is None  # the 8B runs override nothing of the recipe
    # the group filter of the run
    if name == "v4":
        assert cfg["filter"] == {"name": "keep_by_return_std", "enable": True,
                                 "filter_kwargs": {"target_groups": 12, "std_threshold": 0.3318, "std_floor": 1.0,
                                                   "rows": str((c["out"] / "train.jsonl").resolve())}}
    else:
        assert cfg["filter"] == {"name": "mixed_outcome", "enable": True, "filter_kwargs": {"target_groups": 12}}


def test_delta_only_differs_from_default():
    """configs/ppo_q38_delta.yaml writes only keys whose value differs from verl's default, or keys it lacks."""
    default = _flat(yaml.safe_load(tp.Q38_VERL_DEFAULT.read_text()))
    delta = _flat(yaml.safe_load(tp.Q38_DELTA.read_text()))
    same = sorted(k for k, v in delta.items() if k in default and default[k] == v)
    assert not same, same


def test_lr_schedule_step_100(configs):
    """The learning rate the V1 scheduler applies at step 100 is the 8B run's 1.9077617e-06 (also the
    zero_indexed_step convention: counting from 1 gives a different value)."""
    import torch
    from omegaconf import OmegaConf

    from verl.utils.config import omega_conf_to_dataclass

    cfg = OmegaConf.create(configs["v4"]["cfg"])
    trainer = cfg.trainer
    psync = int(trainer.v1.get(trainer.v1.trainer_mode, {}).get("parameter_sync_step", 1))

    def rates(zero_indexed: bool):
        optim = copy.deepcopy(cfg.actor_rollout_ref.actor.optim)
        # TB:806-811: the scheduler horizon is trainer.total_training_steps x parameter_sync_step
        optim.total_training_steps = int(trainer.total_training_steps) * psync
        optim.zero_indexed_step = zero_indexed
        oc = omega_conf_to_dataclass(optim)
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=oc.lr, weight_decay=oc.weight_decay, betas=tuple(oc.betas))
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

        # the engine's own scheduler construction (transformer_impl.py:518-552)
        sched = FSDPEngine._build_lr_scheduler(SimpleNamespace(optimizer_config=oc, rank=1), opt)
        applied, logged = {}, {}
        for step in range(1, 102):
            applied[step] = opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
            logged[step] = sched.get_last_lr()[0]  # what lr_scheduler_step returns (transformer_impl.py:913-919)
        return applied, logged

    assert cfg.actor_rollout_ref.actor.optim.zero_indexed_step is True
    applied, logged = rates(True)
    for step, lr in LR8.items():
        assert math.isclose(applied[step], lr, rel_tol=1e-6), (step, applied[step], lr)
    assert math.isclose(applied[100], 1.9077617e-06, rel_tol=1e-6)
    # V1 logs actor/lr after the scheduler step: its step-s value is the rate step s+1 applies (the 8B log shows the
    # rate step s applied), a logging shift of one step registered with the software stack
    assert logged[100] == applied[101] and not math.isclose(logged[100], LR8[100], rel_tol=1e-6)
    applied_one, _ = rates(False)
    assert not math.isclose(applied_one[100], LR8[100], rel_tol=1e-6)


@pytest.mark.parametrize("name", sorted(RUNS))
def test_dataclasses(configs, name):
    """The V1 dataclasses accept the sections the workers and the trainer build (after TaskRunnerV1's resolve)."""
    from omegaconf import OmegaConf, open_dict

    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.utils.config import omega_conf_to_dataclass

    cfg = OmegaConf.create(configs[name]["cfg"])
    OmegaConf.resolve(cfg)  # main_ppo.py:144-145
    arr = cfg.actor_rollout_ref
    actor = omega_conf_to_dataclass(arr.actor)
    assert actor.optim.clip_grad == 0.3 and actor.optim.lr_scheduler_type == "cosine"
    assert actor.loss_agg_mode == "seq-mean-token-mean" and actor.ppo_mini_batch_size == 12
    # the reference worker's renames before its dataclass (verl/verl/workers/engine_workers.py:550-558)
    ref = OmegaConf.create(OmegaConf.to_container(arr.ref, resolve=True))
    with open_dict(ref):
        ref.ppo_mini_batch_size = arr.actor.ppo_mini_batch_size
        ref.ppo_micro_batch_size = ref.pop("log_prob_micro_batch_size", None)
        ref.ppo_micro_batch_size_per_gpu = ref.pop("log_prob_micro_batch_size_per_gpu", None)
        ref.use_dynamic_bsz = ref.pop("log_prob_use_dynamic_bsz", False)
        ref.ppo_max_token_len_per_gpu = ref.pop("log_prob_max_token_len_per_gpu", None)
    omega_conf_to_dataclass(ref)
    rollout = omega_conf_to_dataclass(arr.rollout)
    assert rollout.limit_images == 13 and rollout.top_p == 1.0 and rollout.calculate_log_probs is True
    assert rollout.enable_sleep_mode is True and rollout.free_cache_engine is True
    omega_conf_to_dataclass(arr.rollout.checkpoint_engine)
    omega_conf_to_dataclass(cfg.algorithm)
    rc = omega_conf_to_dataclass(cfg.algorithm.rollout_correction, dataclass_type=RolloutCorrectionConfig)
    assert rc.rollout_is == "token" and rc.rollout_is_threshold == 2.0 and rc.rollout_rs is None
    omega_conf_to_dataclass(cfg.critic.optim)


@pytest.mark.parametrize("name", sorted(RUNS))
def test_contract_passes(configs, name):
    """vagen.q38.contract accepts the generated configs at the main line's endpoint (evaluate: nothing written)."""
    try:
        from vagen.q38 import contract
    except ImportError as e:  # written in parallel (W1)
        pytest.skip(f"vagen.q38.contract not importable: {e}")
    from omegaconf import OmegaConf

    report = contract.evaluate(OmegaConf.create(configs[name]["cfg"]), ENDPOINT)
    assert report["passed"], contract.summary(report)


def _options(**over):
    base = dict(trial=False, curriculum_easy_updates=None, invalid_turn_nonpositive=False, progress_adv_weight=None,
                logprob_tokens=False, greedy_val=False, history_mode="concat", opt_package="p1", test_freq=None,
                no_internal_val=True)
    base.update(over)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("over", [
    {"trial": True}, {"curriculum_easy_updates": 50}, {"invalid_turn_nonpositive": True},
    {"progress_adv_weight": 0.5}, {"logprob_tokens": True}, {"greedy_val": True}, {"history_mode": "none"},
    {"history_mode": None}, {"opt_package": None}, {"opt_package": "p1gae"}, {"opt_package": "p1tgae"},
    {"test_freq": 25}, {"no_internal_val": False},
    # phase 1 is p1's shape (the contract's data.train_batch_size 24): p2's 48 candidates are phase 2
    {"opt_package": "p2"}, {"candidates": 48}, {"target_groups": 6}, {"rollouts": 4},
])
def test_refused_options(over):
    with pytest.raises(SystemExit):
        tp.q38_check_options(_options(**over))


def test_accepted_options():
    tp.q38_check_options(_options())
    tp.q38_check_options(_options(candidates=24, target_groups=12, rollouts=8))  # p1's own values, given explicitly


def test_refused_8b_configs(configs, work):
    """q38_ppo_config refuses an 8B config the V1 trainer would not run as the 8B one ran."""
    y8 = configs["v4"]["y8"]
    tp.q38_ppo_config(copy.deepcopy(y8), 10798)  # the 8B V4 config itself goes through
    with pytest.raises(SystemExit):
        tp.q38_ppo_config(copy.deepcopy(y8), 10798, type_overrides={"trainer.agent_loop_manager": "x"})
    with pytest.raises(SystemExit):
        tp.q38_ppo_config(copy.deepcopy(y8), 10798, default_path=work / "no_such_default.yaml")
    bad = copy.deepcopy(y8)
    bad["trainer"]["test_freq"] = 25
    with pytest.raises(SystemExit):
        tp.q38_ppo_config(bad, 10798)
    bad = copy.deepcopy(y8)
    bad["trainer"]["total_training_steps"] = 600
    with pytest.raises(SystemExit):
        tp.q38_ppo_config(bad, 10798)
    bad = copy.deepcopy(y8)
    del bad["trainer"]["rollout_data_dir"]
    with pytest.raises(SystemExit):
        tp.q38_ppo_config(bad, 10798)
    with pytest.raises(SystemExit):
        tp.q38_ppo_config(copy.deepcopy(y8), 20)  # no step of 24 tasks
    # P1 decision 7: a recipe without the host-memory log (the chain's host-memory guard reads it) is refused
    _, report = tp.q38_ppo_config(copy.deepcopy(y8), 10798)
    assert report["host_mem_log_interval_s"] == 10
    for interval in (None, 0, -5, True, "10"):
        delta = yaml.safe_load(tp.Q38_DELTA.read_text())
        if interval is None:
            delta["vagen"].pop("host_mem_log")
        else:
            delta["vagen"]["host_mem_log"]["interval_s"] = interval
        path = work / f"delta_host_mem_{interval}.yaml"
        path.write_text(yaml.safe_dump(delta))
        with pytest.raises(SystemExit, match="host_mem_log.interval_s"):
            tp.q38_ppo_config(copy.deepcopy(y8), 10798, delta_path=path)
    # a run's own override of a recipe value is taken and reported
    over = copy.deepcopy(y8)
    over["actor_rollout_ref"]["actor"]["optim"]["lr"] = 1e-6
    cfg, report = tp.q38_ppo_config(over, 10798)
    assert cfg["actor_rollout_ref"]["actor"]["optim"]["lr"] == 1e-6
    assert report["run_keys_over_recipe"] == {"actor_rollout_ref.actor.optim.lr": {"recipe": 2e-6, "run": 1e-6}}
