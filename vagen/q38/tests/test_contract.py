"""T13 (DESIGN_ASYNC 5.1): the configuration contract catches every forbidden combination.

Also: the hand-written 4B smoke config (vagen/q38/configs/smoke_4b_example.yaml) passes the contract and verl's own
start-up checks; EXPECTED rules, profiles (stock, parity) and allow_override; the files check() writes; and the
entry's endpoint handling and trainer selection (vagen/q38/ppo_v1.py, DESIGN 2.5.4).
"""

import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf, open_dict

from vagen.q38 import contract as C

REPO = Path(__file__).resolve().parents[3]
SMOKE = REPO / "vagen" / "q38" / "configs" / "smoke_4b_example.yaml"
R8 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_train_q3s3_20261027")
A = "actor_rollout_ref.actor"
R = "actor_rollout_ref.rollout"

N_ENVS = 4800  # synthetic train pool: steps_per_epoch 4800 // 24 = 200, total_epochs ceil(700 / 200) + 1 = 5
ENV_YAML = """\
envs:
- name: ActiveSpatial
  data_source: active_spatial
  max_turns: %d
  response_length_per_turn: 384
  config: {client_url: "${oc.env:R1_RENDER_URL}"}
  n_envs: %d
  seed: [0, %d]
"""


def put(cfg, key, value):
    with open_dict(cfg):
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)


def delete(cfg, key):
    parent, _, leaf = key.rpartition(".")
    node = OmegaConf.select(cfg, parent)
    with open_dict(cfg):
        del node[leaf]


class Base(unittest.TestCase):
    """A synthetic copy of the smoke config: small env yamls in a temporary directory, no path checks."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.train_env, cls.val_env = root / "train_env.yaml", root / "val_env.yaml"
        cls.train_env.write_text(ENV_YAML % (12, N_ENVS, N_ENVS))
        cls.val_env.write_text(ENV_YAML % (12, 64, 64))
        cls.run_dir = root / "run"
        cls.smoke = OmegaConf.load(SMOKE)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def base(self):
        cfg = OmegaConf.create(OmegaConf.to_container(self.smoke, resolve=False))
        put(cfg, "data.train_files", str(self.train_env))
        put(cfg, "data.val_files", str(self.val_env))
        put(cfg, "trainer.total_epochs", 5)
        put(cfg, "trainer.default_local_dir", str(self.run_dir / "checkpoints"))
        put(cfg, "trainer.v1.sampler.sampler_kwargs.rollout_data_dir", str(self.run_dir / "rollout_data"))
        return cfg

    def evaluate(self, cfg, endpoint=300):
        return C.evaluate(cfg, endpoint, check_paths=False)

    def assertViolates(self, cfg, name, endpoint=300):
        report = self.evaluate(cfg, endpoint)
        self.assertIn(name, report["violations"], C.summary(report))
        self.assertFalse(report["passed"])
        return report

    def assertNotViolates(self, cfg, name, endpoint=300):
        report = self.evaluate(cfg, endpoint)
        self.assertNotIn(name, report["violations"], C.summary(report))
        return report


# one config change per FORBIDDEN rule (DESIGN 3.1 last table); every rule must have a case
FORBIDDEN_CASES = {
    "top_p_below_1_with_tis": [(R + ".top_p", 0.92)],
    "actor_grad_clip": [(A + ".grad_clip", 0.3)],
    "max_ckpt_to_keep": [("trainer.max_actor_ckpt_to_keep", 2)],
    "use_kl_in_reward": [("algorithm.use_kl_in_reward", True)],
    "filter_groups_enable": [("algorithm.filter_groups.enable", True)],
    "onpolicy_shortcut_shape": [("vagen.onpolicy_shortcut", True), (A + ".ppo_epochs", 3)],
    "turn_gae_with_rollout_rs": [("algorithm.adv_estimator", "turn_gae"),
                                 ("algorithm.rollout_correction.rollout_rs", "token_k1")],
    "parameter_sync_step_not_1": [("trainer.v1.trainer_mode", "separate_async"),
                                  ("trainer.v1.separate_async.parameter_sync_step", 4)],
    "epochs_short_of_endpoint": [("trainer.total_epochs", 1)],
    "endpoint_beyond_horizon": [("trainer.total_training_steps", 200)],
    "partial_rollout_false": [("async_training", {"partial_rollout": False})],
    "thinking_template": [("data.apply_chat_template_kwargs", {"enable_thinking": True})],
    "kl_8b_factor": [("vagen.ga_8b", 24)],
    "parity_override_outside_parity": [("vagen.parity_sampling_override", {"temperature": 0.8, "top_p": 0.92})],
    "dynamic_bsz": [(A + ".use_dynamic_bsz", True)],
    # smoke S5 (vagen/q38/fault_injection.py; more cases in tests/test_fault_injection.py)
    "fault_injection_config": [("vagen.fault_injection", {"enable": False, "env_step_exception": {"turn": 2}})],
    "stuck_warn_below_floor": [("vagen.stuck_warn", {"min_s": 1, "factor": 0, "repeat_s": 30})],
}

# the Q5 PT update shape (DESIGN 3.1 last rows; smoke S4 keeps every group with filter.enable false before phase 2)
Q5_PT_SHAPE = [("vagen.profile", "q5_pt"), (A + ".ppo_mini_batch_size", 24), (A + ".ppo_epochs", 3),
               ("vagen.ga_8b", 24), ("filter.enable", False)]


class ForbiddenTest(Base):
    def test_base_passes(self):
        report = self.evaluate(self.base())
        self.assertTrue(report["passed"], C.summary(report))
        self.assertEqual(report["derived"]["dataset_size"], N_ENVS)
        self.assertEqual(report["derived"]["steps_per_epoch"], 200)
        self.assertFalse(any(f["violated"] for f in report["forbidden"]))

    def test_every_rule_has_a_case(self):
        self.assertEqual(set(FORBIDDEN_CASES), {f.name for f in C.FORBIDDEN})

    def test_each_combination_is_caught(self):
        for name, changes in FORBIDDEN_CASES.items():
            with self.subTest(rule=name):
                cfg = self.base()
                for key, value in changes:
                    put(cfg, key, value)
                report = self.assertViolates(cfg, name)
                entry = next(f for f in report["forbidden"] if f["name"] == name)
                self.assertTrue(entry["violated"] and entry["detail"])

    def test_variants(self):
        cases = [
            # (rule, changes, endpoint, violated?)
            ("actor_grad_clip", [(A + ".optim.grad_clip", 0.5)], 300, True),  # the deprecated alias
            ("max_ckpt_to_keep", [("trainer.max_critic_ckpt_to_keep", 2)], 300, True),
            ("top_p_below_1_with_tis", [(R + ".top_p", 0.92), ("algorithm.rollout_correction.rollout_is", None)],
             300, False),
            ("top_p_below_1_with_tis", [("vagen.parity_mode", True),
                                        ("vagen.parity_sampling_override", {"temperature": 0.8, "top_p": 0.92})],
             300, True),  # parity override samples at top-p 0.92 while TIS is still on
            ("onpolicy_shortcut_shape", [("vagen.onpolicy_shortcut", True)], 300, False),  # 1 epoch, mini = 12 kept
            ("onpolicy_shortcut_shape", [("vagen.onpolicy_shortcut", True), (A + ".ppo_mini_batch_size", 24)],
             300, True),
            ("onpolicy_shortcut_shape", [("vagen.onpolicy_shortcut", True), ("filter.enable", False)], 300, True),
            ("parameter_sync_step_not_1", [("trainer.v1.trainer_mode", "separate_async"),
                                           ("trainer.v1.separate_async.parameter_sync_step", 1)], 300, False),
            # the colocate mode reads the key the same way (TB:146): absent is 1, 2 would split the step
            ("parameter_sync_step_not_1", [("trainer.v1.colocate_async.parameter_sync_step", 2)], 300, True),
            ("parameter_sync_step_not_1", [("trainer.v1.colocate_async.parameter_sync_step", 1)], 300, False),
            ("parameter_sync_step_not_1", [("trainer.v1.trainer_mode", "sync"),
                                           ("trainer.v1.sync", {"parameter_sync_step": 3})], 300, True),
            ("epochs_short_of_endpoint", [("trainer.total_epochs", 2)], 400, False),  # 2 x 200 = 400
            ("epochs_short_of_endpoint", [("trainer.total_epochs", 2)], 401, True),
            ("endpoint_beyond_horizon", [], 700, False),
            ("endpoint_beyond_horizon", [], 701, True),
            ("endpoint_beyond_horizon", [], 0, True),
            ("partial_rollout_false", [("async_training", {"partial_rollout": True})], 300, False),
            # LS:347 reads a section without the key (or with null) as false: only true passes
            ("partial_rollout_false", [("async_training", {})], 300, True),
            ("partial_rollout_false", [("async_training", {"partial_rollout": None})], 300, True),
            ("partial_rollout_false", [("async_training", {"staleness_threshold": 1})], 300, True),
            ("partial_rollout_false", [("async_training", None)], 300, True),
            ("thinking_template", [("data.apply_chat_template_kwargs", {})], 300, True),  # enable_thinking missing
            ("thinking_template", [("data.apply_chat_template_kwargs",
                                    {"enable_thinking": False, "preserve_thinking": False})], 300, True),
            ("kl_8b_factor", [(A + ".ppo_mini_batch_size", 24), ("vagen.ga_8b", 24)], 300, False),  # Q5 PT shape
            ("kl_8b_factor", [("vagen.ga_8b", 0)], 300, True),
            ("parity_override_outside_parity", [("vagen.parity_sampling_override", {})], 300, False),
            ("dynamic_bsz", [("actor_rollout_ref.ref.log_prob_use_dynamic_bsz", True)], 300, True),
        ]
        for name, changes, endpoint, violated in cases:
            with self.subTest(rule=name, changes=changes, endpoint=endpoint):
                cfg = self.base()
                for key, value in changes:
                    put(cfg, key, value)
                if violated:
                    self.assertViolates(cfg, name, endpoint)
                else:
                    self.assertNotViolates(cfg, name, endpoint)

    def test_no_endpoint_skips_the_endpoint_rules(self):
        cfg = self.base()
        put(cfg, "trainer.total_epochs", 1)
        report = C.evaluate(cfg, None, check_paths=False)
        self.assertNotIn("epochs_short_of_endpoint", report["violations"])
        self.assertNotIn("endpoint_beyond_horizon", report["violations"])
        self.assertIn("trainer.total_epochs", report["violations"])  # still not ceil(700 / 200) + 1


class ExpectedTest(Base):
    def test_values(self):
        cases = [
            (A + ".optim.lr", 1e-6),
            (A + ".optim.clip_grad", 1.0),
            (A + ".optim.lr_scheduler_type", "constant"),
            (A + ".loss_agg_mode", "token-mean"),
            (A + ".loss_scale_factor", 1.0),  # global_batch_info + SUM aggregation (losses.py:70-85)
            (A + ".optim.zero_indexed_step", False),  # the schedule one step off (T11)
            ("reward.reward_model.enable", True),  # a colocated RM overwrites rm_scores after selection
            ("reward_model.enable", True),  # the legacy key, migrated into reward.reward_model.enable
            ("skip.rollout_tq.enable", True),  # verl's rollout skip: no warm-up dispatch (TB:659)
            ("skip.rollout.enable", True),
            ("skip.async_rollout.enable", True),
            (A + ".use_kl_loss", 1),  # an int is not a bool
            (A + ".checkpoint.save_contents", ["model", "optimizer", "extra"]),
            ("algorithm.rollout_correction.rollout_is", "sequence"),
            (R + ".temperature", 1.0),
            (R + ".limit_images", None),
            (R + ".logprobs_mode", "raw_logprobs"),
            (R + ".agent.agent_loop_config_path", "vagen/configs/agent.yaml"),
            (R + ".kernels", "hopper"),
            (R + ".custom.lmcache.enable", True),  # PEAR's LMCache block
            (R + ".engine_kwargs.vllm.kv_transfer_config", {"kv_connector": "LMCacheConnectorV1"}),
            (R + ".enable_sleep_mode", False),
            ("trainer.v1.sampler.custom_sampler", {"path": None, "name": None}),
            ("trainer.v1.trainer_mode", "separate_async"),
            ("trainer.logger", ["wandb"]),
            ("trainer.save_freq", 10),
            ("trainer.total_epochs", 30),
            ("trainer.total_training_steps", 300),
            ("vagen.failure_policy", "skip"),
            ("vagen.trainer_impl", "custom"),
            ("filter.name", "reward_variance"),
            ("filter.filter_kwargs.std_threshold", 0.3),
            ("data.custom_cls", {"path": "vagen/gym_agent_dataset.py", "name": "AgenticDataset"}),
            ("critic.enable", True),
            # sampler_kwargs are interpolations of values written once
            ("trainer.v1.sampler.sampler_kwargs.rollout_n", 4),
            ("trainer.v1.sampler.sampler_kwargs.rollout_data_dir", "/elsewhere/rollout_data"),
            ("trainer.v1.sampler.sampler_kwargs.filter", {"enable": False}),
            ("vagen.profile", "q5"),
            # the main line's shape (no profile): Q5 PT values fail it
            (A + ".ppo_epochs", 3),
            (A + ".ppo_mini_batch_size", 24),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                cfg = self.base()
                put(cfg, key, value)
                self.assertViolates(cfg, key)

    def test_missing_key(self):
        cfg = self.base()
        delete(cfg, R + ".limit_images")
        report = self.assertViolates(cfg, R + ".limit_images")
        entry = next(e for e in report["expected"] if e["key"] == R + ".limit_images")
        self.assertEqual(entry["actual"], "<absent>")

    def test_rules_on_verl_defaults(self):
        """Keys the recipe takes at verl's default: absent sections pass where the default is safe; the 0-indexed
        schedule must be written (absent is a violation, the scheduler's own default is not trusted)."""
        cfg = self.base()
        for key in ("skip", "reward", "reward_model", A + ".loss_scale_factor"):
            delete(cfg, key)
        report = self.evaluate(cfg)
        self.assertTrue(report["passed"], C.summary(report))
        for key, value in ((A + ".loss_scale_factor", None), ("reward_model.enable", False),
                           ("reward.reward_model.enable", False), ("skip.rollout_tq.enable", False)):
            with self.subTest(key=key, value=value):
                cfg = self.base()
                put(cfg, key, value)
                self.assertNotViolates(cfg, key)
        cfg = self.base()
        delete(cfg, A + ".optim.zero_indexed_step")
        self.assertViolates(cfg, A + ".optim.zero_indexed_step")

    def test_interpolations_resolve(self):
        cfg = self.base()
        put(cfg, "actor_rollout_ref.rollout.n", 4)  # sampler_kwargs.rollout_n follows; rollout.n itself fails
        report = self.evaluate(cfg)
        self.assertIn(R + ".n", report["violations"])
        self.assertNotIn("trainer.v1.sampler.sampler_kwargs.rollout_n", report["violations"])
        self.assertIn("kl_8b_factor", report["violations"])  # 12 x 4 / 12 = 4

    def test_env_specs(self):
        cfg = self.base()
        bad = Path(self.tmp.name) / "train_env_10turns.yaml"
        bad.write_text(ENV_YAML % (10, N_ENVS, N_ENVS))
        put(cfg, "data.train_files", str(bad))
        self.assertViolates(cfg, "data.train_files#max_turns")
        put(cfg, "data.train_files", str(Path(self.tmp.name) / "missing.yaml"))
        report = self.assertViolates(cfg, "data.train_files#max_turns")
        self.assertIn("trainer.total_epochs", report["violations"])  # no dataset size, cannot tell
        self.assertTrue(report["derived"]["errors"])

    def test_stock_profile(self):
        cfg = self.base()
        put(cfg, "vagen.trainer_impl", "stock")
        report = self.evaluate(cfg)
        self.assertIn("trainer.v1.sampler.custom_sampler.path", report["violations"])
        self.assertIn("filter.enable", report["violations"])
        put(cfg, "trainer.v1.sampler.custom_sampler", {"path": None, "name": None})
        put(cfg, "filter.enable", False)
        put(cfg, "trainer.checkpoint_callback_class", None)
        put(cfg, "vagen.failure_policy", "skip")  # not used by the stock sampler
        report = self.evaluate(cfg)
        self.assertTrue(report["passed"], C.summary(report))
        self.assertEqual(report["profile"]["trainer_impl"], "stock")

    def test_parity_profile(self):
        cfg = self.base()
        put(cfg, "vagen.parity_mode", True)
        put(cfg, "trainer.v1.trainer_mode", "sync")
        put(cfg, R + ".temperature", 1.0)
        put(cfg, "algorithm.rollout_correction.rollout_is", None)
        put(cfg, "vagen.parity_sampling_override", dict(C.PARITY_SAMPLING))
        report = self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)
        self.assertTrue(report["passed"], C.summary(report))
        # P8-2 runs 20 updates (DESIGN 5.3): --endpoint, else vagen.endpoint
        self.assertIn("vagen.endpoint", self.evaluate(cfg, endpoint=300)["violations"])
        put(cfg, "vagen.endpoint", C.PARITY_ENDPOINT)
        self.assertTrue(self.evaluate(cfg, endpoint=None)["passed"])
        # raw_logprobs (the 8B's way of measuring rollout_probs_diff) only in the parity run
        put(cfg, R + ".logprobs_mode", "raw_logprobs")
        report = self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)
        self.assertTrue(report["passed"], C.summary(report))
        put(cfg, R + ".logprobs_mode", "raw_logits")
        self.assertIn(R + ".logprobs_mode", self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)["violations"])
        put(cfg, R + ".logprobs_mode", "processed_logprobs")
        # V4 keep like the three 8B seeds
        put(cfg, "filter.name", "mixed_outcome")
        self.assertIn("filter.name", self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)["violations"])
        put(cfg, "filter.name", "keep_by_return_std")
        put(cfg, "filter.enable", False)
        self.assertIn("filter.enable", self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)["violations"])
        put(cfg, "filter.enable", True)
        put(cfg, "trainer.v1.trainer_mode", "colocate_async")
        self.assertIn("trainer.v1.trainer_mode", self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)["violations"])
        put(cfg, "trainer.v1.trainer_mode", "sync")
        put(cfg, "vagen.parity_sampling_override", {"temperature": 0.8})
        self.assertIn("vagen.parity_sampling_override",
                      self.evaluate(cfg, endpoint=C.PARITY_ENDPOINT)["violations"])

    def test_raw_logprobs_outside_parity(self):
        cfg = self.base()
        put(cfg, R + ".logprobs_mode", "raw_logprobs")
        self.assertViolates(cfg, R + ".logprobs_mode")

    def test_q5_pt_profile(self):
        """vagen.profile q5_pt: mini 24, 3 passes, ga_8b 24 and (before phase 2) every group kept; the main shape
        fails it and it fails the main shape."""
        cfg = self.base()
        for key, value in Q5_PT_SHAPE:
            put(cfg, key, value)
        report = self.evaluate(cfg)
        self.assertTrue(report["passed"], C.summary(report))
        self.assertEqual(report["profile"]["shape"], "q5_pt")
        self.assertFalse(report["profile"]["filter"])
        applied = {(e["key"], json.dumps(e.get("expected"))) for e in report["expected"] if e["applies"]}
        self.assertIn((A + ".ppo_epochs", "3"), applied)
        self.assertNotIn((A + ".ppo_epochs", "1"), applied)
        for key, value, rules in (
                (A + ".ppo_mini_batch_size", 12, [A + ".ppo_mini_batch_size", "kl_8b_factor"]),
                (A + ".ppo_epochs", 1, [A + ".ppo_epochs"]),
                ("vagen.ga_8b", 12, ["vagen.ga_8b", "kl_8b_factor"]),
                ("filter.enable", True, ["filter.enable"]),
        ):
            with self.subTest(key=key, value=value):
                bad = self.base()
                for k, v in Q5_PT_SHAPE:
                    put(bad, k, v)
                put(bad, key, value)
                report = self.evaluate(bad)
                self.assertFalse(report["passed"])
                for rule in rules:
                    self.assertIn(rule, report["violations"], C.summary(report))
        # the same shape without the profile is checked against the main line
        main = self.base()
        for key, value in Q5_PT_SHAPE:
            if key != "vagen.profile":
                put(main, key, value)
        report = self.evaluate(main)
        self.assertEqual(report["profile"]["shape"], "main")
        for rule in (A + ".ppo_mini_batch_size", A + ".ppo_epochs", "vagen.ga_8b"):
            self.assertIn(rule, report["violations"])
        # the on-policy shortcut stays refused with 3 passes
        put(cfg, "vagen.onpolicy_shortcut", True)
        self.assertViolates(cfg, "onpolicy_shortcut_shape")
        # profile main written out is the default
        main = self.base()
        put(main, "vagen.profile", "main")
        self.assertTrue(self.evaluate(main)["passed"])
        self.assertEqual(C.profile(main), "main")
        put(main, "vagen.profile", None)
        self.assertEqual(C.profile(main), "main")

    def test_shapes_keep_the_kl_factor(self):
        for name, shape in C.SHAPES.items():
            with self.subTest(profile=name):
                self.assertEqual(shape["ppo_mini_batch_size"] * 8, C.KL_8B_FACTOR * shape["ga_8b"])
        self.assertEqual(set(C.SHAPES), set(C.PROFILES))

    def test_allow_override(self):
        cfg = self.base()
        put(cfg, A + ".optim.lr", 1e-6)
        put(cfg, A + ".use_dynamic_bsz", True)
        put(cfg, "vagen.contract.allow_override", [A + ".optim.lr", "dynamic_bsz"])
        report = self.evaluate(cfg)
        self.assertTrue(report["passed"], C.summary(report))
        self.assertEqual(sorted(report["overridden"]), sorted([A + ".optim.lr", "dynamic_bsz"]))
        put(cfg, "vagen.contract.allow_override", [A + ".optim.lr", "dynamic_bsz", "no_such_rule"])
        report = self.evaluate(cfg)
        self.assertFalse(report["passed"])
        self.assertTrue(any("no_such_rule" in p for p in report["problems"]))

    def test_paths(self):
        cfg = self.base()
        put(cfg, "actor_rollout_ref.model.path", str(Path(self.tmp.name) / "no_model"))
        put(cfg, "filter.filter_kwargs.rows", str(Path(self.tmp.name) / "no_rows.jsonl"))
        report = C.evaluate(cfg, 300, check_paths=True)
        self.assertIn("path:actor_rollout_ref.model.path", report["violations"])
        self.assertIn("path:filter.filter_kwargs.rows", report["violations"])
        self.assertNotIn("path:data.train_files", report["violations"])
        put(cfg, "vagen.contract.allow_override", ["path:actor_rollout_ref.model.path", "path:filter.filter_kwargs.rows"])
        self.assertTrue(C.evaluate(cfg, 300, check_paths=True)["passed"])

    def test_values_equal(self):
        self.assertTrue(C.values_equal(2, 2.0))
        self.assertTrue(C.values_equal(2e-6, 2.0e-06))
        self.assertFalse(C.values_equal(1, True))
        self.assertFalse(C.values_equal(0, False))
        self.assertFalse(C.values_equal(None, False))
        self.assertTrue(C.values_equal({"a": [1, 2.0]}, {"a": [1.0, 2]}))
        self.assertFalse(C.values_equal({"a": 1}, {"a": 1, "b": 2}))
        self.assertFalse(C.values_equal(0.3318, 0.3319))

    def test_constants(self):
        self.assertEqual((C.MAX_TURNS, C.RESPONSE_LENGTH_PER_TURN, C.LIMIT_IMAGES), (12, 384, 13))
        self.assertEqual(12 * 8 // 12, C.KL_8B_FACTOR)


class CheckFilesTest(Base):
    def test_check_writes_files(self):
        cfg = self.base()
        with tempfile.TemporaryDirectory() as out:
            report = C.check(cfg, 300, check_paths=False, run_dir=Path(out))
            data = json.loads((Path(out) / "contract_check.json").read_text())
            self.assertTrue(data["passed"])
            self.assertEqual(data["endpoint"], 300)
            self.assertEqual(data["profile"]["trainer_mode"], "colocate_async")
            keys = {e["key"] for e in data["expected"]}
            self.assertIn(A + ".optim.clip_grad", keys)
            self.assertEqual({f["name"] for f in data["forbidden"]}, {f.name for f in C.FORBIDDEN})
            self.assertTrue(all("source" in e for e in data["expected"]))
            effective = OmegaConf.load(Path(out) / "effective_config.yaml")
            # resolved: the interpolations are values now
            self.assertEqual(effective.trainer.v1.sampler.sampler_kwargs.rollout_n, 8)
            self.assertEqual(effective.trainer.v1.sampler.sampler_kwargs.filter.name, "keep_by_return_std")
            self.assertEqual(report["files"]["contract_check"], str(Path(out) / "contract_check.json"))
            self.assertFalse(any(p.name.endswith(".tmp") for p in Path(out).iterdir()))

    def test_check_raises_and_still_writes(self):
        cfg = self.base()
        put(cfg, "algorithm.use_kl_in_reward", True)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaises(C.ContractError) as caught:
                C.check(cfg, 300, check_paths=False, run_dir=Path(out))
            self.assertIn("use_kl_in_reward", str(caught.exception))
            data = json.loads((Path(out) / "contract_check.json").read_text())
            self.assertFalse(data["passed"])
            self.assertIn("use_kl_in_reward", data["violations"])

    def test_run_dir_from_default_local_dir(self):
        cfg = self.base()
        report = C.evaluate(cfg, 300, check_paths=False)
        self.assertEqual(report["derived"]["run_dir"], str(self.run_dir))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as out:
            path = Path(out) / "ppo.yaml"
            cfg = self.base()
            OmegaConf.save(cfg, path)
            self.assertEqual(C.main(["--config", str(path), "--endpoint", "300", "--no-paths"]), 0)
            self.assertFalse((self.run_dir / "contract_check.json").exists())  # a dry run writes nothing
            put(cfg, A + ".use_dynamic_bsz", True)
            OmegaConf.save(cfg, path)
            self.assertEqual(C.main(["--config", str(path), "--endpoint", "300", "--no-paths"]), 2)


@unittest.skipUnless((R8 / "rl_qw3v4s3" / "train_env.yaml").is_file(), "8B V4 run not on this machine")
class SmokeExampleTest(unittest.TestCase):
    """The hand-written 4B smoke config: the contract with real paths, and verl's own start-up checks."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = OmegaConf.load(SMOKE)

    def test_contract_passes(self):
        for endpoint in (3, 10, 700):
            with self.subTest(endpoint=endpoint):
                report = C.evaluate(self.cfg, endpoint, check_paths=True)
                self.assertTrue(report["passed"], C.summary(report))
                self.assertEqual(report["derived"]["dataset_size"], 10798)
                self.assertEqual(report["derived"]["steps_per_epoch"], 10798 // 24)
                self.assertEqual(self.cfg.trainer.total_epochs, math.ceil(700 / (10798 // 24)) + 1)
        report = C.evaluate(self.cfg, 701, check_paths=True)
        self.assertEqual(report["violations"], ["endpoint_beyond_horizon"])

    def test_smoke_s4_q5_pt(self):
        """Smoke S4 (DESIGN 5.2): the example in the Q5 PT shape, save_freq 3 (the one override), endpoint 3."""
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
        for key, value in Q5_PT_SHAPE + [("trainer.save_freq", 3),
                                         ("vagen.contract.allow_override", ["trainer.save_freq"])]:
            put(cfg, key, value)
        report = C.evaluate(cfg, 3, check_paths=True)
        self.assertTrue(report["passed"], C.summary(report))
        self.assertEqual(report["overridden"], ["trainer.save_freq"])
        self.assertEqual(report["profile"]["shape"], "q5_pt")
        # the sampler gets the filter section through the interpolation: every group, factor 1
        self.assertEqual(OmegaConf.select(cfg, "trainer.v1.sampler.sampler_kwargs.filter.enable"), False)

    def test_shape(self):
        cfg = self.cfg
        self.assertNotIn("grad_clip", cfg.actor_rollout_ref.actor)
        self.assertEqual(cfg.actor_rollout_ref.model.path.rsplit("/", 1)[-1], "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertTrue(Path(cfg.actor_rollout_ref.model.path, "config.json").is_file())
        self.assertEqual(C.trainer_impl(cfg), "vagen")
        self.assertEqual(cfg.trainer.v1.trainer_mode, "colocate_async")

    def test_verl_startup_checks(self):
        """verl.trainer.main_ppo.main's checks (validate_config) and the dataclasses the workers build."""
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import omega_conf_to_dataclass, validate_config

        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
        self.assertTrue(need_reference_policy(cfg))
        self.assertFalse(need_critic(cfg))
        validate_config(config=cfg, use_reference_policy=True, use_critic=False)
        actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
        self.assertEqual(actor.optim.clip_grad, 0.3)
        self.assertEqual(actor.loss_agg_mode, "seq-mean-token-mean")
        rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
        self.assertEqual((rollout.temperature, rollout.top_p, rollout.limit_images), (0.8, 1.0, 13))
        self.assertTrue(rollout.enable_sleep_mode and rollout.free_cache_engine)
        algorithm = omega_conf_to_dataclass(cfg.algorithm)
        # AlgoConfig keeps rollout_correction as a plain mapping (no _target_ in the generated defaults); V1 reads it
        # from the DictConfig (trainer_base.py:1714)
        correction = algorithm.rollout_correction
        correction = correction if isinstance(correction, dict) else vars(correction)
        self.assertEqual((correction["rollout_is"], correction["rollout_is_threshold"]), ("token", 2.0))


class EntryTest(unittest.TestCase):
    """vagen/q38/ppo_v1.py: endpoint handling (DESIGN 2.5.4), runtime env, trainer selection."""

    @staticmethod
    def trainer(global_steps=0, horizon=700, steps_per_epoch=449, total_epochs=3):
        return SimpleNamespace(global_steps=global_steps, total_training_steps=horizon,
                               steps_per_epoch=steps_per_epoch,
                               config=OmegaConf.create({"trainer": {"total_epochs": total_epochs}}))

    def test_prepare_endpoint(self):
        from vagen.q38.ppo_v1 import prepare_endpoint

        t = self.trainer()
        self.assertTrue(prepare_endpoint(t, 10))
        self.assertEqual(t.total_training_steps, 10)  # fit stops (and saves) at the endpoint
        t = self.trainer(global_steps=50)
        self.assertTrue(prepare_endpoint(t, 300))
        self.assertEqual(t.total_training_steps, 300)
        t = self.trainer(global_steps=300)
        self.assertFalse(prepare_endpoint(t, 300))  # already there: no fit
        self.assertEqual(t.total_training_steps, 700)
        self.assertFalse(prepare_endpoint(self.trainer(global_steps=350), 300))
        with self.assertRaises(ValueError):
            prepare_endpoint(self.trainer(), 701)  # beyond the learning-rate horizon
        with self.assertRaises(ValueError):
            prepare_endpoint(self.trainer(), 0)
        with self.assertRaises(ValueError):
            prepare_endpoint(self.trainer(steps_per_epoch=100, total_epochs=2), 201)  # epochs run out first
        self.assertTrue(prepare_endpoint(self.trainer(steps_per_epoch=100, total_epochs=2), 200))

    def test_endpoint_checkpoint(self):
        from vagen.q38.ppo_v1 import endpoint_checkpoint

        cfg = OmegaConf.create({"trainer": {"default_local_dir": "/runs/x/checkpoints"}})
        self.assertEqual(endpoint_checkpoint(cfg, 300), Path("/runs/x/checkpoints/global_step_300"))
        cfg = OmegaConf.create({"trainer": {"default_local_dir": "rel/checkpoints"}})
        self.assertEqual(endpoint_checkpoint(cfg, 3), Path(os.getcwd()) / "rel/checkpoints/global_step_3")

    def test_prepare_config(self):
        from vagen.q38.ppo_v1 import prepare_config

        cfg = OmegaConf.create({"ray_kwargs": {"ray_init": {"num_cpus": None, "runtime_env": {
            "py_executable": None, "env_vars": {"KEEP": "1"}}}}, "vagen": {"endpoint": 5}})
        with tempfile.TemporaryDirectory() as tmp:
            environ = {"R1_RENDER_URL": "http://h:1/render", "PYTHONPATH": "/repo", "V10_RAY_TEMP_DIR": tmp + "/ray"}
            prepare_config(cfg, 300, environ=environ)
            self.assertTrue(Path(tmp, "ray").is_dir())
        self.assertEqual(cfg.vagen.endpoint, 300)
        ray_init = OmegaConf.to_container(cfg.ray_kwargs.ray_init)
        self.assertFalse(ray_init["include_dashboard"])
        self.assertEqual(ray_init["_temp_dir"], environ["V10_RAY_TEMP_DIR"])
        self.assertNotIn("py_executable", ray_init["runtime_env"])
        env_vars = ray_init["runtime_env"]["env_vars"]
        self.assertEqual(env_vars["KEEP"], "1")
        self.assertEqual(env_vars["R1_RENDER_URL"], "http://h:1/render")
        self.assertEqual(env_vars["PYTHONPATH"], "/repo")
        self.assertEqual(env_vars["WANDB_MODE"], "offline")
        self.assertEqual((env_vars["NO_PROXY"], env_vars["no_proxy"], env_vars["PYTHONDONTWRITEBYTECODE"]),
                         ("*", "*", "1"))
        with self.assertRaises(KeyError):  # like the 8B entry: no render service, no run
            prepare_config(OmegaConf.create({"ray_kwargs": {"ray_init": {}}}), 3, environ={"PYTHONPATH": "/r"})

    def test_select_trainer_cls(self):
        from verl.trainer.ppo.v1 import PPOTrainerColocateAsync, PPOTrainerSync

        from vagen.q38 import ppo_v1

        def cfg(mode, impl):
            return OmegaConf.create({"trainer": {"v1": {"trainer_mode": mode}}, "vagen": {"trainer_impl": impl}})

        self.assertIs(ppo_v1.select_trainer_cls(cfg("colocate_async", "stock")), PPOTrainerColocateAsync)
        self.assertIs(ppo_v1.select_trainer_cls(cfg("sync", "stock")), PPOTrainerSync)

        fake = types.ModuleType(ppo_v1.TRAINER_MODULE)

        class VagenColocateAsyncTrainer(PPOTrainerColocateAsync):
            pass

        fake.VagenColocateAsyncTrainer = VagenColocateAsyncTrainer
        fake.VagenSyncTrainer = PPOTrainerColocateAsync  # wrong base on purpose
        saved = sys.modules.get(ppo_v1.TRAINER_MODULE)
        sys.modules[ppo_v1.TRAINER_MODULE] = fake
        try:
            self.assertIs(ppo_v1.select_trainer_cls(cfg("colocate_async", "vagen")), VagenColocateAsyncTrainer)
            with self.assertRaises(TypeError):
                ppo_v1.select_trainer_cls(cfg("sync", "vagen"))
            with self.assertRaises(ValueError):  # phase 4: not written yet
                ppo_v1.select_trainer_cls(cfg("separate_async", "vagen"))
            with self.assertRaises(ValueError):
                ppo_v1.select_trainer_cls(cfg("colocate_async", "other"))
        finally:
            if saved is None:
                sys.modules.pop(ppo_v1.TRAINER_MODULE, None)
            else:
                sys.modules[ppo_v1.TRAINER_MODULE] = saved


if __name__ == "__main__":
    unittest.main()
