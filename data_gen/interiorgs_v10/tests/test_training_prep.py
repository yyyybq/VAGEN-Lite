"""training_prep.py: SFT frames, conversations and the LLaMA-Factory preparation of the first v10 SFT version."""

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.admission import load_policy
from data_gen.interiorgs_v10.pilot import safe_name
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

ROOT = Path(__file__).resolve().parents[3]


def _prepare_sft_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("prepare_active_spatial_sft", ROOT / "scripts/prepare_active_spatial_sft.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingPrepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        rows = _jsonable(generate_scene(paths["two_rooms"], "two_rooms", RunConfig(target_tasks_per_scene=4)))["rows"]
        # a second "scene" for validation: the same tasks under another scene id
        val = [dict(r, scene_id="two_rooms_val", task_id=r["task_id"] + "_val") for r in rows[:2]]
        cls.run_dir = cls.root / "run"
        cls.run_dir.mkdir()
        (cls.run_dir / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + val))
        cls.manifest = cls.root / "split.json"
        cls.manifest.write_text(json.dumps({"version": 2, "train_scenes": ["two_rooms"],
                                            "val_scenes": ["two_rooms_val"], "test_scenes": []}))
        cls.n_train, cls.n_val = len(rows), len(val)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_frames_records_and_llamafactory(self):
        sft = self.root / "sft"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        rows = TP._jsonl(sft / "rows.jsonl")
        self.assertEqual(len(rows), self.n_train + self.n_val)
        frames = TP._jsonl(sft / "frames.jsonl")
        policy = load_policy()
        for r in rows:
            d = r["difficulty"]["d_star"]
            sids = {f["state_id"] for f in frames if f["task_id"] == r["task_id"]}
            self.assertIn("s0", sids)
            self.assertIn(f"s{d}", sids)  # the final state too
            self.assertEqual(sids, {f"s{k}" for k in TP.sft_states(r, policy)})
        self.assertTrue(all(f["overlay"] == {} for f in frames))  # plain frames only
        registry = TP._jsonl(sft / "task_registry.jsonl")
        self.assertEqual({r["split"] for r in registry}, {"train", "val"})
        # fake renders, then the conversations
        from PIL import Image

        (sft / "frames" / "plain").mkdir(parents=True)
        for f in frames:
            Image.new("RGB", (256, 256), (90, 90, 90)).save(sft / "frames" / "plain" / safe_name(f["frame_id"]))
        TP.cmd_sft_records(Namespace(dir=str(sft)))
        records = TP._jsonl(sft / "sft_trajectories.jsonl")
        self.assertEqual(len(records), len(rows))
        for rec in records:
            users = [t for t in rec["conversations"] if t["role"] == "user"]
            self.assertTrue(all("Current camera pose: [" in t["content"] for t in users))
            self.assertTrue(all(Path(t["image_path"]).is_file() for t in users))
            self.assertTrue(all(t["content"].startswith("<action>") for t in rec["conversations"]
                                if t["role"] == "assistant"))
        # LLaMA-Factory: trajectories only, unreviewed admitted explicitly and recorded
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen2_5_vl"}))
        prep = _prepare_sft_module()
        common = dict(trajectory=[str(sft / "sft_trajectories.jsonl")], qa_bank=[], model=str(model),
                      qa_probability=0.0, val_fraction=0.1, split_manifest=str(sft / "split_manifest.json"),
                      task_registry=str(sft / "task_registry.jsonl"), seed=42, cutoff_len=16384)
        with self.assertRaises(ValueError):  # without the option every trajectory lacks a review
            prep.prepare(Namespace(out=str(self.root / "lf_strict"), allow_unreviewed_trajectories=False, **common))
        manifest = prep.prepare(Namespace(out=str(self.root / "lf"), allow_unreviewed_trajectories=True,
                                          save_per_epoch=True, **common))
        self.assertEqual(manifest["trajectory_semantic_review"], "not_required")
        self.assertEqual(manifest["counts"]["trajectory/train"], self.n_train)
        self.assertEqual(manifest["counts"]["trajectory/val"], self.n_val)
        cfg = (self.root / "lf" / "train.yaml").read_text()
        self.assertIn("dataset: trajectory_train", cfg)
        self.assertNotIn("qa_train", cfg)
        import yaml

        lf = yaml.safe_load(cfg)  # the model at the end of every epoch, weights only
        self.assertEqual((lf["save_strategy"], lf["save_only_model"]), ("epoch", True))
        self.assertEqual(lf["num_train_epochs"], 3)  # the default; round 10 trains 5 (--epochs)
        prep.prepare(Namespace(out=str(self.root / "lf_e5"), allow_unreviewed_trajectories=True, epochs=5, **common))
        self.assertEqual(yaml.safe_load((self.root / "lf_e5" / "train.yaml").read_text())["num_train_epochs"], 5)
        # round 12 freeze: the format SFT's smaller learning rate (default 1e-5)
        self.assertEqual(lf["learning_rate"], 1e-5)
        prep.prepare(Namespace(out=str(self.root / "lf_lr"), allow_unreviewed_trajectories=True, learning_rate=2e-6,
                               **common))
        self.assertEqual(yaml.safe_load((self.root / "lf_lr" / "train.yaml").read_text())["learning_rate"], 2e-6)
        prep.prepare(Namespace(out=str(self.root / "lf_steps"), allow_unreviewed_trajectories=True, **common))
        self.assertNotIn("save_strategy", yaml.safe_load((self.root / "lf_steps" / "train.yaml").read_text()))
        # round 10: --keep-text keeps every message as the env shows it (the system prompt and later user turns
        # end with a newline); the default strips them, as rounds 6-9 trained
        self.assertEqual(manifest["trajectory_text"], "stripped")
        kept = prep.prepare(Namespace(out=str(self.root / "lf_keep"), allow_unreviewed_trajectories=True,
                                      keep_text=True, **common))
        self.assertEqual(kept["trajectory_text"], "verbatim")
        source = records[0]["conversations"]
        for out_dir, expect in (("lf", lambda t: t.strip()), ("lf_keep", lambda t: t)):
            first = json.loads((self.root / out_dir / "trajectory_train.jsonl").read_text().splitlines()[0])
            self.assertEqual([m["content"] for m in first["messages"]], [expect(t["content"]) for t in source])
        self.assertTrue(source[0]["content"].endswith("\n"))  # the env's system prompt does end with one

    def test_rl_config_and_prompt_consistency(self):
        import yaml

        sft = self.root / "sft_rl"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        template = self.root / "ppo_template.yaml"
        template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False},
            "actor_rollout_ref": {"model": {"path": "old"}, "actor": {"ppo_mini_batch_size": 8},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4, "agent": {"num_workers": 8}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))
        out = self.root / "rl"
        TP.cmd_rl_config(Namespace(rows=str(sft / "rows.jsonl"), env_template=str(
            Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"), ppo_template=str(template),
            gs_root="/ply", model="/models/m/", name="trial", run_root=str(self.root / "rl_run"), val_limit=1,
            trial=True, out=str(out)))
        env = yaml.safe_load((out / "train_env.yaml").read_text())["envs"][0]
        self.assertEqual(env["n_envs"], self.n_train)
        self.assertEqual(env["config"]["prompt_format"], "no_think")
        self.assertIs(env["config"]["r1_legacy_success"], False)
        self.assertEqual(env["config"]["gs_root"], "/ply")
        self.assertEqual(len(TP._jsonl(Path(env["config"]["jsonl_path"]))), self.n_train)
        self.assertEqual(yaml.safe_load((out / "val_env.yaml").read_text())["envs"][0]["n_envs"], 1)
        # --val-limit keeps rows spread over the (scene-sorted) val rows, not the first ones
        out2 = self.root / "rl2"
        TP.cmd_rl_config(Namespace(rows=str(sft / "rows.jsonl"), env_template=str(
            Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"), ppo_template=str(template),
            gs_root="/ply", model="/m", name="t", run_root=str(self.root / "rl_run2"), val_limit=2,
            trial=False, out=str(out2)))
        val_rows = [r for r in TP._jsonl(sft / "rows.jsonl") if r["split"] == "val"]
        picked = [r["task_id"] for r in TP._jsonl(out2 / "val.jsonl")]
        self.assertEqual(picked, [r["task_id"] for r in val_rows[:: max(1, len(val_rows) // 2)][:2]])
        formal = yaml.safe_load((out2 / "ppo.yaml").read_text())["trainer"]
        self.assertEqual({k: formal[k] for k in ("save_freq", "test_freq", "resume_mode", "ckpt_keep_every",
                                                 "max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep")},
                         {"save_freq": 25, "test_freq": 25, "resume_mode": "auto", "ckpt_keep_every": 50,
                          "max_actor_ckpt_to_keep": 2, "max_critic_ckpt_to_keep": 2})
        self.assertEqual(formal["n_gpus_per_node"], 8)
        ppo = yaml.safe_load((out / "ppo.yaml").read_text())
        self.assertEqual(ppo["trainer"]["n_gpus_per_node"], 1)
        self.assertEqual((ppo["trainer"]["save_freq"], ppo["trainer"]["ckpt_keep_every"],
                          ppo["trainer"]["max_actor_ckpt_to_keep"], ppo["trainer"]["resume_mode"]), (1, 2, 1, "auto"))
        self.assertEqual(ppo["actor_rollout_ref"]["rollout"]["tensor_model_parallel_size"], 1)
        self.assertEqual(ppo["actor_rollout_ref"]["model"]["path"], "/models/m")
        self.assertEqual(ppo["data"]["train_files"], str((out / "train_env.yaml").resolve()))
        # the RL system prompt is the SFT one; a different reward in the env would show
        policy = load_policy()
        cfg = yaml.safe_load((Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml").read_text())
        cfg = cfg["envs"][0]["config"]
        types = ["projective_relations", "occlusion_alignment", "fov_inclusion"]
        self.assertEqual(TP.prompt_mismatches(cfg, types, policy), [])
        self.assertEqual(TP.prompt_mismatches(dict(cfg, format_reward=0.01), types, policy), sorted(types))
        # the check covers every argument the env passes (round 10: the reward text and the collision rule too)
        self.assertEqual(TP.prompt_mismatches(dict(cfg, prompt_reward_text="legacy"), types, policy), sorted(types))
        self.assertEqual(TP.prompt_mismatches(dict(cfg, max_consecutive_collisions=3), types, policy), sorted(types))

    def test_rl_config_round11_options(self):
        """Reward presets (S0 sparse, dense = the S5 shaping), 24 tasks per update, checkpoints every 50 updates
        (all kept, the older ones cut to their weights), no validation inside training."""
        import yaml

        sft = self.root / "sft_rl11"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        template = self.root / "ppo_template11.yaml"
        template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False, "save_freq": 25, "test_freq": 25},
            "actor_rollout_ref": {"model": {"path": "old"}, "actor": {"ppo_mini_batch_size": 8},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4, "agent": {"num_workers": 8}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))
        env_template = Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"
        self.assertEqual(TP.reward_overrides("S0"), {})
        self.assertTrue(TP.reward_overrides("dense")["enable_potential_shaping_reward"])
        with self.assertRaises(ValueError):
            TP.reward_overrides("S5")
        outs = {}
        for preset in ("S0", "dense"):
            out = self.root / f"rl11_{preset}"
            TP.cmd_rl_config(Namespace(rows=str(sft / "rows.jsonl"), env_template=str(env_template), ppo_template=str(template),
                                       gs_root="/ply", model="/m", name="t", run_root=str(self.root / f"run11_{preset}"),
                                       val_limit=1, trial=False, out=str(out), history_mode="concat", reward_preset=preset,
                                       train_batch_size=24, ppo_mini_batch_size=16, save_freq=50, no_internal_val=True))
            outs[preset] = out
        envs = {p: yaml.safe_load((o / "train_env.yaml").read_text())["envs"][0]["config"] for p, o in outs.items()}
        ppo = yaml.safe_load((outs["dense"] / "ppo.yaml").read_text())
        for key, value in TP.reward_overrides("dense").items():
            self.assertEqual(envs["dense"][key], value)
        # the S5 shaping, then the round-11 adjustments: progress term x10, the two terms that never pay off
        self.assertEqual((envs["dense"]["potential_field_gamma"], envs["dense"]["potential_field_reward_scale"]), (0.99, 10.0))
        self.assertTrue(envs["dense"]["enable_potential_shaping_reward"])
        self.assertFalse(envs["dense"]["enable_near_success_reward"])
        self.assertFalse(envs["dense"]["enable_visibility_shaping_reward"])
        self.assertFalse(envs["S0"]["enable_potential_shaping_reward"])
        differ = sorted(k for k in set(envs["S0"]) | set(envs["dense"]) if envs["S0"].get(k) != envs["dense"].get(k)
                        and k != "jsonl_path")
        self.assertEqual(differ, sorted(k for k, v in TP.reward_overrides("dense").items()
                                        if envs["S0"].get(k) != v))  # nothing else differs
        self.assertEqual((ppo["data"]["train_batch_size"], ppo["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"]), (24, 16))
        self.assertEqual({k: ppo["trainer"][k] for k in ("save_freq", "ckpt_keep_every", "ckpt_prune_to_hf",
                                                         "val_before_train", "test_freq")},
                         {"save_freq": 50, "ckpt_keep_every": 50, "ckpt_prune_to_hf": True, "val_before_train": False,
                          "test_freq": -1})
        self.assertEqual(json.loads((outs["dense"] / "rl_config_report.json").read_text())["reward_preset"], "dense")
        # the system prompt is the same for both presets (and matches the SFT one; rl-config checks it)
        types = ["projective_relations", "equidistance"]
        self.assertEqual(TP.prompt_mismatches(envs["dense"], types, load_policy()), [])

    def test_rl_config_optimisation_package(self):
        """Round 11: the optimisation package on top of the run's settings: GRPO without a value network, only
        mixed-outcome tasks (24 candidates, 12 trained on), 8 episodes per task, lr 2e-6, KL coefficient 0.05,
        entropy bonus 0.01; sizes can be made smaller for a one-card pilot; it needs the concat history mode."""
        import yaml

        sft = self.root / "sft_rl_pkg"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        template = self.root / "ppo_template_pkg.yaml"
        template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False, "critic_warmup": 10},
            "filter": {"enable": False, "name": "reward_variance", "filter_kwargs": {"topk": 0.2}},
            "actor_rollout_ref": {"model": {"path": "old"},
                                  "actor": {"ppo_mini_batch_size": 8, "optim": {"lr": 5e-7}, "kl_loss_coef": 0.3,
                                            "entropy_coeff": 0.005, "use_kl_loss": True},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4, "agent": {"num_workers": 8}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))

        def config(out_name, **kw):
            args = dict(rows=str(sft / "rows.jsonl"), env_template=str(Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"),
                        ppo_template=str(template), gs_root="/ply", model="/m", name="t",
                        run_root=str(self.root / f"run_{out_name}"), val_limit=1, trial=False, out=str(self.root / out_name),
                        history_mode="concat", train_batch_size=24, ppo_mini_batch_size=12, critic_warmup=10)
            args.update(kw)
            TP.cmd_rl_config(Namespace(**args))
            return yaml.safe_load((self.root / out_name / "ppo.yaml").read_text()), \
                json.loads((self.root / out_name / "rl_config_report.json").read_text())

        ppo, report = config("rl_pkg", opt_package="p1")
        a = ppo["actor_rollout_ref"]
        self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["algorithm"]["norm_adv_by_std_in_grpo"]), ("grpo", True))
        self.assertEqual((a["rollout"]["n"], ppo["data"]["train_batch_size"], a["actor"]["ppo_mini_batch_size"]), (8, 24, 12))
        self.assertEqual(ppo["filter"], {"enable": True, "name": "mixed_outcome", "filter_kwargs": {"target_groups": 12}})
        self.assertEqual((a["actor"]["optim"]["lr"], a["actor"]["kl_loss_coef"], a["actor"]["entropy_coeff"]), (2e-6, 0.05, 0.01))
        self.assertEqual(ppo["trainer"]["critic_warmup"], 0)
        self.assertTrue(ppo["trainer"]["concat_multi_turn"])
        self.assertEqual(report["opt_package"], "p1")
        self.assertEqual(report["opt_package_overrides"]["data.train_batch_size"], 24)
        # a one-card pilot: smaller sizes, the same algorithm and step sizes
        ppo, _ = config("rl_pkg_pilot", opt_package="p1", candidates=6, target_groups=2, trial=True)
        self.assertEqual((ppo["data"]["train_batch_size"], ppo["filter"]["filter_kwargs"]["target_groups"],
                          ppo["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"], ppo["actor_rollout_ref"]["rollout"]["n"]),
                         (6, 2, 2, 8))
        with self.assertRaises(ValueError):
            TP.package_overrides("p1", candidates=4, target_groups=6)
        with self.assertRaises(ValueError):
            TP.package_overrides("p9")
        with self.assertRaises(SystemExit):  # GRPO over whole episodes needs the concat mode
            config("rl_pkg_noconcat", opt_package="p1", history_mode="none")
        # without the package nothing changes
        ppo, report = config("rl_nopkg")
        self.assertEqual(ppo["algorithm"]["adv_estimator"], "gae")
        self.assertIsNone(report["opt_package"])

    def test_rl_config_standards_and_seed(self):
        """Round 12: named standard configurations (configs/rl_standards.json, the package with S0 and with the dense
        reward; p2 with 48 candidates) fix the recipe's options and the repository's PPO template; an explicit
        option against the standard is refused; --seed sets the task order (data.seed) and nothing in the rollout
        config (this verl's RolloutConfig has no seed: the first round-12 start failed on it)."""
        import yaml

        standards = TP.load_standards()
        self.assertEqual(sorted(standards), ["v10_p1_D", "v10_p1_D_keep_100", "v10_p1_D_keep_177", "v10_p1_Dcond",
                                             "v10_p1_S0", "v10_p1_S0_keep", "v10_p2_D", "v10_p2_S0"])
        for name, std in standards.items():
            self.assertLessEqual(set(std), set(TP.STANDARD_KEYS), name)
            for key in ("ppo_template", "env_template"):
                self.assertTrue((Path(TP.__file__).parent / std[key]).is_file(), (name, key))
            self.assertEqual(std["reward_preset"], "D_cond" if "_Dcond" in name else "dense" if "_D" in name else "S0")
        # the dense reward's first batch: the filter, thresholds and floors of the three runs
        self.assertEqual({n: (standards[n]["group_filter"], standards[n]["keep_std_threshold"], standards[n]["keep_std_floor"])
                          for n in ("v10_p1_D_keep_177", "v10_p1_D_keep_100", "v10_p1_S0_keep")},
                         {"v10_p1_D_keep_177": ("keep_by_return_std", 0.335, 1.77),
                          "v10_p1_D_keep_100": ("keep_by_return_std", 0.335, 1.0),
                          "v10_p1_S0_keep": ("keep_by_return_std", 0.134, 1.77)})
        self.assertEqual({**TP.OPT_PACKAGES["p2"], "candidates": 24}, TP.OPT_PACKAGES["p1"])  # p2 = p1 with 48 candidates
        # the repository template parses to the round-11 template (only a header comment added)
        repo_template = yaml.safe_load((Path(TP.__file__).with_name("configs") / "ppo_template_v10.yaml").read_text())
        self.assertEqual((repo_template["data"]["seed"], repo_template["algorithm"]["adv_estimator"]), (20260930, "no_concat_gae"))
        self.assertNotIn("seed", repo_template["actor_rollout_ref"]["rollout"])

        sft = self.root / "sft_rl_std"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        rows = [json.loads(l) for l in (sft / "rows.jsonl").read_text().splitlines() if l.strip()]
        for r in rows:  # the train rows become reserve rows (what the standards train on)
            r["reserve"] = r["split"] == "train"
        (sft / "rows_reserve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

        def config(out_name, **kw):
            args = dict(rows=str(sft / "rows_reserve.jsonl"), gs_root="/ply", model="/m", name="t",
                        run_root=str(self.root / f"run_{out_name}"), val_limit=1, trial=False, out=str(self.root / out_name))
            args.update(kw)
            TP.cmd_rl_config(Namespace(**args))
            return (yaml.safe_load((self.root / out_name / "ppo.yaml").read_text()),
                    yaml.safe_load((self.root / out_name / "train_env.yaml").read_text())["envs"][0]["config"],
                    json.loads((self.root / out_name / "rl_config_report.json").read_text()))

        ppo, env, report = config("rl_std_d", standard="v10_p1_D", seed=7)
        a = ppo["actor_rollout_ref"]
        self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["data"]["train_batch_size"], a["rollout"]["n"],
                          ppo["filter"]["filter_kwargs"]["target_groups"]), ("grpo", 24, 8, 12))
        self.assertTrue(ppo["trainer"]["concat_multi_turn"])
        self.assertEqual((ppo["trainer"]["save_freq"], ppo["trainer"]["test_freq"]), (50, -1))
        self.assertEqual(ppo["data"]["seed"], 7)
        self.assertNotIn("seed", a["rollout"])  # RolloutConfig(**rollout) must not get an unknown field
        self.assertEqual(env["potential_field_reward_scale"], 10.0)  # the dense reward
        self.assertEqual((report["standard"], report["seed"], report["task_pool"], report["reward_preset"]),
                         ("v10_p1_D", {"data": 7, "rollout": 0}, "reserve", "dense"))
        self.assertEqual(report["train"], sum(1 for r in rows if r["reserve"]))
        self.assertTrue(report["ppo_template"].endswith("ppo_template_v10.yaml"))
        # no seed: the template's (task order 20260930, sampling 0), as in round 11
        ppo, env, report = config("rl_std_s0", standard="v10_p2_S0")
        self.assertEqual(report["seed"], {"data": 20260930, "rollout": 0})
        self.assertEqual((ppo["data"]["train_batch_size"], ppo["filter"]["filter_kwargs"]["target_groups"]), (48, 12))
        self.assertFalse(env.get("enable_potential_shaping_reward", False))
        # an option against the standard is refused; options the standard does not name stay free
        with self.assertRaises(SystemExit):
            config("rl_std_bad", standard="v10_p1_S0", reward_preset="dense")
        with self.assertRaises(SystemExit):
            config("rl_std_bad2", standard="v10_p1_S0", history_mode="none")
        with self.assertRaises(SystemExit):
            config("rl_std_unknown", standard="v10_p9_S0")
        ppo, _, _ = config("rl_std_same", standard="v10_p1_S0", reward_preset="S0", save_freq=50)  # agreeing: fine
        self.assertEqual(ppo["data"]["train_batch_size"], 24)
        self.assertEqual(ppo["filter"]["name"], "mixed_outcome")
        # the dense-reward first batch: the keep_by_return_std filter with its threshold, floor and task file
        ppo, _, report = config("rl_keep", standard="v10_p1_D", group_filter="keep_by_return_std",
                                keep_std_threshold=0.33, keep_std_floor=1.77)
        kw = ppo["filter"]["filter_kwargs"]
        self.assertEqual((ppo["filter"]["name"], kw["target_groups"], kw["std_threshold"], kw["std_floor"]),
                         ("keep_by_return_std", 12, 0.33, 1.77))
        self.assertEqual(Path(kw["rows"]), (self.root / "rl_keep" / "train.jsonl").resolve())
        self.assertEqual((report["group_filter"], report["group_filter_kwargs"]["std_threshold"]), ("keep_by_return_std", 0.33))
        ppo, env, report = config("rl_keep_std", standard="v10_p1_S0_keep")  # a standard names the filter
        self.assertEqual((ppo["filter"]["name"], ppo["filter"]["filter_kwargs"]["std_threshold"],
                          ppo["filter"]["filter_kwargs"]["std_floor"], report["reward_preset"]),
                         ("keep_by_return_std", 0.134, 1.77, "S0"))
        with self.assertRaises(SystemExit):  # a threshold is required
            config("rl_keep_bad", standard="v10_p1_D", group_filter="keep_by_return_std")
        with self.assertRaises(SystemExit):  # it fills the package's batch
            config("rl_keep_bad2", ppo_template=str(Path(TP.__file__).with_name("configs") / "ppo_template_v10.yaml"),
                   env_template=str(Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"),
                   group_filter="keep_by_return_std", keep_std_threshold=0.3)
        with self.assertRaises(SystemExit):  # neither a template nor a standard
            config("rl_no_template")

    def test_rl_config_curriculum(self):
        """Round 12 batch: --curriculum-easy-updates N puts the easy-first sampler into the PPO config (the first N
        updates draw only tasks with d* <= max_dstar) and turns the dataloader workers off (verl's rule); it refuses
        when the train rows have too few easy tasks."""
        import yaml

        sft = self.root / "sft_rl_curr"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        rows = [json.loads(l) for l in (sft / "rows.jsonl").read_text().splitlines() if l.strip()]
        for r in rows:  # the train rows become reserve rows (what the standard trains on)
            r["reserve"] = r["split"] == "train"
        (sft / "rows_reserve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        easy = sum(1 for r in rows if r["split"] == "train" and int(r["difficulty"]["d_star"]) <= 3)
        self.assertGreater(easy, 0)

        def config(out_name, **kw):
            args = dict(rows=str(sft / "rows_reserve.jsonl"), gs_root="/ply", model="/m", name="t", standard="v10_p1_S0",
                        run_root=str(self.root / f"run_{out_name}"), val_limit=1, trial=False, out=str(self.root / out_name))
            args.update(kw)
            TP.cmd_rl_config(Namespace(**args))
            return (yaml.safe_load((self.root / out_name / "ppo.yaml").read_text()),
                    json.loads((self.root / out_name / "rl_config_report.json").read_text()))

        # one candidate per update here, so that the synthetic rows have enough easy tasks
        ppo, report = config("rl_curr", curriculum_easy_updates=easy, curriculum_max_dstar=3, candidates=1, target_groups=1)
        self.assertEqual(ppo["data"]["sampler"], {"class_path": "pkg://vagen.curriculum_sampler",
                                                  "class_name": "DifficultyCurriculumSampler", "easy_updates": easy,
                                                  "max_dstar": 3})
        self.assertEqual(ppo["data"]["dataloader_num_workers"], 0)
        self.assertEqual((report["curriculum"]["easy_tasks"], report["curriculum"]["easy_draws"]), (easy, easy))
        with self.assertRaises(SystemExit):  # more easy draws than easy tasks
            config("rl_curr_bad", curriculum_easy_updates=easy + 1, candidates=1, target_groups=1)
        ppo, report = config("rl_nocurr")  # without the option nothing changes
        self.assertIsNone(report["curriculum"])
        self.assertNotEqual((ppo["data"].get("sampler") or {}).get("class_name"), "DifficultyCurriculumSampler")
        # round 12 batch: --entropy-coeff replaces the package's entropy bonus (0 included); without it, the package's
        self.assertEqual((ppo["actor_rollout_ref"]["actor"]["entropy_coeff"], report["entropy_coeff"],
                          report["entropy_coeff_override"]), (0.01, 0.01, None))
        ppo, report = config("rl_noent", entropy_coeff=0.0)
        self.assertEqual((ppo["actor_rollout_ref"]["actor"]["entropy_coeff"], report["entropy_coeff"],
                          report["entropy_coeff_override"]), (0.0, 0.0, 0.0))
        self.assertEqual(ppo["actor_rollout_ref"]["actor"]["optim"]["lr"], 2e-6)  # the rest of the package stays

    def test_data_scale_subsets(self):
        """Round 12 batch: nested train-scene subsets (fixed seed) with all formal tasks of their scenes for SFT and
        only their scenes' reserve tasks for RL; an arm with one subset's task count over every scene."""
        records, rows = [], []
        counts = {f"sc{i:02d}": 2 + i % 5 for i in range(10)}  # 2..6 train records per scene
        for s, n in counts.items():
            for k in range(n):
                records.append({"id": f"{s}_{k}", "scene_id": s, "split": "train"})
            rows.append({"task_id": f"{s}_r", "scene_id": s, "split": "train", "reserve": True})
            rows.append({"task_id": f"{s}_f", "scene_id": s, "split": "train", "reserve": False})
        records += [{"id": f"v{k}", "scene_id": "val0", "split": "val"} for k in range(3)]
        rows += [{"task_id": "v_f", "scene_id": "val0", "split": "val", "reserve": False},
                 {"task_id": "v_r", "scene_id": "val0", "split": "val", "reserve": True}]
        res = TP.build_subsets(records, rows, [2, 4, 8], equal_tasks_of=4, flagged=["sc03_r"], seed=7)
        sub = res["subsets"]
        self.assertEqual(sorted(sub), ["s002", "s004", "s008", "w10"])
        order = res["scene_order"]
        self.assertEqual(sorted(order), sorted(counts))
        self.assertEqual(sub["s002"]["scenes"], sorted(order[:2]))
        self.assertTrue(set(sub["s002"]["scenes"]) < set(sub["s004"]["scenes"]) < set(sub["s008"]["scenes"]))  # nested
        for name, s in sub.items():
            self.assertEqual(sum(1 for r in s["sft"] if r["split"] == "val"), 3)  # every val record stays
            train_scenes = {r["scene_id"] for r in s["sft"] if r["split"] == "train"}
            self.assertLessEqual(train_scenes, set(s["scenes"]))
            reserve = [r for r in s["rows"] if r["reserve"]]
            self.assertTrue(all(r["scene_id"] in s["scenes"] and r["split"] == "train" for r in reserve))
            self.assertNotIn("sc03_r", {r["task_id"] for r in reserve})  # flagged tasks never reach RL
            self.assertNotIn("v_r", {r["task_id"] for r in s["rows"]})
            self.assertIn("v_f", {r["task_id"] for r in s["rows"]})  # rl-config needs the formal val rows
        self.assertEqual(sub["s004"]["train_records"], sum(counts[s] for s in order[:4]))
        w = sub["w10"]
        self.assertEqual(w["train_records"], sub["s004"]["train_records"])  # same task count, all scenes
        per = {}
        for r in w["sft"]:
            if r["split"] == "train":
                per[r["scene_id"]] = per.get(r["scene_id"], 0) + 1
        self.assertEqual(len(per), 10)
        self.assertLessEqual(max(per.values()) - min(per.values()), 2)
        self.assertEqual(res, TP.build_subsets(list(reversed(records)), rows, [2, 4, 8], 4, ["sc03_r"], seed=7))  # fixed
        self.assertNotEqual(TP.build_subsets(records, rows, [2], seed=8)["scene_order"], order)
        with self.assertRaises(ValueError):
            TP.build_subsets(records, rows, [11])
        # quotas: as even as capacities allow
        self.assertEqual(TP.spread_quota({"a": 1, "b": 5, "c": 5}, 9, seed=1), {"a": 1, "b": 4, "c": 4})
        with self.assertRaises(ValueError):
            TP.spread_quota({"a": 1}, 2)
        # the command: files per subset, the source SFT folder linked in
        out = self.root / "subsets_cmd"
        src = self.root / "subsets_src"
        src.mkdir()
        (src / "frames.jsonl").write_text("")
        (src / "split_manifest.json").write_text("{}")
        (src / "traj.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (src / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        flags = src / "flags.json"
        flags.write_text(json.dumps({"flagged_task_ids": ["sc03_r"]}))
        import os

        try:  # symbolic links need a privilege on Windows; the cluster (Linux) has them
            os.symlink(src / "frames.jsonl", self.root / "symlink_probe")
            can_link = True
        except OSError:
            can_link = False
        TP.cmd_subsets(Namespace(trajectories=str(src / "traj.jsonl"), rows=str(src / "rows.jsonl"), exclude=str(flags),
                                 sizes="2,4", equal_tasks_of=4, seed=7, link_from=str(src) if can_link else None,
                                 out=str(out)))
        summary = json.loads((out / "subsets.json").read_text())
        self.assertEqual(summary["subsets"]["s004"]["train_records"], sub["s004"]["train_records"])
        self.assertEqual(len((out / "sft_trajectories_w10.jsonl").read_text().splitlines()), w["train_records"] + 3)
        self.assertTrue((out / "rows_s002.jsonl").is_file())
        if can_link:
            self.assertTrue((out / "frames.jsonl").exists() and (out / "split_manifest.json").exists())

    def test_rl_config_trains_on_formal_tasks_only(self):
        """Reserve rows (sft-frames --include-reserve) are SFT-only: rl-config leaves them out of train and val."""
        import yaml

        sft = self.root / "sft_rl_reserve"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        rows = TP._jsonl(sft / "rows.jsonl")
        marked = [dict(r, reserve=(i % 2 == 1)) for i, r in enumerate(rows)]
        (sft / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in marked))
        template = self.root / "ppo_template_reserve.yaml"
        template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False},
            "actor_rollout_ref": {"model": {"path": "old"}, "actor": {"ppo_mini_batch_size": 8},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4, "agent": {"num_workers": 8}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))
        out = self.root / "rl_reserve"
        TP.cmd_rl_config(Namespace(rows=str(sft / "rows.jsonl"), env_template=str(
            Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"), ppo_template=str(template),
            gs_root="/ply", model="/m", name="t", run_root=str(self.root / "rl_run_reserve"), val_limit=0,
            trial=False, out=str(out)))
        train = TP._jsonl(out / "train.jsonl")
        want = [r["task_id"] for r in marked if r["split"] == "train" and not r["reserve"]]
        self.assertEqual([r["task_id"] for r in train], want)
        self.assertTrue(want)
        self.assertFalse(any(r.get("reserve") for r in TP._jsonl(out / "val.jsonl")))
        report = json.loads((out / "rl_config_report.json").read_text())
        self.assertEqual(report["reserve_rows_skipped"], sum(1 for r in marked if r["reserve"]))
        self.assertEqual(report["task_pool"], "formal")

        # round 11: --task-pool reserve trains on the reserve tasks only (val stays formal); the SFT conversations
        # and the depth-check flags are checked against the train tasks
        reserve = [r for r in marked if r["split"] == "train" and r["reserve"]]
        formal = [r for r in marked if r["split"] == "train" and not r["reserve"]]
        sft_conv = self.root / "sft_conv_formal.jsonl"
        sft_conv.write_text("".join(json.dumps({"source_task_id": r["task_id"]}) + "\n" for r in formal))
        flags = self.root / "depth_flags.json"
        flags.write_text(json.dumps({"flagged_task_ids": ["not_a_train_task"]}))

        def config(out_name, **kw):
            TP.cmd_rl_config(Namespace(rows=str(sft / "rows.jsonl"), env_template=str(
                Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"), ppo_template=str(template),
                gs_root="/ply", model="/m", name="t", run_root=str(self.root / "rl_run_pool"), val_limit=0,
                trial=False, out=str(self.root / out_name), **kw))
            return self.root / out_name

        out = config("rl_pool_reserve", task_pool="reserve", sft_trajectories=[str(sft_conv)], exclude=str(flags))
        self.assertEqual([r["task_id"] for r in TP._jsonl(out / "train.jsonl")], [r["task_id"] for r in reserve])
        self.assertTrue(reserve)
        self.assertFalse(any(r.get("reserve") for r in TP._jsonl(out / "val.jsonl")))
        report = json.loads((out / "rl_config_report.json").read_text())
        self.assertEqual((report["task_pool"], report["sft_overlap"], report["sft_tasks_checked"],
                          report["depth_flagged_in_train"], report["depth_flagged_checked"]),
                         ("reserve", 0, len(formal), 0, 1))
        self.assertEqual(sum(report["by_family_train"].values()), len(reserve))
        # a train task that SFT trained on, or one the depth check flagged, stops the configuration
        with self.assertRaises(SystemExit):
            config("rl_pool_formal_vs_sft", task_pool="formal", sft_trajectories=[str(sft_conv)])
        flags.write_text(json.dumps({"flagged_task_ids": [reserve[0]["task_id"]]}))
        with self.assertRaises(SystemExit):
            config("rl_pool_flagged", task_pool="reserve", exclude=str(flags))

    def test_rl_config_format_example(self):
        """Round 12 freeze (the user's 3.1.1): --format-example sets the env's format line (placeholder names or
        real action names; default the SFT export's); the start model's SFT conversations must show the same text."""
        import yaml

        from vagen.envs.active_spatial import prompt as P

        sft = self.root / "sft_rl_format"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        rows = [dict(r, reserve=(r["split"] == "train" and i % 2 == 1)) for i, r in enumerate(TP._jsonl(sft / "rows.jsonl"))]
        (sft / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        template = self.root / "ppo_template_format.yaml"
        template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False},
            "actor_rollout_ref": {"model": {"path": "old"}, "actor": {"ppo_mini_batch_size": 8},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4, "agent": {"num_workers": 8}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))
        formal = [r for r in rows if r["split"] == "train" and not r["reserve"]]
        convs = {}
        for fmt in P.FORMAT_EXAMPLES:  # the first user turn ends with the format text, as release._user_turn writes it
            block = P.format_prompt["no_think"](max_actions_per_step=5, action_sep="|", add_example=True,
                                                format_example=fmt)
            convs[fmt] = self.root / f"sft_conv_{fmt}.jsonl"
            convs[fmt].write_text("".join(json.dumps({"source_task_id": r["task_id"], "conversations": [
                {"role": "system", "content": "s"}, {"role": "user", "content": "[Initial Observation]:\n<image>\n" + block},
                {"role": "assistant", "content": "<action>move_forward|</action>"}]}) + "\n" for r in formal))
        self.assertEqual(TP.sft_format_examples([convs["placeholder"]]), ["placeholder"])
        self.assertEqual(TP.sft_format_examples([convs["action_names"]]), ["action_names"])
        self.assertEqual(TP.sft_format_examples([convs["placeholder"], convs["action_names"]]),
                         ["action_names", "placeholder"])

        def config(out_name, **kw):
            TP.cmd_rl_config(Namespace(rows=str(sft / "rows.jsonl"), env_template=str(
                Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"), ppo_template=str(template),
                gs_root="/ply", model="/m", name="t", run_root=str(self.root / f"run_{out_name}"), val_limit=0,
                trial=False, out=str(self.root / out_name), task_pool="reserve", **kw))
            out = self.root / out_name
            return (yaml.safe_load((out / "train_env.yaml").read_text())["envs"][0]["config"],
                    json.loads((out / "rl_config_report.json").read_text()))

        # default: the SFT export's (the policy names none: the placeholder text); nothing new in the env config
        env, report = config("rl_fmt_default", sft_trajectories=[str(convs["placeholder"])])
        self.assertNotIn("prompt_format_example", env)
        self.assertEqual((report["format_example"], report["format_example_sft"]), ("placeholder", ["placeholder"]))
        # real action names for a start model whose SFT conversations show them
        env, report = config("rl_fmt_names", sft_trajectories=[str(convs["action_names"])], format_example="action_names")
        self.assertEqual(env["prompt_format_example"], "action_names")
        self.assertEqual((report["format_example"], report["format_example_sft"]), ("action_names", ["action_names"]))
        self.assertIn("Response format: <action>move_forward|turn_left|</action>",
                      TP.env_system_prompt(env, "fov_inclusion"))
        # refused: the RL text against the start model's conversations, or (no conversations) against the policy
        with self.assertRaises(SystemExit):
            config("rl_fmt_wrong", sft_trajectories=[str(convs["placeholder"])], format_example="action_names")
        with self.assertRaises(SystemExit):
            config("rl_fmt_wrong2", sft_trajectories=[str(convs["action_names"])])
        with self.assertRaises(SystemExit):
            config("rl_fmt_policy", format_example="action_names")
        # round 12 freeze (the user's 3.1.2): --lenient-parse turns on the env's lenient reading, nothing else
        env, report = config("rl_lenient", sft_trajectories=[str(convs["placeholder"])], lenient_parse=True)
        self.assertTrue(env["lenient_parse"])
        self.assertTrue(report["lenient_parse"])
        env0, report0 = config("rl_strict", sft_trajectories=[str(convs["placeholder"])])
        self.assertNotIn("lenient_parse", env0)
        self.assertFalse(report0["lenient_parse"])
        self.assertEqual({k: v for k, v in env.items() if k not in ("lenient_parse", "jsonl_path")},
                         {k: v for k, v in env0.items() if k != "jsonl_path"})
        self.assertIn("lenient_parse", TP.STANDARD_KEYS)
        # 3.1.3: --invalid-penalty sets the penalty the env applies; the prompt (checked against the SFT one) keeps -0.1
        env5, report5 = config("rl_invalid", sft_trajectories=[str(convs["placeholder"])], invalid_penalty=-0.5)
        self.assertEqual((env5["invalid_penalty_override"], env5["invalid_format_penalty"]), (-0.5, -0.1))
        self.assertEqual((report5["invalid_penalty"], report0["invalid_penalty"]), (-0.5, -0.1))
        self.assertNotIn("invalid_penalty_override", env0)
        self.assertEqual(TP.env_system_prompt(env5, "fov_inclusion"), TP.env_system_prompt(env0, "fov_inclusion"))

    def test_rl_config_round10_options(self):
        """History modes, the fixed validation set, greedy validation, test frequency and critic warm-up."""
        import yaml

        sft = self.root / "sft_rl10"
        TP.cmd_sft_frames(Namespace(run=str(self.run_dir), split_manifest=str(self.manifest), splits="train,val",
                                    limit=0, out=str(sft)))
        template = self.root / "ppo_template10.yaml"
        template.write_text(yaml.safe_dump({
            "algorithm": {"adv_estimator": "no_concat_gae"}, "data": {"train_batch_size": 12, "val_batch_size": 8,
                                                                      "max_response_length": 384},
            "trainer": {"n_gpus_per_node": 8, "concat_multi_turn": False, "critic_warmup": 60, "test_freq": 50},
            "actor_rollout_ref": {"model": {"path": "old"}, "actor": {"ppo_mini_batch_size": 8},
                                  "rollout": {"n": 4, "tensor_model_parallel_size": 4,
                                              "val_kwargs": {"n": 4, "do_sample": True, "temperature": 0.8},
                                              "agent": {"num_workers": 8, "agent_loop_config_path": "x.yaml"}}},
            "critic": {"model": {"path": "old", "tokenizer_path": "old"}}}))
        val_rows = [r for r in TP._jsonl(sft / "rows.jsonl") if r["split"] == "val"]
        val_tasks = self.root / "val_tasks.jsonl"
        val_tasks.write_text("".join(json.dumps(r) + "\n" for r in val_rows[:1]))
        env_template = Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"
        self.assertEqual(TP.history_overrides("none", 12), {"env": {}, "ppo": {}})
        self.assertEqual(TP.history_overrides("no_concat", 12)["env"], {"history_window_size": 12})
        with self.assertRaises(ValueError):
            TP.history_overrides("window4", 12)
        base = dict(rows=str(sft / "rows.jsonl"), env_template=str(env_template), ppo_template=str(template),
                    gs_root="/ply", model="/m", name="t", val_limit=0, trial=False)
        for mode in ("no_concat", "concat"):
            out = self.root / f"rl10_{mode}"
            TP.cmd_rl_config(Namespace(**base, run_root=str(self.root / f"run_{mode}"), out=str(out), history_mode=mode,
                                       val_tasks=str(val_tasks), greedy_val=True, test_freq=25, critic_warmup=10))
            ppo = yaml.safe_load((out / "ppo.yaml").read_text())
            env = yaml.safe_load((out / "train_env.yaml").read_text())["envs"][0]
            val_env = yaml.safe_load((out / "val_env.yaml").read_text())["envs"][0]
            self.assertEqual((ppo["trainer"]["test_freq"], ppo["trainer"]["critic_warmup"]), (25, 10))
            self.assertEqual(ppo["actor_rollout_ref"]["rollout"]["val_kwargs"],
                             {"n": 1, "do_sample": False, "temperature": 0.0})
            self.assertEqual((val_env["n_envs"], ppo["data"]["val_batch_size"]), (1, 1))  # the fixed set, one batch
            self.assertEqual(TP._jsonl(out / "val.jsonl")[0]["task_id"], val_rows[0]["task_id"])
            if mode == "no_concat":
                self.assertEqual((env["config"]["history_window_size"], val_env["config"]["history_window_size"]), (12, 12))
                self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["trainer"]["concat_multi_turn"]),
                                 ("no_concat_gae", False))
                self.assertEqual(ppo["data"]["max_response_length"], 384)
            else:
                self.assertNotIn("history_window_size", env["config"])  # the chat history carries the turns
                self.assertEqual((ppo["algorithm"]["adv_estimator"], ppo["trainer"]["concat_multi_turn"]), ("gae", True))
                self.assertEqual(ppo["actor_rollout_ref"]["rollout"]["agent"]["agent_loop_config_path"],
                                 "vagen/configs/agent.yaml")
                self.assertEqual(ppo["data"]["max_response_length"], TP.CONCAT_RESPONSE_LENGTH)
            report = json.loads((out / "rl_config_report.json").read_text())
            self.assertEqual((report["history_mode"], report["greedy_val"], report["val"]), (mode, True, 1))
        # without the options nothing changes (rounds 6-9 behaviour)
        out = self.root / "rl10_plain"
        TP.cmd_rl_config(Namespace(**base, run_root=str(self.root / "run_plain"), out=str(out)))
        ppo = yaml.safe_load((out / "ppo.yaml").read_text())
        self.assertEqual((ppo["trainer"]["test_freq"], ppo["trainer"]["critic_warmup"],
                          ppo["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"], ppo["data"]["val_batch_size"]),
                         (25, 60, 4, 8))


if __name__ == "__main__":
    unittest.main()
