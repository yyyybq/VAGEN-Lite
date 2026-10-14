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
        prep.prepare(Namespace(out=str(self.root / "lf_steps"), allow_unreviewed_trajectories=True, **common))
        self.assertNotIn("save_strategy", yaml.safe_load((self.root / "lf_steps" / "train.yaml").read_text()))

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


if __name__ == "__main__":
    unittest.main()
