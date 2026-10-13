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
        manifest = prep.prepare(Namespace(out=str(self.root / "lf"), allow_unreviewed_trajectories=True, **common))
        self.assertEqual(manifest["trajectory_semantic_review"], "not_required")
        self.assertEqual(manifest["counts"]["trajectory/train"], self.n_train)
        self.assertEqual(manifest["counts"]["trajectory/val"], self.n_val)
        cfg = (self.root / "lf" / "train.yaml").read_text()
        self.assertIn("dataset: trajectory_train", cfg)
        self.assertNotIn("qa_train", cfg)


if __name__ == "__main__":
    unittest.main()
