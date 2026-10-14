"""rl_checks.py: SFT conversations, shortest sequences and model-free policies in the RL env (S0 config), on a
synthetic scene with a stub renderer."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10 import rl_checks as RC
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.pilot import safe_name
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

# one fixed textured frame: the env's renderer is replaced by it, and the "rendered" SFT frames are this image
FRAME = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))


class RlChecksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        scene_dir = paths["two_rooms"]
        cls.rows = _jsonable(generate_scene(scene_dir, scene_dir.name, RunConfig(target_tasks_per_scene=5)))["rows"]
        cls.cfg = RC.env_config(Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml", "http://127.0.0.1:1/render")
        cls.cfg["gs_root"] = str(scene_dir.parent)
        cls.patch = mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image", lambda self: FRAME.copy())
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls._tmp.cleanup()

    def test_shortest_sequences_succeed_at_d_star_with_the_s0_reward(self):
        res = RC.shortest_replay(self.rows, self.root / "short", self.cfg)
        s = RC.summarize_shortest(res)
        self.assertEqual((s["errors"], s["success"], s["success_at_d_star"], s["done_on_last_planned_turn"]),
                         (0, len(self.rows), len(self.rows), len(self.rows)))
        self.assertEqual(s["max_reconstruction_error"], 0.0)
        # S0: +5 for success on the last turn, -0.01 per turn, nothing else
        self.assertEqual({k.split(":")[1] for k in s["rewards"] if k.startswith("last")}, {"4.99"})
        self.assertTrue(all(k.endswith(":-0.01") for k in s["rewards"] if k.startswith("earlier")))

    def test_baselines(self):
        res = RC.baselines(self.rows[:2], self.root / "base", self.cfg, seed=1)
        s = RC.summarize_baselines(res)
        self.assertEqual(sorted(s), sorted(RC.POLICIES))
        for policy, v in s.items():
            self.assertEqual(v["tasks"], 2)
            self.assertEqual(v["fewer_turns_than_d_star"], 0)  # d* is a lower bound

    def test_sft_conversations_match_the_env(self):
        run = self.root / "run"
        run.mkdir(exist_ok=True)
        (run / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        manifest = self.root / "split.json"
        manifest.write_text(json.dumps({"version": 2, "train_scenes": [self.rows[0]["scene_id"]], "val_scenes": [],
                                        "test_scenes": []}))
        sft = self.root / "sft"
        TP.cmd_sft_frames(Namespace(run=str(run), split_manifest=str(manifest), splits="train", limit=0, out=str(sft)))
        (sft / "frames" / "plain").mkdir(parents=True)
        for f in TP._jsonl(sft / "frames.jsonl"):
            FRAME.save(sft / "frames" / "plain" / safe_name(f["frame_id"]))
        TP.cmd_sft_records(Namespace(dir=str(sft)))
        records = TP._jsonl(sft / "sft_trajectories.jsonl")
        rows = {r["task_id"]: r for r in TP._jsonl(sft / "rows.jsonl")}
        res = RC.sft_replay(records, rows, self.root / "sftcheck", self.cfg, Image.open)
        s = RC.summarize_sft(res)
        n = len(records)
        self.assertEqual((s["errors"], s["system_identical"], s["final_success"], s["final_done"], s["early_done"]),
                         (0, n, n, n, 0))
        # every user turn: the same image and, with the pose lines from the env's own pose replay, the same text
        self.assertEqual((s["text_identical"], s["pose_identical"], s["pixel_max"]), (s["user_turns"], s["user_turns"], 0))
        self.assertEqual(s["ends_with"], {"assistant": n})


if __name__ == "__main__":
    unittest.main()
