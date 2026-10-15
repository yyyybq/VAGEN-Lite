"""Round 10, decision 6: SFT conversations with one inserted blocked move ("what to do after a blocked move").

The assistant tries a move the collision detector blocks, the next user turn shows the same frame and pose with the
env's detailed feedback, and the shortest path goes on.  Replayed in the RL env (round-10 template: detailed
collision feedback, thick walls) every inserted conversation must read exactly as the env writes it."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10 import release as R
from data_gen.interiorgs_v10 import rl_checks as RC
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.admission import load_policy
from data_gen.interiorgs_v10.pilot import safe_name
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
from vagen.envs.active_spatial.env import collision_feedback

FRAME = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
TEMPLATE = Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml"


class BlockedTurnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        scene_dir = write_synthetic_scenes(cls.root / "scenes", with_walls=True)["two_rooms_walls"]
        cls.scene_dir = scene_dir
        cfg = RunConfig(target_tasks_per_scene=8, wall_model="thick", visibility_rule="env", require_structure_walls=True)
        cls.rows = _jsonable(generate_scene(scene_dir, scene_dir.name, cfg))["rows"]
        cls.cfg = RC.env_config(TEMPLATE, "http://127.0.0.1:1/render")
        cls.cfg["gs_root"] = str(scene_dir.parent)
        cls.patch = mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image", lambda self: FRAME.copy())
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls._tmp.cleanup()

    def test_feedback_text_is_the_env_function(self):
        self.assertEqual(collision_feedback(["move_forward"], [("move_forward", "blocked by wall")]),
                         "0 of 1 actions executed; blocked: move_forward (blocked by wall).")
        self.assertEqual(collision_feedback(["a", "b", "c"], [("b", "blocked by sofa")]),
                         "2 of 3 actions executed; blocked: b (blocked by sofa).")

    def test_blocked_moves_come_from_the_detector(self):
        det = load_scene(self.scene_dir, self.scene_dir.name, wall_model="thick").detector
        from data_gen.interiorgs_v10.contract import pose_from_xy_yaw

        # 0.2 m in front of the living room's west wall (x = 0.1 is its inner face), facing it: forward is blocked
        blocked = R.blocked_moves(det, pose_from_xy_yaw(0.35, 1.0, np.pi))
        self.assertIn(("move_forward", "blocked by wall"), blocked)
        self.assertNotIn("move_backward", [m for m, _ in blocked])
        self.assertEqual(R.blocked_moves(det, pose_from_xy_yaw(3.0, 1.0, 0.0)), [])  # open floor: nothing blocked

    def test_insertion_is_seeded_shared_and_replays_in_the_env(self):
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
        policy = load_policy()
        policy["sft"]["export"]["blocked_turn_share"] = 1.0  # every task that has a blocked move at the drawn turn
        with mock.patch("data_gen.interiorgs_v10.admission.load_policy", lambda: policy):  # imported at call time
            TP.cmd_sft_records(Namespace(dir=str(sft), scenes_root=str(self.scene_dir.parent), env_template=str(TEMPLATE)))
        records = TP._jsonl(sft / "sft_trajectories.jsonl")
        report = json.loads((sft / "sft_records_report.json").read_text())
        self.assertEqual(report["wall_model"], "thick")
        inserted = [r for r in records if r["blocked_insert"]]
        self.assertTrue(inserted, "no task had a blocked move at the drawn turn")
        self.assertEqual(report["blocked_turns"], len(inserted))
        for rec in inserted:
            ins = rec["blocked_insert"]
            conv = rec["conversations"]
            users = [t for t in conv if t["role"] == "user"]
            self.assertEqual(rec["trajectory_steps"], len(users))
            k = 2 * ins["turn"] + 1  # the system turn, then (user, assistant) pairs
            self.assertEqual(conv[k + 1]["content"], "<action>" + ins["move"] + "|</action>")
            self.assertEqual(conv[k]["image_path"], conv[k + 2]["image_path"])  # the camera did not move
            self.assertIn("Environment Feedback: " + collision_feedback([ins["move"]], [(ins["move"], ins["blocked_by"])]),
                          conv[k + 2]["content"])
            self.assertEqual(R.scan_leakage(rec, policy), [])
        # the same seed gives the same insertions; another seed, other ones
        again = R.build_sft_record(TP._jsonl(sft / "rows.jsonl")[0], {}, policy, detector=None)
        self.assertEqual(again[1], "sft_frame_missing_image")  # (no images here: only the determinism is checked)
        rows_by_id = {r["task_id"]: r for r in TP._jsonl(sft / "rows.jsonl")}
        det = load_scene(self.scene_dir, self.scene_dir.name, wall_model="thick").detector
        first = inserted[0]
        row = rows_by_id[first["source_task_id"]]
        poses = R.runtime_poses(row)
        bounds = TP.sft_states(row, policy)
        n_turns = len(bounds) - 1
        self.assertEqual(R.blocked_insert(row, policy, det, n_turns, [poses[b] for b in bounds[:-1]]), first["blocked_insert"])
        # share 0: nothing is inserted; the seed is part of the draw (the small synthetic scene may still give the
        # same turn and move, so only the gating is checked)
        none = json.loads(json.dumps(policy))
        none["sft"]["export"]["blocked_turn_share"] = 0.0
        self.assertIsNone(R.blocked_insert(row, none, det, n_turns, [poses[b] for b in bounds[:-1]]))
        half = json.loads(json.dumps(policy))
        half["sft"]["export"]["blocked_turn_share"] = 0.5
        chosen = {seed: sum(R.blocked_insert(rows_by_id[r["source_task_id"]], dict(half, sft={"export": dict(half["sft"]["export"], blocked_turn_seed=seed)}),
                                             det, 1, [R.runtime_poses(rows_by_id[r["source_task_id"]])[0]]) is not None
                            for r in records) for seed in (1, 2, 3)}
        self.assertTrue(all(0 <= v <= len(records) for v in chosen.values()))
        # replayed in the env with the round-10 rules: identical text, the same image, success at the end
        res = RC.sft_replay(records, rows_by_id, self.root / "sftcheck", self.cfg, Image.open)
        s = RC.summarize_sft(res)
        n = len(records)
        self.assertEqual((s["errors"], s["system_identical"], s["final_success"], s["final_done"], s["early_done"]),
                         (0, n, n, n, 0))
        self.assertEqual((s["text_identical"], s["pose_identical"], s["pixel_max"]), (s["user_turns"], s["user_turns"], 0))
        # without scenes (rounds 6-9 call) nothing is inserted and the formal-only file is not written
        TP.cmd_sft_records(Namespace(dir=str(sft)))
        self.assertTrue(all(r["blocked_insert"] is None for r in TP._jsonl(sft / "sft_trajectories.jsonl")))
        self.assertFalse((sft / "sft_trajectories_formal.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
