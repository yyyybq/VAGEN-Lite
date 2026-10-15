"""evaluation.py: the fixed test subset and the metrics table (the episodes themselves need a GPU)."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import evaluation as E


def _row(i, fam, d):
    return {"task_id": f"t{i:04d}", "task_type": fam, "scene_id": f"s{i % 7}", "difficulty": {"d_star": d}}


class EvaluationTest(unittest.TestCase):
    def test_select_is_stratified_and_fixed(self):
        rows = [_row(i, "projective_relations", 1 + i % 8) for i in range(200)]
        rows += [_row(1000 + i, "occlusion_alignment", 1 + i % 3) for i in range(20)]   # short of tasks
        rows += [_row(2000 + i, "equidistance", 7 + i % 2) for i in range(80)]          # one band only
        a = E.select_tasks(rows, 30, seed=5)
        b = E.select_tasks(list(reversed(rows)), 30, seed=5)
        self.assertEqual([r["task_id"] for r in a], [r["task_id"] for r in b])  # order of the input is irrelevant
        fam = {f: [r for r in a if r["task_type"] == f] for f in ("projective_relations", "occlusion_alignment",
                                                                   "equidistance")}
        self.assertEqual({f: len(v) for f, v in fam.items()},
                         {"projective_relations": 30, "occlusion_alignment": 20, "equidistance": 30})
        bands = [E.band_name(r["difficulty"]["d_star"]) for r in fam["projective_relations"]]
        self.assertEqual(sorted(bands.count(b) for b in ("1-3", "4-6", "7-8")), [10, 10, 10])
        self.assertNotEqual([r["task_id"] for r in a], [r["task_id"] for r in E.select_tasks(rows, 30, seed=6)])

    def test_metrics_and_table(self):
        rows = [{"task_type": "fov_inclusion", "d_star": 2, "success": True, "actions_to_success": 4, "turns": 2,
                 "actions": 4, "collisions": 1, "invalid_turns": 0},
                {"task_type": "fov_inclusion", "d_star": 8, "success": False, "actions_to_success": None, "turns": 12,
                 "actions": 12, "collisions": 0, "invalid_turns": 3},
                {"index": 2, "error": "renderer"}]
        m = E.metrics(rows)
        self.assertEqual((m["episodes"], m["errors"], m["success"]), (2, 1, 0.5))
        self.assertEqual(m["success_by_band"], {"1-3": 1.0, "7-8": 0.0})
        self.assertEqual(m["actions_over_dstar"], 2.0)
        self.assertEqual((m["collision_rate"], m["invalid_rate"]), (0.0625, round(3 / 14, 4)))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "sft"
            d.mkdir()
            (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            E.cmd_table(type("A", (), {"eval": tmp})())
            table = (Path(tmp) / "table.md").read_text(encoding="utf-8")
        self.assertIn("| sft | 2 | 0.500 |", table)

    def test_run_episode_records_the_trajectory(self):
        from PIL import Image

        class Env:  # two turns: a blocked move, then success at the third action
            current_item = {"task_id": "t1", "scene_id": "s1", "task_type": "fov_inclusion", "task_description": "Go.",
                            "difficulty": {"d_star": 2}}
            _current_step = 0

            def _obs(self, shade):
                return {"obs_str": f"<image>\nturn {self._current_step}",
                        "multi_modal_data": {"<image>": [Image.new("RGB", (8, 8), (shade, 0, 0))]}}

            def reset(self, seed):
                return self._obs(10), {}

            def system_prompt(self):
                return "sys"

            def step(self, text):
                acts = text.split("|")
                self._current_step += len(acts)
                done = self._current_step >= 3
                info = {"actions": acts, "metrics": {"turn_metrics": {"collision_count": int(not done),
                                                                      "action_is_valid": True}},
                        "termination_reason": "success_auto" if done else "continuing"}
                return self._obs(200 if done else 90), (4.99 if done else -0.16), done, info

            def _calculate_canonical_metric(self):
                ok = self._current_step >= 3
                return {"success": ok, "gates": {"distance_ok": ok, "target_a_clearly_visible": True},
                        "continuous_gate_progress": min(1.0, self._current_step / 3)}

        answers = iter(["move_forward|turn_left", "move_forward"])
        policy = type("P", (), {"generate": lambda self, s, u, im: next(answers)})()
        with tempfile.TemporaryDirectory() as tmp:
            row = E.run_episode(Env(), policy, 7, 12, frames=Path(tmp) / "frames")
            files = sorted(p.name for p in (Path(tmp) / "frames").iterdir())
        self.assertEqual(files, ["007_00.jpg", "007_01.jpg", "007_02.jpg"])
        self.assertEqual((row["success"], row["actions_to_success"], row["turns"], row["collisions"]), (True, 3, 2, 1))
        t0, t1 = row["trajectory"]
        self.assertEqual((t0["frame"], t0["user"], t0["answer"]), ("frames/007_00.jpg", "<image>\nturn 0",
                                                                    "move_forward|turn_left"))
        self.assertEqual((t0["actions"], t0["collisions"], t0["reward"], t0["met"]),
                         (["move_forward", "turn_left"], 1, -0.16, False))
        self.assertEqual((t1["met"], row["final_frame"], row["end_reason"], row["task_description"]),
                         (True, "frames/007_02.jpg", "success_auto", "Go."))
        self.assertEqual(row["start"], {"failed_gates": ["distance_ok"], "progress": 0.0})  # before the first turn
        self.assertEqual((t0["failed_gates"], t0["progress"], t1["failed_gates"], t1["progress"]),
                         (["distance_ok"], 0.6667, [], 1.0))
        answers = iter(["move_forward|turn_left", "move_forward"])
        self.assertNotIn("trajectory", E.run_episode(Env(), policy, 7, 12))  # without a frames folder: counts only

    def test_switches_reach_the_env_config(self):
        import argparse
        import os
        from unittest import mock

        template = Path(E.__file__).with_name("configs") / "rl_env_v10_S0.yaml"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"R1_RENDER_URL": "http://x/render"}):
            tasks = Path(tmp) / "tasks.jsonl"
            tasks.write_text("")
            plain = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace()))
            self.assertEqual(E.switch_overrides(argparse.Namespace()), {})  # no switch: the template's rules
            self.assertNotIn("success_check_each_action", plain)
            on = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace(check_each_action=True)))
            self.assertTrue(on["success_check_each_action"])
            self.assertEqual({k: v for k, v in on.items() if k != "success_check_each_action"}, plain)

    def test_replace_flagged(self):
        picked = [_row(i, "projective_relations", 2) for i in range(4)] + [_row(10, "occlusion_alignment", 8)]
        pool = picked + [_row(20 + i, "projective_relations", 1 + i % 3) for i in range(3)] + [_row(30, "projective_relations", 7)]
        res = E.replace_flagged(picked, ["t0001", "t0010", "t0021"], pool, seed=3)
        ids = [r["task_id"] for r in res["tasks"]]
        self.assertEqual(len(ids), 4)                         # the OA task had no substitute
        self.assertNotIn("t0001", ids)
        self.assertNotIn("t0021", ids)                        # flagged tasks never come back in
        swap = next(s for s in res["swaps"] if s["out"] == "t0001")
        self.assertIn(swap["in"], {"t0020", "t0022"})          # same family and d* band (1-3), unflagged
        self.assertIsNone(next(s for s in res["swaps"] if s["out"] == "t0010")["in"])
        self.assertEqual(res, E.replace_flagged(picked, ["t0001", "t0010", "t0021"], list(reversed(pool)), seed=3))


if __name__ == "__main__":
    unittest.main()
