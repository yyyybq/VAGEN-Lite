"""grounding_probe.py and evaluation.run_episode's --grounding-probe (the user's supplement of 2026-11-12, item 5)."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import evaluation as E
from data_gen.interiorgs_v10 import grounding_probe as G

# eval_val_v2 task 0 (absolute_positioning, the standing air conditioner 1.5 m, d* 2) and its start pose
AC = {"id": "145", "label": "vertical air conditioning", "ref": "the standing air conditioner",
      "bbox_min": [-1.1043389702580089, -3.525038083916621, 3.9e-08],
      "bbox_max": [-0.6995206297419911, -3.119246516083379, 1.8000000805685732],
      "center": [-0.9019298, -3.3221423000000003, 0.90000006]}
ITEM = {"task_id": "v10_90d6bf453173b12783d99329", "scene_id": "0339_840404", "task_type": "absolute_positioning",
        "difficulty": {"d_star": 2},
        "task_description": "Move to a spot about 1.5 meters away from the standing air conditioner (measured to its "
                            "nearest side), keeping the standing air conditioner in view.",
        "init_camera": {"intrinsics": [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
                        "extrinsics": [[-0.4969864615194961, 0.0, 0.8677582941501225, -1.8489284025897093],
                                       [-0.8677582941501226, 0.0, -0.496986461519496, -2.76611938717639],
                                       [0.0, -1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]},
        "target_region": {"type": "footprint_distance_band",
                          "params": {"distance_m": 1.5, "distance_definition": "horizontal_to_aabb_footprint",
                                     "object_a_id": "145", "object_a_center": AC["center"]}},
        "target_object": {"objects": [AC]}}
START = np.array(ITEM["init_camera"]["extrinsics"])


def facing_x(x=0.0, y=0.0):
    """A level OpenCV c2w at (x, y, 1.5) facing world +X: right = down x forward = -Y."""
    return np.array([[0.0, 0.0, 1.0, x], [-1.0, 0.0, 0.0, y], [0.0, -1.0, 0.0, 1.5], [0, 0, 0, 1.0]])


def box(cx, cy, half=0.1, oid="1", ref="the box"):
    return {"id": oid, "label": "box", "ref": ref, "bbox_min": [cx - half, cy - half, 0.0],
            "bbox_max": [cx + half, cy + half, 1.0], "center": [cx, cy, 0.5]}


class StatementTest(unittest.TestCase):
    def test_parse(self):
        p = G.parse_statement
        self.assertEqual(p("<state>bearing=front; distance=1.5</state>\n<action>move_forward|</action>"),
                         {"present": True, "bearing": "front", "distance": 1.5})
        self.assertEqual(p("<STATE> Bearing: Behind , distance = 2 m </STATE><action>turn_left|</action>"),
                         {"present": True, "bearing": "back", "distance": 2.0})
        self.assertEqual(p("<state>bearing=ahead; distance=0.75 meters</state>")["bearing"], "front")
        self.assertEqual(p("<state>bearing=front-left; distance=.5</state>"),
                         {"present": True, "bearing": None, "distance": 0.5})  # not one of the four: unreadable
        self.assertEqual(p("<state>bearing=left</state>"), {"present": True, "bearing": "left", "distance": None})
        self.assertEqual(p("<action>move_forward|</action>"), {"present": False, "bearing": None, "distance": None})
        self.assertEqual(p("<state>bearing=left; distance=1"), {"present": False, "bearing": None, "distance": None})
        self.assertEqual(p(None)["present"], False)
        # the first block counts
        self.assertEqual(p("<state>bearing=left; distance=1</state><state>bearing=right; distance=9</state>")
                         ["bearing"], "left")

    def test_answer_part_keeps_the_action_reading(self):
        from vagen.envs.active_spatial.utils import parse_free_think

        a = "<state>bearing=front; distance=1.5</state>\n<action>move_backward|move_backward|</action>"
        self.assertEqual(G.answer_part(a), "\n<action>move_backward|move_backward|</action>")
        self.assertEqual(parse_free_think(G.answer_part(a)), parse_free_think("<action>move_backward|move_backward|</action>"))
        plain = "<action>turn_left|</action>"
        self.assertEqual(G.answer_part(plain), plain)  # no statement: unchanged
        bad = "<state>bearing=<action>turn_left|</action></state><action>move_forward|</action>"
        self.assertEqual(parse_free_think(G.answer_part(bad))["actions_blob"], "move_forward|")  # only the action part

    def test_instruction_names_object_a(self):
        text = G.instruction(ITEM)
        self.assertTrue(text.startswith("\n\nBefore the action"))
        self.assertIn("<state>bearing=B; distance=D</state>", text)
        self.assertIn("B is where the standing air conditioner is now", text)
        two = {"target_region": {"params": {"object_a_id": "2"}},
               "target_object": {"objects": [box(0, 0, oid="1", ref="the first"), box(1, 1, oid="2", ref="the second")]}}
        self.assertEqual(G.object_ref(two), "the second")  # object_a_id wins over the order
        self.assertEqual(G.object_ref({"target_object": {"objects": [{"label": "lamp", "center": [0, 0, 0]}]}}),
                         "the lamp")
        with self.assertRaises(ValueError):
            G.object_a({"task_id": "x", "target_object": {"objects": []}})


class TruthTest(unittest.TestCase):
    def test_sectors_and_tolerance(self):
        self.assertEqual([G.sector(a) for a in (0, 45, 46, 135, 136, 180, -180, -45, -46, -135, -136, 225)],
                         ["front", "front", "right", "right", "back", "back", "back", "front", "left", "left", "back",
                          "left"])
        self.assertEqual(G.accepted_bearings(30.0), ["front"])
        self.assertEqual(G.accepted_bearings(40.0), ["front", "right"])
        self.assertEqual(G.accepted_bearings(54.0), ["front", "right"])
        self.assertEqual(G.accepted_bearings(56.0), ["right"])
        self.assertEqual(G.accepted_bearings(140.0), ["right", "back"])
        self.assertEqual(G.accepted_bearings(-50.0), ["front", "left"])
        self.assertEqual(G.accepted_bearings(None), [])

    def test_bearing_geometry(self):
        pose = facing_x()
        self.assertAlmostEqual(G.bearing_angle(pose, [2.0, 0.0, 0.5]), 0.0)
        self.assertAlmostEqual(G.bearing_angle(pose, [0.0, -2.0, 0.5]), 90.0)  # -Y is the camera's right
        self.assertAlmostEqual(G.bearing_angle(pose, [0.0, 2.0, 0.5]), -90.0)
        self.assertAlmostEqual(abs(G.bearing_angle(pose, [-2.0, 0.0, 0.5])), 180.0)
        self.assertAlmostEqual(G.bearing_angle(pose, [1.0, -1.0, 9.0]), 45.0)  # height does not matter
        pitched = pose.copy()  # looking 30 degrees down: the floor projection of +Z still faces +X
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        pitched[:3, 1] = [s, 0.0, -c]
        pitched[:3, 2] = [c, 0.0, -s]
        self.assertAlmostEqual(G.bearing_angle(pitched, [1.0, -1.0, 0.0]), 45.0)
        down = pose.copy()
        down[:3, 2] = [0.0, 0.0, -1.0]
        self.assertIsNone(G.bearing_angle(down, [1.0, 0.0, 0.0]))

    def test_bearing_sign_matches_the_success_metrics_camera(self):
        """Right of the facing direction = right half of the frame in canonical_task_metrics_v2's projection."""
        from vagen.envs.active_spatial.canonical_task_metrics_v2 import _Context

        ctx = _Context(ITEM, START)
        for dx, dy in ((1.0, 0.3), (1.0, -0.3), (0.5, 0.4), (0.8, -0.6), (1.2, 0.0)):
            fwd, right = START[:3, 2], START[:3, 0]
            p = START[:3, 3] + dx * fwd + dy * right
            angle = G.bearing_angle(START, p)
            u = float(ctx.uv(ctx.to_camera(np.array([p])))[0][0])
            self.assertEqual(angle > 1e-9, u > ctx.K[0, 2] + 1e-6, (dx, dy, angle, u))

    def test_truth_on_the_eval_task(self):
        gt = G.truth(ITEM, START)
        self.assertEqual(gt["bearing"], "front")
        self.assertLess(abs(gt["bearing_deg"]), 2.0)
        self.assertAlmostEqual(gt["distance_m"], math.hypot(-1.1043389702580089 + 1.8489284025897093,
                                                            -2.76611938717639 + 3.119246516083379), places=3)
        self.assertEqual((gt["required_m"], gt["tolerance_m"], gt["condition"]), (1.5, 0.25, "too_close"))
        # the same distance the success metric measures
        from vagen.envs.active_spatial.canonical_task_metrics_v2 import judge_absolute_positioning

        self.assertAlmostEqual(judge_absolute_positioning(ITEM, START)["distance_m"], gt["distance_m"], places=3)

    def test_truth_other_family_has_no_condition(self):
        item = {"task_type": "fov_inclusion", "target_object": {"objects": [box(3.0, 0.0), box(0.0, -3.0, oid="2")]}}
        gt = G.truth(item, facing_x())
        self.assertEqual((gt["bearing"], gt["distance_m"]), ("front", 2.9))
        self.assertNotIn("condition", gt)
        inside = G.truth({"task_type": "fov_inclusion", "target_object": {"objects": [box(0.05, 0.0)]}}, facing_x())
        self.assertEqual(inside["distance_m"], 0.0)  # inside the footprint


class ScoreTest(unittest.TestCase):
    def test_score(self):
        gt = G.truth(ITEM, START)  # front, 0.824 m, too_close
        r = G.score("<state>bearing=front; distance=0.9</state><action>move_backward|</action>", gt, 0)
        self.assertEqual((r["bearing_ok"], r["distance_ok"], r["both_ok"], r["stated_condition"], r["condition_ok"]),
                         (True, True, True, "too_close", True))
        r = G.score("<state>bearing=left; distance=1.5</state><action>move_backward|</action>", gt, 3)
        self.assertEqual((r["turn"], r["bearing_ok"], r["distance_ok"], r["stated_condition"], r["condition_ok"]),
                         (3, False, False, "ok", False))
        r = G.score("<action>move_backward|</action>", gt, 1)  # no statement: every field wrong
        self.assertEqual((r["present"], r["bearing_ok"], r["distance_ok"], r["both_ok"], r["condition_ok"]),
                         (False, False, False, False, False))
        # the distance tolerance: max(0.3 m, 25%)
        far = {"bearing_accepted": ["front"], "distance_m": 4.0}
        self.assertTrue(G.score("<state>bearing=front; distance=5.0</state>", far, 0)["distance_ok"])
        self.assertFalse(G.score("<state>bearing=front; distance=5.01</state>", far, 0)["distance_ok"])
        near = {"bearing_accepted": ["front"], "distance_m": 0.4}
        self.assertTrue(G.score("<state>bearing=front; distance=0.15</state>", near, 0)["distance_ok"])
        self.assertFalse(G.score("<state>bearing=front; distance=0.75</state>", near, 0)["distance_ok"])
        self.assertNotIn("condition_ok", G.score("<state>bearing=front; distance=0.15</state>", near, 0))

    def test_summary(self):
        def turn(t, b, d):
            return {"turn": t, "present": True, "bearing_ok": b, "distance_ok": d, "both_ok": b and d}

        rows = [{"index": 0, "task_type": "a", "success": True, "grounding_turns": [turn(0, True, True), turn(1, True, False)]},
                {"index": 1, "task_type": "a", "success": True, "grounding_turns": [turn(0, True, True)]},
                {"index": 2, "task_type": "b", "success": False, "grounding_turns": [turn(0, False, False), turn(1, True, False)]},
                {"index": 3, "task_type": "b", "success": False, "grounding_turns": [turn(0, False, True)]},
                {"index": 4, "error": "RuntimeError: x"}]
        s = G.summarize(rows)
        self.assertEqual((s["episodes"], s["turns"], s["success"]), (4, 6, 0.5))
        self.assertEqual(s["accuracy"]["bearing_ok"], round(4 / 6, 4))
        self.assertIsNone(s["accuracy"]["condition_ok"])
        self.assertEqual(s["by_turn"]["1"], {"n": 2, "present": 1.0, "bearing_ok": 1.0, "distance_ok": 0.0, "both_ok": 0.0})
        self.assertEqual(s["by_outcome"]["success"]["bearing_ok"], 1.0)
        self.assertEqual(s["by_outcome"]["failure"]["turns"], 3)
        self.assertEqual(s["first_turn_vs_success"]["bearing_ok"],
                         {"right_n": 2, "right_success": 1.0, "wrong_n": 2, "wrong_success": 0.0, "phi": 1.0})
        self.assertGreater(s["episode_share_vs_success"]["bearing_ok"], 0.8)
        self.assertEqual(s["by_family"]["b"]["n"], 3)
        self.assertEqual(s["baselines"], {})  # no truth recorded in these rows

    def test_baselines(self):
        tr = [{"bearing_accepted": ["front"], "distance_m": 1.0, "condition": "too_close"},
              {"bearing_accepted": ["front", "right"], "distance_m": 2.0, "condition": "too_close"},
              {"bearing_accepted": ["left"], "distance_m": 4.0, "condition": "ok"},
              {"bearing_accepted": ["back"], "distance_m": 8.0}]
        b = G.baselines(tr)
        self.assertEqual((b["bearing_front"], b["distance_median_m"]), (0.5, 3.0))
        self.assertEqual(b["distance_median"], 0.25)  # 3.0 is within 25% of 4.0 only
        self.assertEqual((b["condition_majority"], b["condition_majority_share"]), ("too_close", 0.6667))

    def test_summary_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "m_grd"
            res.mkdir()
            row = {"index": 0, "task_type": "a", "success": True,
                   "grounding_turns": [{"turn": 0, "present": True, "bearing_ok": True, "distance_ok": False,
                                        "both_ok": False}]}
            (res / "results.jsonl").write_text(json.dumps(row) + "\n{\"partial")
            G.main(["summary", "--result", str(res), "--out", str(Path(tmp) / "s.json")])
            s = json.loads((Path(tmp) / "s.json").read_text())
        self.assertEqual((s["m_grd"]["episodes"], s["m_grd"]["accuracy"]["bearing_ok"]), (1, 1.0))


class RunEpisodeTest(unittest.TestCase):
    """The switch in evaluation.run_episode; off, the calls and the row are exactly as before."""

    def make_env(self):
        class Env:
            current_item = ITEM
            _current_step = 0

            def __init__(self):
                self.view_engine = SimpleNamespace(get_pose=lambda: self.pose.copy())
                self.pose = START.copy()
                self.read = []

            def reset(self, seed):
                return {"obs_str": "<image>\nturn 0", "multi_modal_data": {"<image>": ["frame0"]}}, {}

            def system_prompt(self):
                return "sys"

            def step(self, text):
                self.read.append(text)
                self._current_step += 1
                self.pose[:3, 3] -= 0.25 * self.pose[:3, 2]  # move_backward: the next frame's pose
                obs = {"obs_str": f"<image>\nturn {self._current_step}",
                       "multi_modal_data": {"<image>": [f"frame{self._current_step}"]}}
                return obs, -0.01, self._current_step >= 2, {"actions": ["move_backward"], "metrics": {}}

            def _calculate_canonical_metric(self):
                return {"success": self._current_step >= 2}

        return Env()

    def make_policy(self, answers):
        calls = []

        class Policy:
            def generate(self, system, user_text, image, earlier=()):
                calls.append((user_text, list(earlier)))
                return answers[len(calls) - 1]

        return Policy(), calls

    def test_off_is_unchanged(self):
        env = self.make_env()
        policy, calls = self.make_policy(["<action>move_backward|</action>"] * 2)
        row = E.run_episode(env, policy, 0, 12, history=True)
        self.assertEqual([c[0] for c in calls], ["<image>\nturn 0", "<image>\nturn 1"])
        self.assertEqual(env.read, ["<action>move_backward|</action>"] * 2)
        self.assertEqual(calls[1][1], [("<image>\nturn 0", "frame0", "<action>move_backward|</action>")])
        self.assertEqual(set(row), {"index", "task_id", "scene_id", "task_type", "d_star", "success",
                                    "actions_to_success", "turns", "actions", "collisions", "invalid_turns"})

    def test_on(self):
        env = self.make_env()
        answers = ["<state>bearing=front; distance=0.8</state>\n<action>move_backward|</action>",
                   "<action>move_backward|</action>"]
        policy, calls = self.make_policy(answers)
        row = E.run_episode(env, policy, 0, 12, history=True, grounding=True)
        probe = G.instruction(ITEM)
        self.assertEqual([c[0] for c in calls], ["<image>\nturn 0" + probe, "<image>\nturn 1" + probe])
        self.assertEqual(calls[1][1], [("<image>\nturn 0" + probe, "frame0", answers[0])])  # history: what it saw
        self.assertEqual(env.read, ["\n<action>move_backward|</action>", "<action>move_backward|</action>"])
        t0, t1 = row["grounding_turns"]
        self.assertEqual((t0["turn"], t0["bearing_ok"], t0["distance_ok"], t0["condition_ok"]), (0, True, True, True))
        self.assertEqual(t0["truth"], G.truth(ITEM, START))  # the pose of the first frame
        moved = START.copy()
        moved[:3, 3] -= 0.25 * START[:3, 2]
        self.assertEqual(t1["truth"], G.truth(ITEM, moved))  # the pose after the first action
        self.assertEqual((t1["present"], t1["both_ok"]), (False, False))
        self.assertEqual(row["grounding"]["both_ok"], 0.5)
        self.assertTrue(row["success"])

    def test_resume_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp)
            E.check_resume_probe(res, True)  # no results yet: either mode
            (res / "results.jsonl").write_text(json.dumps({"index": 0}) + "\n")
            E.check_resume_probe(res, False)  # no model.json: not a probe run
            with self.assertRaises(SystemExit):
                E.check_resume_probe(res, True)
            (res / "model.json").write_text(json.dumps({"grounding_probe": G.VERSION}))
            E.check_resume_probe(res, True)
            with self.assertRaises(SystemExit):
                E.check_resume_probe(res, False)

    def test_option(self):
        from unittest import mock

        seen = []
        base = ["run", "--eval", "e", "--name", "n", "--model", "m"]
        with mock.patch.object(E, "cmd_run", lambda a: seen.append(a.grounding_probe)):
            E.main(base)
            E.main(base + ["--grounding-probe"])
        self.assertEqual(seen, [False, True])


if __name__ == "__main__":
    unittest.main()
