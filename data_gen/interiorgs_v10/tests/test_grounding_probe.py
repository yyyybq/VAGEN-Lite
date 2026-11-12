"""grounding_probe.py and evaluation's --grounding-probe (the user's supplement of 2026-11-12, item 5)."""

import dataclasses
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
MODEL_KEYS = {"name", "model", "version", "prompt_format", "greedy", "samples", "temperature", "top_p", "sample_seed",
              "env_switches", "max_turns", "history", "sft_text", "entropy", "load_seconds"}  # model.json before item 5


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
        for word in ("in front", "forward", "straight ahead", "ahead", "In_Front"):
            self.assertEqual(p(f"<state>bearing={word}; distance=0.75 meters</state>")["bearing"], "front", word)
        self.assertEqual(p("< state >bearing=left;distance=150 cm< /state >"),
                         {"present": True, "bearing": "left", "distance": 1.5})  # whitespace in the tags, cm
        self.assertEqual(p("<state>bearing=right; distance=1,25m</state>")["distance"], 1.25)  # comma decimal
        self.assertEqual(p("<state>bearing=right distance=2metres</state>"),
                         {"present": True, "bearing": "right", "distance": 2.0})
        self.assertEqual(p("<state>bearing=front-left; distance=.5</state>"),
                         {"present": True, "bearing": None, "distance": 0.5})  # not one of the four: unreadable
        self.assertEqual(p("<state>bearing=<B>; distance=<D></state>"),  # the placeholders copied
                         {"present": True, "bearing": None, "distance": None})
        self.assertEqual(p("<state>bearing=left</state>"), {"present": True, "bearing": "left", "distance": None})
        self.assertEqual(p("<action>move_forward|</action>"), {"present": False, "bearing": None, "distance": None})
        self.assertEqual(p("<state>bearing=left; distance=1"), {"present": False, "bearing": None, "distance": None})
        self.assertEqual(p(None)["present"], False)
        self.assertEqual(p("<state>bearing=left; distance=1</state><state>bearing=right; distance=9</state>")
                         ["bearing"], "left")  # the first block counts

    def test_answer_part_keeps_the_action_reading(self):
        from vagen.envs.active_spatial.utils import parse_free_think

        a = "<state>bearing=front; distance=1.5</state>\n<action>move_backward|move_backward|</action>"
        self.assertEqual(G.answer_part(a), "\n<action>move_backward|move_backward|</action>")
        self.assertEqual(parse_free_think(G.answer_part(a)),
                         parse_free_think("<action>move_backward|move_backward|</action>"))
        self.assertEqual(G.answer_part("< state>x</state >\n<action>turn_left|</action>"), "\n<action>turn_left|</action>")
        plain = "<action>turn_left|</action>"
        self.assertEqual(G.answer_part(plain), plain)  # no statement: unchanged
        bad = "<state>bearing=<action>turn_left|</action></state><action>move_forward|</action>"
        self.assertEqual(parse_free_think(G.answer_part(bad))["actions_blob"], "move_forward|")  # only the action part

    def test_instruction(self):
        text = G.instruction(ITEM)
        self.assertEqual(text, (
            "\n\nBefore the action, first write one line in this form:\n"
            "<state>bearing=<B>; distance=<D></state>\n"
            "<B> is where the standing air conditioner is now: front (within 45 degrees of straight ahead, inside "
            "your view), left or right (45 to 135 degrees to that side), or back (otherwise).\n"
            "<D> is its horizontal distance from you in meters, to its nearest side, as a number.\n"
            "Then write the action as above."))
        import re

        self.assertEqual(re.findall(r"\d+(?:\.\d+)?", text), ["45", "45", "135"])  # no numeric example
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
        self.assertEqual(G.accepted_bearings(None), list(G.BEARINGS))  # no angle: any bearing

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
            p = START[:3, 3] + dx * START[:3, 2] + dy * START[:3, 0]
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
        from vagen.envs.active_spatial.canonical_task_metrics_v2 import judge_absolute_positioning

        self.assertAlmostEqual(judge_absolute_positioning(ITEM, START)["distance_m"], gt["distance_m"], places=3)

    def test_truth_other_family_and_inside_the_footprint(self):
        item = {"task_type": "fov_inclusion", "target_object": {"objects": [box(3.0, 0.0), box(0.0, -3.0, oid="2")]}}
        gt = G.truth(item, facing_x())
        self.assertEqual((gt["bearing"], gt["bearing_accepted"], gt["distance_m"]), ("front", ["front"], 2.9))
        self.assertNotIn("condition", gt)
        # the camera inside the footprint (off the centre, so the angle exists): any bearing is right
        inside = G.truth({"task_type": "fov_inclusion", "target_object": {"objects": [box(0.05, 0.05)]}}, facing_x())
        self.assertEqual((inside["distance_m"], inside["bearing_accepted"]), (0.0, list(G.BEARINGS)))
        self.assertTrue(G.score("<state>bearing=back; distance=0</state>", inside, 0)["both_ok"])
        self.assertFalse(G.score("<state>distance=0</state>", inside, 0)["bearing_ok"])  # unreadable is still wrong


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
        far = {"bearing_accepted": ["front"], "distance_m": 4.0}  # the tolerance: max(0.3 m, 25%)
        self.assertTrue(G.score("<state>bearing=front; distance=5.0</state>", far, 0)["distance_ok"])
        self.assertFalse(G.score("<state>bearing=front; distance=5.01</state>", far, 0)["distance_ok"])
        near = {"bearing_accepted": ["front"], "distance_m": 0.4}
        self.assertTrue(G.score("<state>bearing=front; distance=0.15</state>", near, 0)["distance_ok"])
        self.assertFalse(G.score("<state>bearing=front; distance=0.75</state>", near, 0)["distance_ok"])
        self.assertNotIn("condition_ok", G.score("<state>bearing=front; distance=0.15</state>", near, 0))

    def test_baseline_marks(self):
        gt = {"bearing_accepted": ["front", "right"], "distance_m": 1.4, "required_m": 1.5, "tolerance_m": 0.25,
              "condition": "ok"}
        m = G.baseline_marks(gt, 2.0)
        self.assertEqual(m["bearing_ok"], {"front": True, "right": True, "back": False, "left": False})
        self.assertEqual({n for n, v in m["distance_ok"].items() if v}, {"1.5 m", "echo required"})  # tol 0.35 m
        self.assertTrue(m["condition_ok"]["echo required"] and m["condition_ok"]["band ok"])
        self.assertFalse(m["condition_ok"]["median"])  # 2.0 m is too far for 1.5 +- 0.25
        self.assertTrue(m["both_ok"]["right / 1.5 m"])
        other = G.baseline_marks({"bearing_accepted": ["left"], "distance_m": 1.4}, 2.0)
        self.assertFalse(other["distance_ok"]["echo required"])  # no required distance: no answer
        self.assertNotIn("condition_ok", other)


def _row(i, success, turns, family="absolute_positioning"):
    return {"index": i, "task_type": family, "success": success,
            "grounding_turns": [G.score(a, gt, k) for k, (a, gt) in enumerate(turns)]}


GT_A = {"bearing": "front", "bearing_accepted": ["front"], "distance_m": 1.0, "required_m": 1.5, "tolerance_m": 0.25,
        "condition": "too_close"}
GT_B = {"bearing": "left", "bearing_accepted": ["left"], "distance_m": 3.0, "required_m": 1.5, "tolerance_m": 0.25,
        "condition": "too_far"}


class SummaryTest(unittest.TestCase):
    def rows(self):
        st = "<state>bearing={}; distance={}</state><action>move_forward|</action>".format
        return [_row(0, True, [(st("front", 1.0), GT_A)]),
                _row(1, True, [(st("front", 1.1), GT_A)]),
                _row(2, False, [(st("front", 1.0), GT_B), (st("left", 3.0), GT_B)]),
                _row(3, False, [("<action>move_forward|</action>", GT_B)]),
                {"index": 4, "error": "RuntimeError: x"}]

    def test_readings_next_to_the_constant_answers(self):
        s = G.summarize(self.rows())
        self.assertEqual((s["episodes"], s["turns"], s["success"], s["present"]), (4, 5, 0.5, 0.8))
        self.assertEqual(s["baseline_answers"]["median_m"], 3.0)
        acc = s["accuracy"]["bearing_ok"]  # 3/5; "left" also gets 3/5 on these turns
        self.assertEqual((acc["value"], acc["best_baseline"], acc["best_baseline_name"], acc["excess"]),
                         (0.6, 0.6, "left", 0.0))
        rd = s["accuracy_readable"]["bearing_ok"]  # 3 of the 4 readable turns; a constant gets 2 of them
        self.assertEqual((rd["n"], rd["value"], rd["best_baseline"], rd["excess"]), (4, 0.75, 0.5, 0.25))
        em = s["accuracy_episode_mean"]["bearing_ok"]  # shares 1, 1, 0.5, 0
        self.assertEqual((em["value"], em["best_baseline"], em["excess"]), (0.625, 0.5, 0.125))
        cond = s["accuracy"]["condition_ok"]
        self.assertEqual((cond["value"], cond["best_baseline"], cond["excess"]), (0.6, 0.6, 0.0))
        # the confound: a constant "front" is right exactly on the successful episodes' first turns
        ft = s["first_turn_vs_success"]["bearing_ok"]
        self.assertEqual((ft["phi"], ft["best_baseline_phi"], ft["best_baseline_phi_name"], ft["excess_phi"]),
                         (1.0, 1.0, "front", 0.0))
        corr = s["episode_share_vs_success"]["bearing_ok"]
        self.assertEqual((corr["best_baseline"], corr["best_baseline_name"]), (1.0, "front"))
        self.assertLess(corr["excess"], 0.0)
        self.assertIn("condition_ok", s["episode_share_vs_success"])
        self.assertEqual(s["by_turn"]["1"]["condition_ok"]["value"], 1.0)  # condition_ok per turn index
        self.assertEqual(s["by_turn"]["1"]["n"], 1)
        fam = s["by_family"]["absolute_positioning"]
        self.assertEqual((fam["episodes"], fam["n"], fam["first_turn_vs_success"]["condition_ok"]["phi"]), (4, 5, 1.0))
        self.assertEqual(fam["bearing_ok"]["best_baseline"], 0.6)  # per-family baselines
        self.assertEqual((s["by_outcome"]["failure"]["n"], s["by_outcome"]["success"]["bearing_ok"]["value"]), (3, 1.0))

    def test_family_without_spread(self):
        st = "<state>bearing=front; distance=1.0</state>"
        gt = dict(GT_A)
        gt.pop("condition"), gt.pop("required_m"), gt.pop("tolerance_m")
        s = G.summarize([_row(0, True, [(st, gt)], "fov_inclusion"), _row(1, True, [(st, gt)], "fov_inclusion")])
        ft = s["by_family"]["fov_inclusion"]["first_turn_vs_success"]
        self.assertIsNone(ft["bearing_ok"]["phi"])  # every episode succeeded
        self.assertEqual(ft["condition_ok"], {"episodes": 0})
        self.assertEqual(s["accuracy"]["condition_ok"], {"n": 0, "value": None})
        self.assertEqual(G.summarize([])["turns"], 0)

    def test_summary_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = Path(tmp) / "m_grd"
            res.mkdir()
            rows = self.rows()[:2]
            (res / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows) + "{\"partial")
            G.main(["summary", "--result", str(res), "--out", str(Path(tmp) / "s.json")])
            s = json.loads((Path(tmp) / "s.json").read_text())
            key = f"{Path(tmp).name}/m_grd"  # SET/NAME
            self.assertEqual((s[key]["episodes"], s[key]["accuracy"]["bearing_ok"]["value"]), (2, 1.0))
            (res / "results.jsonl").write_text("{\"partial\n" + json.dumps(rows[0]) + "\n")
            with self.assertRaises(json.JSONDecodeError):  # only the last line may be partial
                G.main(["summary", "--result", str(res)])


class _Frame:  # a frame that can be saved as a JPEG
    def convert(self, mode):
        from PIL import Image

        return Image.new(mode, (4, 4))


def make_env():
    class Env:
        current_item = ITEM
        _current_step = 0

        def __init__(self, cfg=None):
            self.view_engine = SimpleNamespace(get_pose=lambda: self.pose.copy())
            self.pose = START.copy()
            self.read = []

        def reset(self, seed):
            self._current_step, self.pose = 0, START.copy()
            return {"obs_str": "<image>\nturn 0", "multi_modal_data": {"<image>": [_Frame()]}}, {}

        def system_prompt(self):
            return "sys"

        def step(self, text):
            self.read.append(text)
            self._current_step += 1
            self.pose[:3, 3] -= 0.25 * self.pose[:3, 2]  # move_backward: the next frame's pose
            obs = {"obs_str": f"<image>\nturn {self._current_step}", "multi_modal_data": {"<image>": [_Frame()]}}
            return obs, -0.01, self._current_step >= 2, {"actions": ["move_backward"], "metrics": {}}

        def _calculate_canonical_metric(self):
            return {"success": self._current_step >= 2}

    return Env


def make_policy(answers, calls, side_calls=None):
    class Policy:
        last_prompt_tokens = None

        def __init__(self, *a, **k):
            pass

        def generate(self, system, user_text, image, earlier=()):
            calls.append((user_text, [(u, a) for u, _, a in earlier]))
            return answers[(len(calls) - 1) % len(answers)]

        def side_generate(self, system, user_text, image, earlier=(), as_sft=False, prefix="", stop="", max_tokens=0):
            if side_calls is not None:
                side_calls.append((user_text, [(u, a) for u, _, a in earlier], prefix, stop, max_tokens))
            return {"text": "front; distance=0.8</state>", "finish": "stop", "prompt_tokens": 99}

    return Policy


STATED = ["<state>bearing=front; distance=0.8</state>\n<action>move_backward|</action>",
          "<action>move_backward|</action>"]


class RunEpisodeTest(unittest.TestCase):
    """The switch in evaluation.run_episode; off, the calls and the row are exactly as before, even when the answers
    carry <state> lines."""

    def test_off_is_unchanged(self):
        env, calls = make_env()(), []
        with tempfile.TemporaryDirectory() as tmp:
            row = E.run_episode(env, make_policy(STATED, calls)(), 0, 12, history=True, frames=Path(tmp) / "frames")
        self.assertEqual([c[0] for c in calls], ["<image>\nturn 0", "<image>\nturn 1"])
        self.assertEqual(env.read, STATED)  # the <state> line reaches the env unchanged
        self.assertEqual(calls[1][1], [("<image>\nturn 0", STATED[0])])
        self.assertEqual([t["user"] for t in row["trajectory"]], ["<image>\nturn 0", "<image>\nturn 1"])
        self.assertEqual([t["answer"] for t in row["trajectory"]], STATED)
        self.assertFalse({"grounding", "grounding_turns"} & set(row))
        self.assertFalse(any("grounding" in t for t in row["trajectory"]))

    def test_on(self):
        env, calls = make_env()(), []
        row = E.run_episode(env, make_policy(STATED, calls)(), 0, 12, history=True, grounding=True)
        probe = G.instruction(ITEM)
        self.assertEqual([c[0] for c in calls], ["<image>\nturn 0" + probe, "<image>\nturn 1" + probe])
        self.assertEqual(calls[1][1], [("<image>\nturn 0" + probe, STATED[0])])  # history: what it saw
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
            E.check_resume_probe(res, G.VERSION)  # no results yet: any
            (res / "results.jsonl").write_text(json.dumps({"index": 0}) + "\n")
            E.check_resume_probe(res, None)  # no model.json: not a probe run
            with self.assertRaises(SystemExit):
                E.check_resume_probe(res, G.VERSION)
            (res / "model.json").write_text(json.dumps({"grounding_probe": G.VERSION}))
            E.check_resume_probe(res, G.VERSION)
            for other in (None, "grounding_probe_v0"):
                with self.assertRaises(SystemExit):
                    E.check_resume_probe(res, other)

    def test_option(self):
        seen = []
        base = ["run", "--eval", "e", "--name", "n", "--model", "m"]
        with mock.patch.object(E, "cmd_run", lambda a: seen.append(a.grounding_probe)):
            E.main(base)
            E.main(base + ["--grounding-probe"])
        self.assertEqual(seen, [False, True])


class CmdRunTest(unittest.TestCase):
    """`evaluation run` end to end with a scripted env and policy: model.json, results and what the env reads."""

    def run_cmd(self, tmp, name, opts, model="/m"):
        @dataclasses.dataclass
        class Cfg:
            jsonl_path: str = ""
            prompt_format: str = ""

        Env, calls = make_env(), []
        envs = []

        def build(cfg):
            envs.append(Env(cfg))
            return envs[-1]

        env_mod = types.ModuleType("vagen.envs.active_spatial.env")
        env_mod.ActiveSpatialEnv = build
        cfg_mod = types.ModuleType("vagen.envs.active_spatial.env_config")
        cfg_mod.ActiveSpatialEnvConfig = Cfg
        ev = Path(tmp)
        (ev / "tasks.jsonl").write_text(json.dumps(ITEM) + "\n")
        with mock.patch.dict(sys.modules, {env_mod.__name__: env_mod, cfg_mod.__name__: cfg_mod}), \
                mock.patch.object(E, "VllmPolicy", make_policy(STATED, calls)), \
                mock.patch.dict(os.environ, {"R1_RENDER_URL": "http://scripted.invalid"}):
            E.main(["run", "--eval", str(ev), "--name", name, "--model", model, *opts])
        out = ev / name
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        return json.loads((out / "model.json").read_text()), rows, envs[-1].read, calls

    def test_off_and_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, rows, read, calls = self.run_cmd(tmp, "plain", ["--format-example", "action_names"])
            self.assertEqual(set(model), MODEL_KEYS)
            self.assertEqual(read, STATED)
            self.assertEqual([t["user"] for t in rows[0]["trajectory"]], ["<image>\nturn 0", "<image>\nturn 1"])
            self.assertFalse({"grounding", "grounding_turns"} & set(rows[0]))
            model, rows, read, calls = self.run_cmd(tmp, "probe", ["--format-example", "action_names",
                                                                   "--grounding-probe"])
            self.assertEqual(set(model), MODEL_KEYS | {"grounding_probe"})
            self.assertEqual(model["grounding_probe"], G.VERSION)
            self.assertEqual(read, [G.answer_part(a) for a in STATED])
            self.assertEqual(len(rows[0]["grounding_turns"]), rows[0]["turns"])
            with self.assertRaises(SystemExit):  # the plain folder cannot be resumed as a probe run
                self.run_cmd(tmp, "plain", ["--grounding-probe"])

    def test_side_mode(self):
        def strip(row):
            return {k: v for k, v in row.items() if k not in ("grounding", "grounding_turns")}

        with tempfile.TemporaryDirectory() as tmp:
            m0, plain, read0, calls0 = self.run_cmd(tmp, "plain", ["--format-example", "action_names"])
            m1, side, read1, calls1 = self.run_cmd(tmp, "side", ["--format-example", "action_names",
                                                                 "--grounding-side"])
            self.assertEqual(m1.pop("grounding_probe"), G.VERSION_SIDE)
            for m in (m0, m1):
                m.pop("load_seconds"), m.pop("name")
            self.assertEqual(m0, m1)  # model.json: only the probe key differs
            self.assertEqual((read1, calls1), (read0, calls0))  # the action stream and what the env reads
            self.assertEqual([strip(r) for r in side], plain)  # results: only the probe fields differ
            self.assertEqual(side[0]["grounding"]["version"], G.VERSION_SIDE)
            self.assertEqual(side[0]["grounding_turns"][0]["side_output"], "front; distance=0.8</state>")
            self.assertEqual(
                sorted(p.name for p in (Path(tmp) / "side" / "frames").iterdir()),
                sorted(p.name for p in (Path(tmp) / "plain" / "frames").iterdir()))
            with self.assertRaises(SystemExit):  # one probe per run
                self.run_cmd(tmp, "both", ["--grounding-probe", "--grounding-side"])
            with self.assertRaises(SystemExit):  # the plain folder cannot be resumed with the side call
                self.run_cmd(tmp, "plain", ["--grounding-side"])
            cam = Path(tmp) / "cambrian"
            cam.mkdir()
            (cam / "config.json").write_text(json.dumps({"model_type": "cambrian_qwen"}))
            with self.assertRaises(SystemExit) as ctx:  # refused before anything starts
                self.run_cmd(tmp, "cam", ["--grounding-side"], model=str(cam))
            self.assertIn("not supported for cambrian_qwen", str(ctx.exception))
            self.assertFalse((Path(tmp) / "cam" / "model.json").exists())


class SideEpisodeTest(unittest.TestCase):
    """v2 in run_episode: the action calls are exactly the plain run's (a policy that samples from its own RNG gives the
    same answers with the side calls on), the side calls see the action call's history and never enter it."""

    def policy(self):
        import random

        class RngPolicy:
            last_prompt_tokens = None

            def __init__(self):
                self.rng, self.calls, self.side_calls = random.Random(7), [], []

            def generate(self, system, user_text, image, earlier=(), seed=None):
                self.calls.append((system, user_text, [(u, a) for u, _, a in earlier], seed))
                return f"<action>{self.rng.choice(['move_backward', 'turn_left', 'move_forward'])}|</action>"

            def side_generate(self, system, user_text, image, earlier=(), as_sft=False, prefix="", stop="",
                              max_tokens=0):
                self.side_calls.append(dict(system=system, user=user_text, earlier=[(u, a) for u, _, a in earlier],
                                            prefix=prefix, stop=stop, max_tokens=max_tokens, image=image))
                return {"text": ["right; distance=1,5 m", "front; distance=0.8</state>"][(len(self.side_calls) - 1) % 2],
                        "finish": "length", "prompt_tokens": 7, "layout_ok": True, "shared_tokens": 5}

        return RngPolicy()

    def env(self):
        Env = make_env()

        class Longer(Env):  # four turns
            def step(self, text):
                obs, reward, done, info = super().step(text)
                return obs, reward, self._current_step >= 4, info

            def _calculate_canonical_metric(self):
                return {"success": self._current_step >= 4}

        return Longer()

    def test_action_stream_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            env0, p0 = self.env(), self.policy()
            f0, f1 = Path(tmp) / "off" / "frames", Path(tmp) / "on" / "frames"  # the rows name the frames folder
            row0 = E.run_episode(env0, p0, 0, 12, frames=f0, history=True, sample=1, seed=5)
            env1, p1 = self.env(), self.policy()
            row1 = E.run_episode(env1, p1, 0, 12, frames=f1, history=True, sample=1, seed=5, grounding_side=True)
            frames0 = {p.name: p.read_bytes() for p in f0.iterdir()}
            frames1 = {p.name: p.read_bytes() for p in f1.iterdir()}
        self.assertEqual(p1.calls, p0.calls)  # same system, user text, history, seed: the plain calls
        self.assertEqual(env1.read, env0.read)
        self.assertEqual({k: v for k, v in row1.items() if k not in ("grounding", "grounding_turns")}, row0)
        self.assertEqual(frames1, frames0)
        self.assertEqual(len(p1.side_calls), row1["turns"])
        probe = G.side_question(ITEM)
        for k, side in enumerate(p1.side_calls):
            system, user, earlier, _ = p1.calls[k]
            self.assertEqual((side["system"], side["user"], side["earlier"]), (system, user + probe, earlier))
            self.assertEqual((side["prefix"], side["stop"], side["max_tokens"]),
                             (G.SIDE_PREFIX, G.SIDE_STOP, G.SIDE_MAX_TOKENS))
        for _, _, earlier, _ in p1.calls:  # the side questions and answers never enter the history
            self.assertFalse(any("Question" in u or "<state>" in a for u, a in earlier))
        t = row1["grounding_turns"]
        self.assertEqual(t[0]["truth"], G.truth(ITEM, START))
        self.assertEqual((t[0]["side_output"], t[0]["stated_bearing"], t[0]["stated_distance"], t[0]["side_finish"]),
                         ("right; distance=1,5 m", "right", 1.5, "length"))
        self.assertEqual((t[1]["bearing_ok"], t[1]["side_layout_ok"], t[1]["side_prompt_tokens"]), (True, True, 7))
        self.assertEqual(row1["grounding"]["version"], G.VERSION_SIDE)
        with self.assertRaises(ValueError):
            E.run_episode(self.env(), self.policy(), 0, 12, grounding=True, grounding_side=True)

    def test_side_text(self):
        self.assertEqual(G.side_text("front; distance=1.5</state>"), "<state>bearing=front; distance=1.5</state>")
        self.assertEqual(G.side_text("left; distance=2"), "<state>bearing=left; distance=2</state>")  # closed
        self.assertEqual(G.parse_statement(G.side_text("in front; distance=150cm")),
                         {"present": True, "bearing": "front", "distance": 1.5})
        self.assertEqual(G.parse_statement(G.side_text(""))["bearing"], None)
        self.assertEqual(G.layout_check([1, 9, 2, 3], [1, 9, 8, 9, 2, 3, 5, 6], [5, 6], 9),
                         {"layout_ok": True, "shared_tokens": 2})
        self.assertFalse(G.layout_check([1, 9, 2, 3], [1, 9, 2, 4, 5, 6], [5, 6], 9)["layout_ok"])
        self.assertFalse(G.layout_check([1, 2, 3], [1, 2, 3, 5, 6], [5, 6], 9)["layout_ok"])  # no end token
        text = G.side_question(ITEM)
        self.assertTrue(text.startswith("\n\nQuestion (answer it, no action this time): where is the standing air "
                                        "conditioner now?\n"))
        self.assertIn("<state>bearing=<B>; distance=<D></state>", text)


TOKENIZER = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232"


@unittest.skipUnless(Path(TOKENIZER, "tokenizer.json").is_file(), "the Qwen3-VL C0 SFT checkpoint's tokenizer")
class SidePolicyTest(unittest.TestCase):
    """VllmPolicy.side_generate with the model's own tokenizer and chat template, the engine faked: LLM.preprocess_chat
    is vLLM 0.11's (the chat template with the generation prompt, then the ids), generate records its inputs."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def fake_llm(self, texts):
        tok, seen = self.tok, []

        def parsed(messages):  # vLLM's "openai" content format for this template: image parts become {"type": "image"}
            out = []
            for m in messages:
                content = m["content"]
                if isinstance(content, list):
                    content = [{"type": "image"} if p.get("type") == "image_url" else p for p in content]
                out.append({"role": m["role"], "content": content})
            return out

        class LLM:
            def get_tokenizer(self):
                return tok

            def preprocess_chat(self, messages, add_generation_prompt=True):
                text = tok.apply_chat_template(parsed(messages), add_generation_prompt=add_generation_prompt,
                                               tokenize=False)
                return [{"prompt_token_ids": tok.encode(text, add_special_tokens=False),
                         "multi_modal_data": {"image": ["frame"]}}]

            def generate(self, prompts, sampling_params=None, use_tqdm=False):
                seen.append((prompts[0], sampling_params))
                return [SimpleNamespace(prompt_token_ids=prompts[0]["prompt_token_ids"],
                                        outputs=[SimpleNamespace(text=texts[len(seen) - 1], finish_reason="stop")])]

            def chat(self, messages, sampling_params=None, use_tqdm=False):
                return self.generate(self.preprocess_chat(messages), sampling_params=sampling_params)

        return LLM(), seen

    def test_prefix_layout(self):
        from PIL import Image

        vllm = types.ModuleType("vllm")
        vllm.SamplingParams = lambda **k: SimpleNamespace(**k)
        llm, seen = self.fake_llm(["<action>move_backward|</action>", "front; distance=1.5</state>"])
        policy = E.VllmPolicy.__new__(E.VllmPolicy)
        policy.llm, policy.cambrian, policy.temperature, policy.top_p, policy.entropy = llm, None, 0.0, 1.0, False
        policy.last_prompt_tokens = policy.last_entropy = policy.last_tokens = policy.last_prompt_ids = None
        frame = Image.new("RGB", (8, 8))
        earlier = [("<image>\nturn 0", frame, "<action>turn_left|</action>")]
        user = "<image>\nturn 1\nRespond with only the action."
        with mock.patch.dict(sys.modules, {"vllm": vllm}):
            action = policy.generate("sys", user, frame, earlier=earlier)
            tokens_after_action = policy.last_prompt_tokens
            side = policy.side_generate("sys", user + G.side_question(ITEM), frame, earlier=earlier,
                                        prefix=G.SIDE_PREFIX, stop=G.SIDE_STOP, max_tokens=G.SIDE_MAX_TOKENS)
            policy.cambrian = object()
            with self.assertRaises(RuntimeError):
                policy.side_generate("sys", user, frame)
        self.assertEqual(action, "<action>move_backward|</action>")
        (act_prompt, act_params), (side_prompt, side_params) = seen
        act_text = self.tok.decode(act_prompt["prompt_token_ids"])
        side_text = self.tok.decode(side_prompt["prompt_token_ids"])
        # the plain prompt ends with the generation prompt (this template has no think block); the side prompt is the
        # same chat with the question in the user turn, the same generation prompt, then the prefix: no end of turn
        gen = "<|im_end|>\n<|im_start|>assistant\n"
        cur = "<|vision_start|><|image_pad|><|vision_end|>" + user.partition("<image>")[2]
        self.assertTrue(act_text.endswith(cur + gen), act_text[-160:])
        head = act_text[: -len(cur + gen)]
        self.assertTrue(head.startswith("<|im_start|>system\nsys<|im_end|>\n") and "<action>turn_left|</action>" in head)
        self.assertEqual(side_text, head + cur + G.side_question(ITEM) + gen + G.SIDE_PREFIX)
        self.assertNotIn("<think>", side_text)
        self.assertEqual(side_prompt["prompt_token_ids"],  # the prefix's tokens = the prefix tokenised in place
                         self.tok.encode(side_text, add_special_tokens=False))
        self.assertEqual(side_prompt["multi_modal_data"], act_prompt["multi_modal_data"])
        self.assertEqual((side_params.temperature, side_params.max_tokens, side_params.stop,
                          side_params.include_stop_str_in_output, getattr(side_params, "seed", None)),
                         (0.0, G.SIDE_MAX_TOKENS, [G.SIDE_STOP], True, None))
        self.assertEqual(act_params.stop, ["</action>"])  # the action call: the plain sampling options
        self.assertEqual((side["text"], side["finish"], side["layout_ok"]), ("front; distance=1.5</state>", "stop", True))
        self.assertEqual(side["prompt_tokens"], len(side_prompt["prompt_token_ids"]))
        self.assertGreater(side["shared_tokens"], len(self.tok.encode(head, add_special_tokens=False)))
        self.assertEqual(policy.last_prompt_tokens, tokens_after_action)  # the action call's reading stays


class SummaryKeyTest(unittest.TestCase):
    def test_keys_by_set_and_merge(self):
        st = "<state>bearing=front; distance=1.0</state>"
        with tempfile.TemporaryDirectory() as tmp:
            folders = []
            for set_name, n in (("eval_val_v2", 3), ("eval_test_v2", 2)):
                res = Path(tmp) / set_name / "m_grd"
                res.mkdir(parents=True)
                rows = [_row(i, i % 2 == 0, [(st, GT_A)]) for i in range(n)]
                for r in rows:
                    r["grounding_turns"][0].update(side_output="front; distance=1.0</state>", side_finish="stop",
                                                   side_prompt_tokens=10, side_layout_ok=True)
                (res / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                folders.append(str(res))
            out = Path(tmp) / "s.json"
            G.main(["summary", "--result", folders[0], "--result", folders[1], "--out", str(out)])
            s = json.loads(out.read_text())
            self.assertEqual({k: v["episodes"] for k, v in s.items()}, {"eval_val_v2/m_grd": 3, "eval_test_v2/m_grd": 2})
            G.main(["summary", "--result", folders[0], "--result", folders[1], "--merge-name", "--out", str(out)])
            s = json.loads(out.read_text())
            self.assertEqual((list(s), s["m_grd"]["episodes"], len(s["m_grd"]["results"])), (["m_grd"], 5, 2))
            self.assertEqual((s["m_grd"]["side"]["parsed"], s["m_grd"]["side"]["layout_ok"],
                              s["m_grd"]["side"]["finish"]), (1.0, 1.0, {"stop": 5}))
            with self.assertRaises(SystemExit):  # the same key twice without --merge-name
                G.main(["summary", "--result", folders[0], "--result", folders[0]])


if __name__ == "__main__":
    unittest.main()
