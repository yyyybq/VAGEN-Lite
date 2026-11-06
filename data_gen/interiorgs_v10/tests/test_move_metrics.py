"""move_metrics.py (experiment 3A, the user's decision 甲 of 2026-11-06): the turn share and the blocked shares of
evaluation results, by replay without rendering, under each result's own rule (model.json env_switches)."""

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_gen.interiorgs_v10 import move_metrics as MM

HAS_SCIPY = importlib.util.find_spec("scipy") is not None
TEMPLATE = Path(MM.__file__).with_name("configs") / "rl_env_v10_S0.yaml"
# five answers that reach the 12-action limit (unless the task is met first): turns, moves, an unreadable answer
ANSWERS = ["<action>turn_left|turn_left|</action>", "<action>move_forward|turn_right|</action>",
           "<action>turn_right|move_backward|turn_left|</action>", "<action>jump|</action>",
           "<action>move_left|turn_left|move_right|turn_right|turn_left|</action>"]


class TurnRecordTest(unittest.TestCase):
    def test_turn_record_reads_the_env_feedback(self):
        fb = "1 of 2 actions executed; blocked: turn_left (blocked by tea table)."
        self.assertEqual(MM.turn_record(["turn_left", "move_forward"], 3, 5, fb, 1),
                         (["turn_left", "move_forward"], ["turn_left"]))
        self.assertEqual(MM.turn_record(["turn_left", "move_forward"], 3, 5, fb, 2), (["turn_left", "move_forward"], None))
        self.assertEqual(MM.turn_record(["turn_left"], 0, 1, "Action executed.", 0), (["turn_left"], []))
        # a turn cut by the action limit (or a success): the parsed actions after it were not executed
        self.assertEqual(MM.turn_record(["move_forward"] * 5, 10, 12, "Action executed.", 0), (["move_forward"] * 2, []))
        two = "1 of 3 actions executed; blocked: move_forward (blocked by wall); turn_right (blocked by wall)."
        self.assertEqual(MM.turn_record(["move_forward", "turn_right", "move_left"], 0, 3, two, 2),
                         (["move_forward", "turn_right", "move_left"], ["move_forward", "turn_right"]))
        # the old feedback names nothing: unknown when a move was blocked; an unreadable answer executes nothing
        self.assertEqual(MM.turn_record(["move_forward"], 0, 1, "Action blocked by collision. [Collision: blocked by wall]", 1),
                         (["move_forward"], None))
        self.assertEqual(MM.turn_record([], 4, 4, "Action had no effect.", 0), ([], []))
        if HAS_SCIPY:  # the strings are the env's
            from vagen.envs.active_spatial.env import collision_feedback

            self.assertEqual(collision_feedback(["turn_left", "move_forward"], [("turn_left", "blocked by tea table")]), fb)
            self.assertEqual(collision_feedback(["move_forward", "turn_right", "move_left"],
                                                [("move_forward", "blocked by wall"), ("turn_right", "blocked by wall")]), two)

    def test_summarize(self):
        eps = [{"index": 0, "replay_same": True, "turns": [
                    {"executed": ["turn_left", "move_forward"], "blocked": ["turn_left"], "blocked_count": 1},
                    {"executed": ["move_forward", "move_forward", "turn_right"], "blocked": [], "blocked_count": 0}]},
               {"index": 1, "replay_same": False, "turns": [
                    {"executed": ["move_left", "turn_left"], "blocked": ["move_left"], "blocked_count": 1},
                    {"executed": ["turn_left"], "blocked": None, "blocked_count": 1},
                    {"executed": [], "blocked": [], "blocked_count": 0}]},
               {"index": 2, "error": "KeyError: 'x'"}]
        self.assertEqual(MM.summarize(eps, 0.25), {
            "rule": 0.25, "episodes": 3, "errors": 1, "replay_same": 1, "turns": 5, "turns_unknown": 1,
            "actions": 8, "rotations": 4, "rotation_share": 0.5, "rotations_blocked": 1, "rotation_blocked_share": 0.25,
            "translations": 4, "translations_blocked": 1, "translation_blocked_share": 0.25,
            "blocked": 3, "blocked_share": 0.375})
        self.assertEqual(MM.summarize([])["actions"], 0)
        self.assertIsNone(MM.summarize([])["rotation_share"])


@unittest.skipUnless(HAS_SCIPY, "the env needs scipy (the cluster env)")
class ReplayTest(unittest.TestCase):
    """Two tasks of the golden scene (one starting above the tea table, inside its footprint; one on open floor),
    evaluated by evaluation.run_episode with a scripted policy and the real env built as cmd_run builds it, under the
    turn rule and under today's rule; move_metrics then replays both folders."""

    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10.contract import pose_from_xy_yaw
        from data_gen.interiorgs_v10.tests.test_turn_footprint import GOLDEN, stub_frame, write_scene

        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        cls.scenes = cls.root / "scenes"
        write_scene(golden, cls.scenes)
        rows = []
        for row, xy_yaw in zip(golden["rows"], ((2.9, 2.5, 0.0), (3.0, 1.0, 0.0))):
            rows.append(dict(row, init_camera=dict(row["init_camera"], extrinsics=pose_from_xy_yaw(*xy_yaw).tolist())))
        cls.ev = cls.root / "eval_set"
        cls.ev.mkdir()
        (cls.ev / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        cls.patch = mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image",
                               lambda self, frame=stub_frame(): frame.copy())
        cls.patch.start()
        cls.new = cls.record("sft_fp", {"prompt_format_example": "action_names", "turn_footprint_radius": 0.25})
        cls.old = cls.record("sft", {"prompt_format_example": "action_names"})

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls._tmp.cleanup()

    @classmethod
    def record(cls, name, switches):
        """An evaluation result folder as evaluation.cmd_run writes it (results.jsonl with trajectories, model.json,
        done.json); returns the env's own counts (executed actions, turns blocked by the footprint)."""
        from data_gen.interiorgs_v10 import evaluation as E
        from vagen.envs.active_spatial.env import ActiveSpatialEnv
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        out = cls.ev / name
        out.mkdir()
        with mock.patch.dict(os.environ, {"R1_RENDER_URL": "http://127.0.0.1:1/render"}):
            cfg = E.env_config(TEMPLATE, cls.ev / "tasks.jsonl", str(cls.scenes), None, switches)
        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                         if k in ActiveSpatialEnvConfig.__dataclass_fields__}))
        counts = {"actions": 0, "footprint_blocks": 0, "met_at_start": []}
        real_step = env.step

        def step(text):
            res = real_step(text)
            counts["footprint_blocks"] += int(res[3]["metrics"]["turn_metrics"].get("turn_footprint_blocks", 0))
            return res

        env.step = step

        class Script:
            def __init__(self):
                self.answers = iter(ANSWERS)

            def generate(self, system, user_text, image):
                return next(self.answers, "<action>jump|</action>")

        rows = []
        for i in range(2):
            env.reset(seed=i)
            counts["met_at_start"].append(bool((env._calculate_canonical_metric() or {}).get("success")))
            rows.append(E.run_episode(env, Script(), i, 12, frames=out / "frames"))
            counts["actions"] += int(env._current_step)
        (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (out / "model.json").write_text(json.dumps({"name": name, "prompt_format": cfg["prompt_format"],
                                                    "env_switches": switches}))
        (out / "done.json").write_text(json.dumps({"episodes": len(rows)}))
        return counts

    def metrics(self, *names, workers=1):
        out = self.root / f"mm_{'_'.join(names)}_{workers}.json"
        args = [a for n in names for a in ("--result", str(self.ev / n))]
        MM.main(args + ["--out", str(out), "--gs-root", str(self.scenes), "--workers", str(workers)])
        return json.loads(out.read_text()), out

    def test_both_rules_replay_as_recorded(self):
        self.assertEqual(self.new["met_at_start"] + self.old["met_at_start"], [False] * 4)
        data, out = self.metrics("sft_fp", "sft")
        new, old = data["results"][str(self.ev / "sft_fp")], data["results"][str(self.ev / "sft")]
        for s in (new, old):
            self.assertEqual((s["episodes"], s["errors"], s["replay_same"], s["turns_unknown"]), (2, 0, 2, 0))
        self.assertEqual((new["rule"], old["rule"]), (0.25, None))
        self.assertEqual((new["actions"], old["actions"]), (self.new["actions"], self.old["actions"]))
        self.assertGreater(new["rotations_blocked"], 0)
        self.assertEqual(new["rotations_blocked"], self.new["footprint_blocks"])  # the env's own count
        self.assertEqual((old["rotations_blocked"], self.old["footprint_blocks"]), (0, 0))  # today's rule
        self.assertEqual(new["blocked"], new["rotations_blocked"] + new["translations_blocked"])
        episodes = data["episodes"][str(self.ev / "sft_fp")]
        self.assertEqual(sorted(e["index"] for e in episodes), [0, 1])
        first = next(e for e in episodes if e["index"] == 0)["turns"][0]  # above the tea table: both turns blocked
        self.assertEqual(first, {"executed": ["turn_left", "turn_left"], "blocked": ["turn_left", "turn_left"],
                                 "blocked_count": 2})
        table = out.with_suffix(".md").read_text(encoding="utf-8")
        self.assertIn("| eval_set/sft_fp | 转身脚印 0.25 m | 2 | 0 | 2 |", table)
        self.assertIn("| eval_set/sft | 旧规则 | 2 | 0 | 2 |", table)
        # the worker pool gives the same numbers
        pooled, _ = self.metrics("sft_fp", workers=2)
        self.assertEqual(pooled["results"][str(self.ev / "sft_fp")], new)

    def test_the_evaluation_records_footprint_blocks_only_under_the_rule(self):
        """evaluation.run_episode with the real env (review fix 3c): under the turn rule every recorded turn carries
        footprint_blocks (the env's turn_footprint_blocks, adding up to the env's own count), under today's rule none."""
        new, old = (MM.episodes_of(self.ev / name) for name in ("sft_fp", "sft"))
        turns = [t for r in new for t in r["trajectory"]]
        self.assertTrue(turns and all(type(t["footprint_blocks"]) is int for t in turns))
        self.assertEqual(sum(t["footprint_blocks"] for t in turns), self.new["footprint_blocks"])
        self.assertTrue(all(t["footprint_blocks"] <= t["collisions"] for t in turns))  # a blocked turn is a blocked move
        self.assertFalse(any("footprint_blocks" in t for r in old for t in r["trajectory"]))

    def test_refusals(self):
        typo = self.ev / "sft_typo"
        shutil.copytree(self.ev / "sft_fp", typo)
        model = json.loads((typo / "model.json").read_text())
        model["env_switches"] = {"turn_footprint_radus": 0.25}  # not an env setting: the env would drop it
        (typo / "model.json").write_text(json.dumps(model))
        with self.assertRaises(SystemExit):
            self.metrics("sft_typo")
        unfinished = self.ev / "sft_unfinished"
        shutil.copytree(self.ev / "sft_fp", unfinished)
        (unfinished / "done.json").unlink()
        with self.assertRaises(SystemExit):
            self.metrics("sft_unfinished")


if __name__ == "__main__":
    unittest.main()
