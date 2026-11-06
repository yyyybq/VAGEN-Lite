"""Group G of matrix v3 ("point first, then act", the user's instruction of 2026-11-06): the targets formats, the
reading and judging of the targets segment (vagen/envs/active_spatial/pointing.py) and the env's pointing reward
(off by default: every other run keeps its prompts, rewards and metrics), on a synthetic scene with a stub
renderer and the live S0 template."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10 import rl_checks as RC
from data_gen.interiorgs_v10 import train_chain as TCH
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
from vagen.envs.active_spatial import pointing as PT
from vagen.envs.active_spatial import prompt as P

FRAME = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
ACT = "<action>turn_left|</action>"
TRUTH = [{"letter": "A", "in_view": True, "box": [100.0, 80.0, 140.0, 100.0]},
         {"letter": "B", "in_view": False, "box": None}]


class FormatTest(unittest.TestCase):
    def test_format_lines(self):
        self.assertEqual(P.action_format_line("|", 5, "targets_pixel"),
                         "<targets>A: (x,y); B: (x,y)</targets><action>move_forward|turn_left|</action>")
        self.assertEqual(P.action_format_line("|", 5, "targets_norm1000", n_slots=1),
                         "<targets>A: (x,y)</targets><action>move_forward|turn_left|</action>")
        self.assertEqual(P.action_format_line("|", 5, "action_names"), "<action>move_forward|turn_left|</action>")
        self.assertEqual(P.FORMAT_EXAMPLES, ("placeholder", "action_names", "targets_pixel", "targets_norm1000"))
        with self.assertRaises(ValueError):
            P.action_format_line("|", 5, "targets")

    def test_format_blocks(self):
        first = P.format_prompt_no_think(max_actions_per_step=5, action_sep="|", add_example=True,
                                         format_example="targets_pixel", target_refs=["the lamp", "the sofa"])
        self.assertEqual(first, "Respond with the targets (pixels of the 256 x 256 image), then the action:\n"
                                "<targets>A: (x,y); B: (x,y)</targets><action>move_forward|turn_left|</action>\n"
                                "A is the lamp; B is the sofa.\n\nYou can take up to 5 actions per step.\n\n"
                                "Example: <targets>A: (120,88); B: none</targets><action>move_forward|</action>")
        later = P.format_prompt_no_think(max_actions_per_step=5, action_sep="|", add_example=False,
                                         format_example="targets_norm1000", target_refs=["the lamp"])
        self.assertEqual(later, "Respond with the targets (0 to 1000 across the image), then the action:\n"
                                "<targets>A: (x,y)</targets><action>move_forward|turn_left|</action>\n"
                                "A is the lamp.\n\nYou can take up to 5 actions per step.\n")
        for first_turn in (True, False):  # the other formats ignore the objects
            self.assertEqual(P.format_prompt_no_think(max_actions_per_step=5, format_example="action_names",
                                                      target_refs=["the lamp"], add_example=first_turn),
                             P.format_prompt_no_think(max_actions_per_step=5, format_example="action_names",
                                                      add_example=first_turn))

    def test_system_prompt(self):
        """A targets format changes the description, the response format and the example, nothing else."""
        kw = dict(format="no_think", max_actions_per_step=5, reward_text="actual", task_type="equidistance")
        old = P.system_prompt(format_example="action_names", **kw)
        for fmt in P.TARGET_FORMATS:
            new = P.system_prompt(format_example=fmt, **kw)
            self.assertEqual(new, old.replace("You should provide only your action.", P.targets_description(fmt))
                             .replace("Response format: <action>", "Response format: " + P.targets_line(2) + "<action>")
                             .replace("Example:\n<action>move_forward|</action>", "Example:\n" + P.targets_example(fmt)))
        self.assertNotIn("<targets>", old)
        with self.assertRaises(ValueError):
            P.system_prompt(format="free_think", format_example="targets_pixel")


class ParseJudgeTest(unittest.TestCase):
    def test_parse(self):
        ok = PT.parse_targets("<targets>A: (120,88); B: none</targets>" + ACT, 2)
        self.assertEqual((ok["ok"], ok["entries"]), (True, [(120.0, 88.0), None]))
        loose = PT.parse_targets(" <targets> a : ( 1 , 2.5 ) ;B:NONE </targets>\n<action>turn_left|</action><|im_end|>", 2)
        self.assertEqual((loose["ok"], loose["entries"]), (True, [(1.0, 2.5), None]))
        bad = [("no_targets", ACT),
               ("letters", "<targets>A: (1,2)</targets>" + ACT),
               ("letters", "<targets>B: (1,2); A: none</targets>" + ACT),
               ("structure", "I see the lamp. <targets>A: (1,2); B: none</targets>" + ACT),
               ("structure", "<targets>A: (1,2); B: none</targets>" + ACT + " Done."),
               ("structure", ACT + "<targets>A: (1,2); B: none</targets>"),
               ("structure", "<targets>A: (1,2); B: none</targets>"),
               ("entry", "<targets>A: near the door; B: none</targets>" + ACT),
               ("entry", "<targets>A: (1,2); B: none;</targets>" + ACT)]
        for error, text in bad:
            got = PT.parse_targets(text, 2)
            self.assertEqual((got["ok"], got["error"]), (False, error), text)
        self.assertEqual({k: PT.parse_targets(ACT, 2)[k] for k in ("present", "action_only")},
                         {"present": False, "action_only": True})
        self.assertFalse(PT.parse_targets("I will turn. " + ACT, 2)["action_only"])
        self.assertTrue(PT.parse_targets("<targets>A: (1,2)</targets>" + ACT, 1)["ok"])

    def test_judge_and_label(self):
        score = lambda entries, fmt="targets_pixel": PT.score(entries, TRUTH, fmt)  # noqa: E731
        self.assertEqual(score([(120, 88), None]), 1.0)
        self.assertEqual(score([(10, 10), None]), 0.5)
        self.assertEqual(score([(120, 88), (5, 5)]), 0.5)  # a point for an object not in view
        self.assertEqual(score([None, None]), 0.5)  # "none" for an object in view
        self.assertEqual(score([(469, 344), None], "targets_norm1000"), 1.0)  # (120.1, 88.1) pixels
        self.assertEqual(score([(120, 88), None], "targets_norm1000"), 0.5)  # (30.7, 22.5) pixels
        self.assertEqual(score(None), 0.0)
        self.assertEqual(PT.label(TRUTH, "targets_pixel"), "<targets>A: (120,90); B: none</targets>")
        self.assertEqual(PT.label(TRUTH, "targets_norm1000"), "<targets>A: (469,352); B: none</targets>")
        edge = [{"letter": "A", "in_view": True, "box": [10.6, 0.0, 11.4, 255.0]}]
        for truth in (TRUTH, edge):
            for fmt in P.TARGET_FORMATS:
                parsed = PT.parse_targets(PT.label(truth, fmt) + ACT, len(truth))
                self.assertTrue(parsed["ok"])
                self.assertEqual(PT.score(parsed["entries"], truth, fmt), 1.0, (truth, fmt))
        # a turn: ok only with a readable action; the old format is read, not a format failure
        item = {"target_object": {"objects": [{"ref": "the lamp"}, {"ref": "the sofa"}]}}
        with mock.patch.object(PT, "truth", lambda item, c2w, occ: TRUTH):
            turn = PT.judge_turn("<targets>A: (120,88); B: none</targets>" + ACT, item, None, object(), "targets_pixel")
            self.assertEqual((turn["score"], turn["ok"], turn["old"], turn["bad"]), (1.0, True, False, False))
            turn = PT.judge_turn("<targets>A: (120,88); B: none</targets>" + ACT, item, None, object(), "targets_pixel",
                                 readable=False)
            self.assertEqual((turn["score"], turn["ok"], turn["bad"], turn["error"]), (0.0, False, True, "action"))
            turn = PT.judge_turn(ACT, item, None, object(), "targets_pixel")
            self.assertEqual((turn["score"], turn["old"], turn["bad"]), (0.0, True, False))
            turn = PT.judge_turn("<targets>A: (120,88); B: none</targets>" + ACT, item, None, None, "targets_pixel")
            self.assertEqual((turn["ok"], turn["error"]), (False, "no_truth"))
        self.assertEqual(PT.target_refs(item), ["the lamp", "the sofa"])

    def test_agent_loop_reads_the_judgment(self):
        from vagen.agent_loop.gym_agent_loop import turn_pointing

        info = {"metrics": {"turn_metrics": {"pointing_score": 0.5, "pointing_applied": 0.05, "targets_bad": False,
                                             "targets_old": True, "targets_ok": False, "action_is_valid": True}}}
        self.assertEqual(turn_pointing(info), {"pointing_score": 0.5, "pointing_applied": 0.05, "targets_bad": False,
                                               "targets_old": True})
        self.assertEqual(turn_pointing({"metrics": {"turn_metrics": {"action_is_valid": True}}}), {})
        self.assertEqual(turn_pointing(None), {})

    def test_chain_passes_the_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = TCH.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 300, "gs_root": "/ply",
                               "rl": {"standard": "v10_p1_S0", "seed": 20261019, "rows": "/rows.jsonl",
                                      "task_pool": "reserve", "format_example": "targets_pixel"},
                               "rl_runs": {"ps0": {}}}, sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_g_e9", "model": "/lf/e9"}
            calls = {}
            chain.sh = lambda args, **k: calls.__setitem__(args[args.index("--out") + 1], args)
            chain.rl_config()
            self.assertNotIn("--pointing-reward", calls[str(root / "rl_ps0")])
            chain.plan["rl"]["pointing_reward"] = 0.1
            chain.rl_config()
            args = calls[str(root / "rl_ps0")]
            self.assertEqual((args[args.index("--pointing-reward") + 1], args[args.index("--format-example") + 1]),
                             ("0.1", "targets_pixel"))


class EnvPointingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        scene_dir = paths["two_rooms"]
        cls.rows = _jsonable(generate_scene(scene_dir, scene_dir.name, RunConfig(target_tasks_per_scene=8)))["rows"]
        cls.cfg = RC.env_config(Path(TP.__file__).with_name("configs") / "rl_env_v10_S0.yaml", "http://127.0.0.1:1/render")
        cls.cfg["gs_root"] = str(scene_dir.parent)
        cls.patch = mock.patch("vagen.envs.active_spatial.env.ActiveSpatialEnv._render_image", lambda self: FRAME.copy())
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls._tmp.cleanup()

    def env(self, row, name, **switches):
        env, _ = RC.make_env([row], self.root / f"{name}.jsonl", dict(self.cfg, **switches))
        obs, _ = env.reset(seed=0)
        return env, obs

    @staticmethod
    def step(env, text):
        _, reward, done, info = env.step(text)
        trace = info.get("reward_trace") or {}
        return float(reward), info["metrics"]["turn_metrics"], info, trace

    def truth(self, env, row):
        """The truth at the start frame, from the env and (as the SFT labels get it) from the scene files."""
        c2w = env.view_engine.get_pose()
        mine = PT.truth(row, c2w, PT.scene_occluders(self.cfg["gs_root"], row["scene_id"], self.cfg["wall_model"]))
        self.assertEqual(mine, PT.truth(row, c2w, env._visibility_occluders()))
        return mine

    def rows_with(self, n_slots):
        return [r for r in self.rows if len(PT.slot_objects(r)) == n_slots]

    def test_off_by_default(self):
        row = self.rows[0]
        for name, fmt in (("off_names", "action_names"), ("off_placeholder", "placeholder")):
            env, obs = self.env(row, name, prompt_format_example=fmt)
            self.assertNotIn("<targets>", obs["obs_str"] + env.system_prompt())
            reward, tm, info, trace = self.step(env, "<targets>A: (1,1)</targets>" + ACT)
            self.assertTrue(tm["action_is_valid"])  # the action is read as before
            self.assertFalse({"pointing_score", "targets_ok", "targets_old", "targets_bad"} & set(tm))
            self.assertNotIn("pointing", info)
            self.assertNotIn("pointing", trace.get("components") or {})

    def test_prompt_names_the_objects(self):
        for n in (1, 2):
            rows = self.rows_with(n)
            if not rows:
                continue
            env, obs = self.env(rows[0], f"prompt{n}", prompt_format_example="targets_pixel")
            block = P.targets_format_block("targets_pixel", 5, "|", True, PT.target_refs(rows[0]))
            self.assertTrue(obs["obs_str"].endswith("\n" + block), obs["obs_str"])
            self.assertIn("Response format: <targets>A: (x,y); B: (x,y)</targets>", env.system_prompt())
            obs2, _, _, _ = env.step(ACT)
            self.assertTrue(obs2["obs_str"].endswith("\n" + P.targets_format_block("targets_pixel", 5, "|", False,
                                                                                    PT.target_refs(rows[0]))))

    def test_judged_on_the_frame_answered(self):
        """At weight 0 the env judges and logs the segment and the reward is that of the same action without it; at
        weight 0.1 it adds 0.1 times the score, recorded as the "pointing" component."""
        rows = self.rows_with(2)[:2] + self.rows_with(1)[:1]
        self.assertTrue(self.rows_with(2))
        for row in rows:
            base_env, _ = self.env(row, "base", prompt_format_example="action_names")
            base, _, _, _ = self.step(base_env, ACT)
            env, _ = self.env(row, "judge0", prompt_format_example="targets_pixel")
            truth = self.truth(env, row)
            self.assertTrue(any(t["in_view"] for t in truth))  # the start rule: a target identifiable at the start
            right = PT.label(truth, "targets_pixel") + ACT
            reward, tm, info, trace = self.step(env, right)
            self.assertEqual((tm["pointing_score"], tm["targets_ok"], tm["targets_old"], tm["targets_bad"]),
                             (1.0, True, False, False))
            self.assertEqual(info["pointing"]["truth"], truth)
            self.assertAlmostEqual(reward, base, places=9)
            self.assertEqual((trace["components"]["pointing"]["applied"], tm["pointing_applied"]), (0.0, 0.0))
            cases = [(0.1, right, 1.0)]
            if len(truth) == 2:  # half right: the first object as labelled, the second wrong
                first = PT.label(truth[:1], "targets_pixel")[len("<targets>"):-len("</targets>")]
                second = "B: (300,300)" if truth[1]["in_view"] else "B: (5,5)"
                half = f"<targets>{first}; {second}</targets>" + ACT
                cases += [(0.1, half, 0.5), (0.25, half, 0.5)]
            for weight, text, score in cases:
                env, _ = self.env(row, f"judge_{weight}", prompt_format_example="targets_pixel", pointing_reward=weight)
                reward, tm, info, trace = self.step(env, text)
                self.assertEqual(tm["pointing_score"], score)
                self.assertAlmostEqual(tm["pointing_applied"], weight * score, places=12)
                self.assertAlmostEqual(reward, base + weight * score, places=9)
                self.assertAlmostEqual(trace["components"]["pointing"]["applied"], weight * score, places=9)
                self.assertAlmostEqual(trace["accounting"]["reconstruction_error"], 0.0, places=9)

    def test_old_bad_and_unreadable_answers(self):
        row = self.rows[0]
        base_env, _ = self.env(row, "base_old", prompt_format_example="action_names")
        base, _, _, _ = self.step(base_env, ACT)
        cases = [(ACT, (False, True, False), base),  # the old format: read, no pointing reward
                 ("I will turn. " + ACT, (False, False, True), base),  # free text: read, but not the short structure
                 ("<targets>A: (1,1)</targets>", (False, False, True), None)]  # no action: the invalid penalty only
        for text, flags, expected in cases:
            env, _ = self.env(row, "old_bad", prompt_format_example="targets_pixel", pointing_reward=0.1)
            reward, tm, info, trace = self.step(env, text)
            self.assertEqual((tm["targets_ok"], tm["targets_old"], tm["targets_bad"]), flags, text)
            self.assertEqual(tm["pointing_score"], 0.0)
            self.assertAlmostEqual(reward, base if expected is not None else env._invalid_penalty(), places=9)
            self.assertAlmostEqual(trace["accounting"]["reconstruction_error"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
