"""Round 9 env switches (all off by default, so R1 and the v10 runs keep their behaviour), on a synthetic scene with
a stub renderer and the S0 config."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10 import rl_checks as RC
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

FRAME = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
INVERSE = {"move_forward": "move_backward", "move_backward": "move_forward", "move_left": "move_right",
           "move_right": "move_left", "turn_left": "turn_right", "turn_right": "turn_left"}


def feedback(obs_str):
    """The "Environment Feedback: ..." line of an observation."""
    return next(line for line in obs_str.splitlines() if line.startswith("Environment Feedback: "))


def answer(actions):
    return "<action>" + "".join(a + "|" for a in actions) + "</action>"


class EnvSwitchTest(unittest.TestCase):
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
        env.reset(seed=0)
        return env

    def test_success_checked_after_every_action(self):
        """Switch 1: the shortest path plus one action that undoes its last step, in one answer."""
        rows = [r for r in self.rows if 1 <= len(r["difficulty"]["certificate"]) <= 4][:3]
        self.assertTrue(rows)
        for row in rows:
            cert = list(row["difficulty"]["certificate"])
            text = answer(cert + [INVERSE[cert[-1]]])
            off = RC.step(self.env(row, "off"), text)  # default: checked once, after the whole answer
            self.assertEqual((off["success"], off["done"], off["env_step"]), (False, False, len(cert) + 1))
            env = self.env(row, "each", success_check_each_action=True)
            on = RC.step(env, text)
            self.assertEqual((on["success"], on["done"], on["env_step"]), (True, True, len(cert)))
            self.assertEqual(on["reward"], 5.0)  # success, and the turn ends inside the action loop: no step penalty
            self.assertIn("auto_terminated", on["flags"])

    def test_action_limit_is_a_setting(self):
        """Switch 2: 15 turns in place go past the template's 12-action limit when it is raised to 20."""
        for row in self.rows:
            runs = {}
            for limit in (12, 20):
                env = self.env(row, f"limit{limit}", max_episode_steps=limit)
                runs[limit] = [RC.step(env, answer(["turn_left"] * 5)) for _ in range(3)]
            if any(r["success"] for rs in runs.values() for r in rs):
                continue  # this task is met by turning in place; take another
            self.assertEqual([(r["env_step"], r["done"]) for r in runs[12]], [(5, False), (10, False), (12, True)])
            self.assertIn("truncated_max_primitive_steps", runs[12][-1]["flags"])
            self.assertEqual([(r["env_step"], r["done"]) for r in runs[20]], [(5, False), (10, False), (15, False)])
            return
        self.fail("no task to test on")


    def blocked_run(self):
        """A task and a move that runs into something: 3 turns of 5 identical moves end by repeated collisions."""
        if getattr(self, "_blocked", None) is None:
            for row in self.rows:
                for move in ("move_forward", "move_backward", "move_left", "move_right"):
                    env = self.env(row, "probe")
                    steps = []
                    for _ in range(3):
                        steps.append(RC.step(env, answer([move] * 5)))
                        if steps[-1]["done"]:
                            break
                    if "early_terminated_collision" in steps[-1]["flags"] and not any(s["success"] for s in steps):
                        type(self)._blocked = (row, move, len(steps))
                        return self._blocked
            self.fail("no move runs into anything in the synthetic scene")
        return self._blocked

    def test_repeated_collisions_need_not_end_the_task(self):
        """Switch 3: with max_consecutive_collisions = 0 the task goes on; each blocked move still costs -0.15."""
        row, move, n = self.blocked_run()
        env = self.env(row, "nocollend", max_consecutive_collisions=0)
        steps = [RC.step(env, answer([move] * 5)) for _ in range(n)]
        self.assertFalse(any(s["done"] for s in steps))
        self.assertFalse(any("early_terminated_collision" in s["flags"] for s in steps))
        last = steps[-1]
        self.assertGreater(last["collisions"], 0)
        self.assertEqual(last["components"]["collision"]["events"], last["collisions"])
        self.assertEqual(last["components"]["collision"]["scale"], -0.15)
        self.assertNotIn("consecutive_collision", last["components"])
        self.assertAlmostEqual(last["reward"], -0.01 - 0.15 * last["collisions"])

    def test_collision_feedback_says_what_was_blocked(self):
        """Switch 4: the same blocked moves, with the old feedback and with the detailed one."""
        row, move, n = self.blocked_run()
        runs = {}
        for detail in (False, True):
            env = self.env(row, f"fb{detail}", max_consecutive_collisions=0, collision_feedback_detail=detail)
            runs[detail] = [RC.step(env, answer([move] * 5)) for _ in range(n)]
        blocked_turns = [k for k, s in enumerate(runs[True]) if s["collisions"]]
        self.assertTrue(blocked_turns)
        for k in blocked_turns:
            c = runs[True][k]["collisions"]
            self.assertEqual(runs[False][k]["collisions"], c)
            old = runs[False][k]["obs"]["obs_str"]
            new = runs[True][k]["obs"]["obs_str"]
            self.assertIn("Environment Feedback: Action executed." if c < 5 else
                          "Environment Feedback: Action blocked by collision.", old)  # partial blocks were silent
            self.assertIn(f"Environment Feedback: {5 - c} of 5 actions executed; blocked: ", new)
            blocked_list = feedback(new).split("blocked: ", 1)[1]
            self.assertEqual(blocked_list.count(f"{move} (blocked by "), c)
            self.assertEqual(old.replace(feedback(old), ""), new.replace(feedback(new), ""))  # the rest is identical


    def test_pose_zero_sign_setting_reaches_the_pose_line(self):
        """Next data version: pose_normalize_negative_zero is passed to the pose formatter (off by default)."""
        from vagen.envs.active_spatial.utils import format_pose6_deg

        for setting in (False, True):
            with mock.patch("vagen.envs.active_spatial.env.format_pose6_deg", wraps=format_pose6_deg) as fmt:
                env = self.env(self.rows[0], f"pose{setting}", pose_normalize_negative_zero=setting)
                RC.step(env, answer(["turn_left"]))
            self.assertTrue(fmt.call_args_list)
            self.assertEqual({c.kwargs.get("normalize_zero") for c in fmt.call_args_list}, {setting})


if __name__ == "__main__":
    unittest.main()
