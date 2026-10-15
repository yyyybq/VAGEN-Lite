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


if __name__ == "__main__":
    unittest.main()
