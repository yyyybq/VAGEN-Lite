"""runtime_success_check: the env's own success rule first fires exactly at d*."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10.difficulty import certify_d_star, make_success_fn
from data_gen.interiorgs_v10.initial_states import RUNTIME_LOOSER_FAMILIES
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class RuntimeSuccessCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = write_synthetic_scenes(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_env_rule_reaches_success_exactly_at_d_star(self):
        self.assertEqual(RUNTIME_LOOSER_FAMILIES, {"projective_relations", "fov_inclusion"})
        checked = 0
        for name in ("two_rooms", "dense_duplicates"):
            sdir = self.paths[name]
            cfg = RunConfig(target_tasks_per_scene=6, runtime_success_check=True, budget_mode="work",
                            budget_work_units=1e9)
            res = _jsonable(generate_scene(sdir, name, cfg))
            scene = load_scene(sdir, name)
            for row in res["rows"]:
                if row["task_type"] not in RUNTIME_LOOSER_FAMILIES:
                    continue
                d = int(row["difficulty"]["d_star"])
                rt = certify_d_star(scene, make_success_fn(row), np.asarray(row["init_camera"]["extrinsics"]),
                                    d_max=8, max_terminal_checks=0)
                self.assertEqual(rt["d_star"], d, row["task_id"])
                checked += 1
        self.assertGreater(checked, 0)

    def test_off_by_default(self):
        self.assertFalse(RunConfig().runtime_success_check)


if __name__ == "__main__":
    unittest.main()
