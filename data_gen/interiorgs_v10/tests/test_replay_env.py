"""replay_env.py: generated tasks replayed in an env copy succeed exactly at d*."""

import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import replay_env as RE
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class ReplayEnvTest(unittest.TestCase):
    def test_select_rows(self):
        rows = [{"task_id": f"t{i}", "task_type": "a" if i % 2 else "b",
                 "collision_convention": {"requires_env_override": i < 3,
                                          "ignored_object_ids": ["9"] if i == 7 else []}} for i in range(20)]
        picked = RE.select_rows(rows, per_family=2, per_group=1, seed=0)
        ids = {r["task_id"] for r in picked}
        self.assertIn("t7", ids)  # every row of a scene with ignored objects
        self.assertLessEqual(len(ids), 2 + 2 + 1 + 1)
        self.assertEqual(RE.select_rows(rows, 2, 1, seed=0), picked)
        rows += [dict(r, task_id=f"u{i}", collision_convention={"ignored_object_ids": ["9"]})
                 for i, r in enumerate(rows)]
        capped = RE.select_rows(rows, per_family=0, per_group=0, seed=0, per_ignored=3)
        self.assertEqual(len(capped), 3)
        self.assertEqual(len(RE.select_rows(rows, 0, 0, seed=0)), 21)

    def test_generated_tasks_succeed_at_d_star_in_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_synthetic_scenes(Path(tmp))
            scene_dir = paths["two_rooms"]  # the env loads <gs_root>/<scene_id>
            res = _jsonable(generate_scene(scene_dir, scene_dir.name, RunConfig(target_tasks_per_scene=5)))
            rows = res["rows"]
            self.assertGreaterEqual(len(rows), 3)
            out = RE.replay(rows, scene_dir.parent)
            summary = RE.summarize(out)
            self.assertTrue(summary["all_ok"], summary["failures"])
            self.assertEqual(sorted(r["success_step"] for r in out),
                             sorted(int(r["difficulty"]["d_star"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
