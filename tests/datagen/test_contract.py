import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class ContractTest(unittest.TestCase):
    def test_apply_action_matches_view_manipulator(self):
        from vagen.envs.active_spatial.utils import ViewManipulator

        vm = ViewManipulator(step_translation=C.STEP_TRANSLATION_M, step_rotation_deg=C.STEP_ROTATION_DEG)
        start = C.pose_from_xy_yaw(1.0, 2.0, math.radians(37.0))
        rng = np.random.default_rng(0)
        pose = start.copy()
        vm.reset(start.copy())
        for _ in range(60):
            a = C.ACTIONS[int(rng.integers(len(C.ACTIONS)))]
            pose = C.apply_action(pose, a)
            ref = vm.step(a)
            np.testing.assert_allclose(pose, ref, atol=1e-9)
        # pitch stays zero and height stays constant
        self.assertAlmostEqual(float(pose[2, 3]), C.CAMERA_HEIGHT_M, places=9)
        self.assertAlmostEqual(float(pose[2, 2]), 0.0, places=9)

    def test_inverse_actions(self):
        p = C.pose_from_xy_yaw(0.5, -1.0, 1.0)
        for a in C.ACTIONS:
            q = C.apply_action(C.apply_action(p, a), C.INVERSE_ACTION[a])
            np.testing.assert_allclose(p, q, atol=1e-9)

    def test_turn_left_increases_world_yaw(self):
        p = C.pose_from_xy_yaw(0.0, 0.0, 0.0)
        q = C.apply_action(p, "turn_left")
        self.assertAlmostEqual(math.degrees(C.yaw_of(q)), C.STEP_ROTATION_DEG, places=6)

    def test_synthetic_scenes_load_with_existing_parsers(self):
        from vagen.envs.active_spatial.collision_detector import CollisionDetector

        with tempfile.TemporaryDirectory() as tmp:
            scenes = write_synthetic_scenes(Path(tmp))
            det = CollisionDetector()
            self.assertTrue(det.load_scene(scenes["two_rooms"], scene_id="two_rooms"))
            self.assertEqual(det.structure_y_sign, -1.0)


if __name__ == "__main__":
    unittest.main()
