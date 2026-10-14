"""release.py: the pose line of the SFT user turns is printed from the env's own pose replay (round 7)."""

import unittest

import numpy as np


class PoseLineTest(unittest.TestCase):
    def test_pose_lines_come_from_the_env_pose_replay(self):
        """A round-7 case: after the third action the generator's replica prints ry=0.00, the env -0.00."""
        from data_gen.interiorgs_v10 import release as R
        from vagen.envs.active_spatial.utils import ViewManipulator

        row = {"init_camera": {"extrinsics": [[0.9751683829486486, 0.0, -0.22146472607916115, -2.645024082553552],
                                              [0.22146472607916115, 0.0, 0.9751683829486486, -4.419794879171985],
                                              [0.0, -1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]},
               "difficulty": {"certificate": ["move_backward", "move_backward", "turn_left", "move_backward"]}}
        replica, runtime = R.certificate_poses(row), R.runtime_poses(row)
        self.assertIn("ry=0.00", R.pose_line(replica[3]))
        self.assertIn("ry=-0.00", R.pose_line(runtime[3]))
        vm = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
        vm.reset(np.asarray(row["init_camera"]["extrinsics"], float))
        for k, a in enumerate(row["difficulty"]["certificate"], start=1):
            vm.step(a)
            self.assertEqual(R.pose_line(runtime[k]), R.pose_line(vm.get_pose()))
            np.testing.assert_allclose(runtime[k], replica[k], atol=1e-9)  # same pose, only the printing differed


if __name__ == "__main__":
    unittest.main()
