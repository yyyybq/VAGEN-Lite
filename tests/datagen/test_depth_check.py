"""depth_check.py: the visibility rule with the rendered depth (terminal frames of val and test)."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import depth_check as D
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing import synthetic_scene as S
from vagen.envs.active_spatial import target_visibility as TV


class DepthCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        labels = [S._box("1", "refrigerator", 4.30, 1.50, 0.0, 5.00, 2.20, 1.90)] + S._walls_for_rect(100, 0, 0, 6, 4)
        d = S._write(root / "fridge", labels, [("", [(0, 0), (6, 0), (6, 4), (0, 4)])], [], y_sign=-1.0, truth={})
        cls.scene = load_scene(d, "fridge")
        cls.occ = TV.OccluderSet.from_detector(cls.scene.detector)
        fridge = next(o for o in cls.scene.objects if o.id == "1").target_object_entry()
        cls.item = {"task_type": "absolute_positioning", "init_camera": {"intrinsics": C.NATIVE_K},
                    "target_region": {"params": {}}, "target_object": {"objects": [fridge]}}
        cls.pose = C.pose_from_xy_yaw(1.0, 1.85, 0.0)  # facing the fridge along +x, 3.3 m away

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _depth(self, value):
        return np.full((256, 256), value, np.float32), np.ones((256, 256), np.float32)

    def test_open_view_and_unlabelled_wall(self):
        far, alpha = self._depth(10.0)  # nothing in front of the fridge
        res = D.check_frame(self.item, self.pose, far, alpha, self.occ)
        self.assertTrue(res["objects"]["target_a"]["rule_visible"])
        self.assertTrue(res["objects"]["target_a"]["depth_visible"])
        self.assertFalse(res["hidden_by_unlabelled_geometry"])
        wall, _ = self._depth(1.5)  # an unlabelled wall 1.5 m ahead covers the view
        res = D.check_frame(self.item, self.pose, wall, alpha, self.occ)
        self.assertTrue(res["objects"]["target_a"]["rule_visible"])
        self.assertFalse(res["objects"]["target_a"]["depth_visible"])
        self.assertEqual(res["flagged_objects"], ["target_a"])
        # the fridge's own surface (just inside its box) and sparse splats never hide it
        own, _ = self._depth(3.25)
        self.assertFalse(D.check_frame(self.item, self.pose, own, alpha, self.occ)["hidden_by_unlabelled_geometry"])
        self.assertFalse(D.check_frame(self.item, self.pose, wall, np.full((256, 256), 0.2, np.float32),
                                       self.occ)["hidden_by_unlabelled_geometry"])

    def test_terminal_pose_and_summary(self):
        row = {"init_camera": {"extrinsics": C.pose_from_xy_yaw(1.0, 1.0, 0.0).tolist()},
               "difficulty": {"certificate": ["move_forward", "turn_left"]}}
        want = C.apply_action(C.apply_action(C.pose_from_xy_yaw(1.0, 1.0, 0.0), "move_forward"), "turn_left")
        np.testing.assert_allclose(D.terminal_pose(row), want)
        tasks = [{"task_id": "a", "task_type": "fov_inclusion", "split": "val"},
                 {"task_id": "b", "task_type": "fov_inclusion", "split": "test"},
                 {"task_id": "c", "task_type": "equidistance", "split": "test"}]
        results = [{"task_id": "a", "hidden_by_unlabelled_geometry": True, "flagged_objects": ["target_b"]},
                   {"task_id": "b", "hidden_by_unlabelled_geometry": False, "flagged_objects": []},
                   {"task_id": "c", "error": "boom"}]
        s = D.summarize(tasks, results)
        self.assertEqual((s["checked"], s["errors"], s["flagged"]), (2, 1, 1))
        self.assertEqual(s["by_family"]["fov_inclusion"], {"checked": 2, "flagged": 1, "share": 0.5})
        self.assertEqual(s["flagged_task_ids"], ["a"])


if __name__ == "__main__":
    unittest.main()
