"""2026-10-14, round 6 (face_visible_v3): the visibility rule's walls come from structure.json ``walls`` with
door openings cut out, and curtains hide what is behind them; the generator's old box_v1 rule is unchanged."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing import synthetic_scene as S
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.collision_detector import occlusion_wall_panels

Y_SIGN = -1.0  # structure.json stored with the historical -y convention, as in the synthetic scenes


def _kitchen_and_dining(root: Path, with_walls: bool = True) -> Path:
    """Kitchen [0,4]x[0,4] below a dining room [0,4]x[4,8]; the wall between them (y = 4) has a door at
    x in [1, 2] up to 2.1 m.  A curtain hangs across the dining room at y = 6.5, x in [2.6, 4]."""
    labels = [
        S._box("1", "refrigerator", 3.0, 1.0, 0.0, 3.6, 1.6, 1.9),   # behind the solid wall
        S._box("2", "cabinet", 1.2, 1.0, 0.0, 1.8, 1.4, 0.8),        # seen through the door
        S._box("3", "range hood", 1.3, 3.2, 2.2, 1.7, 3.6, 2.6),     # behind the part above the door
        S._box("4", "curtain", 2.6, 6.45, 0.0, 4.0, 6.55, 2.7),
        S._box("5", "sofa", 3.0, 5.0, 0.0, 3.8, 5.8, 0.9),           # behind the curtain from (3.3, 7.5)
    ] + S._walls_for_rect(100, 0.0, 0.0, 4.0, 8.0)
    rooms = [("", [(0, 0), (4, 0), (4, 4), (0, 4)]), ("", [(0, 4), (4, 4), (4, 8), (0, 8)])]
    d = S._write(root / ("walls" if with_walls else "no_walls"), labels, rooms, [], y_sign=Y_SIGN, truth={})
    structure = json.loads((d / "structure.json").read_text())
    door = [[1.0, 4.0, 2.1], [2.0, 4.0, 2.1], [2.0, 4.0, 0.0], [1.0, 4.0, 0.0]]
    window = [[2.5, 0.0, 2.0], [3.5, 0.0, 2.0], [3.5, 0.0, 0.9], [2.5, 0.0, 0.9]]
    structure["holes"] = [{"type": "DOOR", "thickness": 0.2, "profile": [[x, Y_SIGN * y, z] for x, y, z in door]},
                          {"type": "WINDOW", "thickness": 0.2,
                           "profile": [[x, Y_SIGN * y, z] for x, y, z in window]}]
    if with_walls:
        segs = [((0, 4), (4, 4)), ((0, 0), (4, 0)), ((4, 0), (4, 8)), ((4, 8), (0, 8)), ((0, 8), (0, 0))]
        structure["walls"] = [{"thickness": 0.2, "height": 2.8,
                               "location": [[a[0], Y_SIGN * a[1]], [b[0], Y_SIGN * b[1]]]} for a, b in segs]
    (d / "structure.json").write_text(json.dumps(structure))
    return d


class WallPanelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.scene = load_scene(_kitchen_and_dining(root), "walls")
        cls.old = load_scene(_kitchen_and_dining(root, with_walls=False), "no_walls")
        cls.K = np.asarray(TV.item_camera({"init_camera": {"intrinsics": C.NATIVE_K}})[0])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _visible(self, scene, oid, pose, occ=None):
        obj = next(o for o in scene.objects if o.id == oid).target_object_entry()
        occ = occ if occ is not None else TV.OccluderSet.from_detector(scene.detector)
        return TV.object_visibility(pose, obj, occ, self.K, 256, 256)["visible"]

    def test_panels_cut_the_door_and_keep_the_lintel(self):
        self.assertEqual(self.scene.detector.structure_y_sign, Y_SIGN)
        panels = self.scene.detector.occlusion_walls
        between = sorted((round(float(a[0]), 3), round(float(b[0]), 3), z0, z1) for a, b, z0, z1 in panels
                         if abs(a[1] - 4.0) < 1e-9 and abs(b[1] - 4.0) < 1e-9)
        # solid 0-1, the door span 1-2 above the door only, solid 2-4; the window wall (y = 0) stays whole
        self.assertEqual(between, [(0.0, 1.0, 0.0, 2.8), (1.0, 2.0, 2.1, 2.8), (2.0, 4.0, 0.0, 2.8)])
        self.assertEqual(sum(1 for a, b, _, _ in panels if abs(a[1]) < 1e-9 and abs(b[1]) < 1e-9), 1)
        self.assertIsNone(self.old.detector.occlusion_walls)  # no walls list: room outlines as before
        self.assertIsNone(occlusion_wall_panels({"rooms": []}, 1.0))
        self.assertEqual(occlusion_wall_panels({"walls": []}, 1.0), [])

    def test_walls_with_a_door_hide_what_is_behind_them(self):
        pose = C.pose_from_xy_yaw(1.5, 7.0, -math.pi / 2)  # in the dining room, facing the kitchen
        self.assertTrue(self._visible(self.scene, "2", pose))    # through the door
        self.assertFalse(self._visible(self.scene, "1", pose))   # behind the solid part of the wall
        self.assertFalse(self._visible(self.scene, "3", pose))   # behind the part above the door
        # before (room outlines minus every edge touching a door): the whole wall was open
        self.assertTrue(self._visible(self.old, "1", pose))
        self.assertTrue(self._visible(self.old, "3", pose))

    def test_curtains_hide_under_v3_only(self):
        pose = C.pose_from_xy_yaw(3.3, 7.5, -math.pi / 2)  # behind the curtain, facing the sofa
        self.assertFalse(self._visible(self.scene, "5", pose))
        boxes = self.scene.detector.label_boxes
        # the generator's box_v1 occluders (and from_boxes by default) keep curtains see-through
        self.assertTrue(self._visible(self.scene, "5", pose, TV.OccluderSet.from_boxes(boxes)))
        self.assertTrue(TV.is_curtain_label("window curtain") and not TV.is_curtain_label("curtain rod"))

    def test_rule_version(self):
        self.assertEqual(TV.VISIBILITY_RULE_VERSION, "face_visible_v3")


if __name__ == "__main__":
    unittest.main()
