"""2026-10-15, round 10: the thick wall model (collision_detector.WALL_MODELS "thick").

Every structure.json wall is a slab (its centre line widened by half its thickness to each side) with its
passages cut out: the camera cannot enter the slab except through a passage, and a sight line through a door
has to pass the opening on both faces, so the jambs beside the door block the view.  The historical model
("room_edges", R1 and v10 up to round 9) is unchanged: collision drops every room edge that touches a door and
the visibility rule uses the centre lines."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import difficulty as D
from data_gen.interiorgs_v10.run_all import load_config
from data_gen.interiorgs_v10.run_scene import RunConfig
from data_gen.interiorgs_v10.scene import load_scene
from test_walls_r6 import Y_SIGN, _kitchen_and_dining
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.collision_detector import (WALL_MODELS, CollisionDetector, occlusion_wall_panels,
                                                          thick_wall_segments)

try:
    import scipy  # noqa: F401  (the RL env needs it; the generator-only checks do not)
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

CONFIG_R10 = next(p for p in Path(__file__).resolve().parents if (p / "setup.py").is_file()) / "data_gen" / "interiorgs_v10" / "configs" / "generation_r10_20261015.json"


def _ray_blocked(occ: TV.OccluderSet, origin, point) -> bool:
    o = np.asarray(origin, float)
    p = np.asarray(point, float)
    return bool(TV.blocked_mask(occ, o[None], p[None], p - 1e-3, p + 1e-3)[0, 0])


class ThickWallTest(unittest.TestCase):
    """Kitchen [0,4]x[0,4] below a dining room [0,4]x[4,8]; the 0.2 m wall between them (y = 4) has a door at
    x in [1, 2] up to 2.1 m (test_walls_r6._kitchen_and_dining)."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.scene_dir = _kitchen_and_dining(root)
        cls.structure = json.loads((cls.scene_dir / "structure.json").read_text())
        cls.thin = load_scene(cls.scene_dir, "walls")
        cls.thick = load_scene(cls.scene_dir, "walls", wall_model="thick")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_collision_segments_outline_the_solid_pieces(self):
        segs = thick_wall_segments(self.structure, Y_SIGN)
        between = [s for s in segs if all(abs(float(p[1]) - 4.0) <= 0.1 + 1e-9 for p in s)]
        self.assertEqual(len(between), 8)  # two solid pieces (x 0-1 and 2-4), four edges each
        self.assertEqual(sorted({round(float(p[0]), 3) for s in between for p in s}), [0.0, 1.0, 2.0, 4.0])
        self.assertEqual(sorted({round(float(p[1]), 3) for s in between for p in s}), [3.9, 4.1])
        det, old = self.thick.detector, self.thin.detector
        self.assertEqual((det.wall_model, old.wall_model), ("thick", "room_edges"))
        self.assertEqual(WALL_MODELS, ("room_edges", "thick"))
        with self.assertRaises(ValueError):
            CollisionDetector(wall_model="paper")
        # the door's centre is free, 0.1 m from a jamb is not, and the solid part cannot be crossed
        self.assertFalse(det.check_collision(np.array([1.5, 4.0, 1.5])).has_collision)
        self.assertTrue(det.check_collision(np.array([1.1, 4.0, 1.5])).has_collision)
        self.assertTrue(det.check_collision(np.array([3.0, 4.3, 1.5]),
                                            previous_position=np.array([3.0, 3.5, 1.5])).has_collision)
        self.assertFalse(det.check_collision(np.array([1.5, 4.3, 1.5]),
                                             previous_position=np.array([1.5, 3.5, 1.5])).has_collision)
        # the historical model drops the whole room edge that touches the door: crossing anywhere along it
        self.assertFalse(old.check_collision(np.array([3.0, 4.3, 1.5]),
                                             previous_position=np.array([3.0, 3.5, 1.5])).has_collision)
        # the generator's clearance walls stay the room edges under both models
        self.assertEqual(len(det.clearance_segments), len(old.wall_segments))
        self.assertEqual(len(old.clearance_segments), len(old.wall_segments))
        self.assertEqual(self.thick.wall_distance(2.0, 2.0), self.thin.wall_distance(2.0, 2.0))
        self.assertGreater(len(det.wall_segments), len(old.wall_segments))

    def test_sight_through_a_door_needs_the_opening_on_both_faces(self):
        thin = occlusion_wall_panels(self.structure, Y_SIGN)
        thick = occlusion_wall_panels(self.structure, Y_SIGN, thick=True)
        self.assertEqual(len(thin), len(self.thin.detector.occlusion_walls))  # the default is the r6 model
        self.assertEqual(len(thick), 2 * len(thin) + 2 * len(self.structure["walls"]))  # two faces + two caps
        occ_thin = TV.OccluderSet.from_detector(self.thin.detector)
        occ_thick = TV.OccluderSet.from_detector(self.thick.detector)
        for occ in (occ_thin, occ_thick):
            self.assertFalse(_ray_blocked(occ, (1.5, 7.0, 1.5), (1.5, 2.0, 1.2)))  # straight through the door
            self.assertTrue(_ray_blocked(occ, (3.0, 7.0, 1.5), (3.0, 1.0, 1.5)))   # through the solid wall
            self.assertTrue(_ray_blocked(occ, (1.1, 7.0, 1.5), (1.1, 1.0, 3.0)))   # crosses the wall at 2.25 m: the lintel
        # an oblique line: it meets the centre line inside the door (x = 1.94) but leaves the slab through the
        # far face outside it (x = 2.15): free before, blocked by the jamb now
        self.assertFalse(_ray_blocked(occ_thin, (0.7, 4.6, 1.5), (3.9, 3.05, 1.0)))
        self.assertTrue(_ray_blocked(occ_thick, (0.7, 4.6, 1.5), (3.9, 3.05, 1.0)))
        # the r6 facts still hold with thick walls
        K = np.asarray(TV.item_camera({"init_camera": {"intrinsics": C.NATIVE_K}})[0])
        pose = C.pose_from_xy_yaw(1.5, 7.0, -math.pi / 2)

        def visible(oid):
            obj = next(o for o in self.thick.objects if o.id == oid).target_object_entry()
            return TV.object_visibility(pose, obj, occ_thick, K, 256, 256)["visible"]

        self.assertTrue(visible("2"))    # the cabinet, through the door
        self.assertFalse(visible("1"))   # the refrigerator, behind the solid part
        self.assertFalse(visible("3"))   # the range hood, behind the part above the door

    def test_floor_level_holes_are_passages_only_in_the_thick_model(self):
        structure = json.loads(json.dumps(self.structure))
        hole = [[3.0, 4.0, 2.2], [3.8, 4.0, 2.2], [3.8, 4.0, 0.0], [3.0, 4.0, 0.0]]   # a doorway without a leaf
        high = [[2.2, 4.0, 2.0], [2.8, 4.0, 2.0], [2.8, 4.0, 1.0], [2.2, 4.0, 1.0]]   # a serving hatch (sill 1 m)
        structure["holes"] += [{"type": "HOLE", "profile": [[x, Y_SIGN * y, z] for x, y, z in hole]},
                               {"type": "OPENING", "profile": [[x, Y_SIGN * y, z] for x, y, z in high]}]
        occ_thin = TV.OccluderSet.from_boxes([], wall_panels=occlusion_wall_panels(structure, Y_SIGN))
        occ_thick = TV.OccluderSet.from_boxes([], wall_panels=occlusion_wall_panels(structure, Y_SIGN, thick=True))
        self.assertTrue(_ray_blocked(occ_thin, (3.4, 7.0, 1.5), (3.4, 1.0, 1.5)))    # DOOR only, as in r6
        self.assertFalse(_ray_blocked(occ_thick, (3.4, 7.0, 1.5), (3.4, 1.0, 1.5)))  # the doorway opens the wall
        self.assertTrue(_ray_blocked(occ_thick, (2.5, 7.0, 1.5), (2.5, 1.0, 1.5)))   # the hatch does not
        segs = thick_wall_segments(structure, Y_SIGN)
        xs = sorted({round(float(p[0]), 3) for s in segs for p in s if abs(float(p[1]) - 4.0) <= 0.1 + 1e-9})
        self.assertEqual(xs, [0.0, 1.0, 2.0, 3.0, 3.8, 4.0])  # pieces 0-1, 2-3 and 3.8-4; the hatch stays solid

    def test_fast_geometry_matches_the_detector_under_thick_walls(self):
        sc = self.thick
        rng = np.random.default_rng(3)
        n = 6000
        xy = rng.uniform([-0.5, -0.5], [4.5, 8.5], size=(n, 2))
        xy[: n // 5] = np.round(xy[: n // 5] / 0.05) * 0.05  # exact ties with the axis-aligned faces
        slow = np.array([sc.is_pose_legal(C.pose_from_xy_yaw(float(x), float(y), 0.0))[0] for x, y in xy])
        np.testing.assert_array_equal(sc.legal_mask(xy), slow)
        self.assertGreater(int(slow.sum()), 500)
        ang = rng.uniform(0.0, 2 * math.pi, n)
        b = xy + np.stack([np.cos(ang), np.sin(ang)], 1) * 0.3
        slow_t = np.array([not sc.detector.check_collision(np.array([q[0], q[1], 1.5]),
                                                           previous_position=np.array([p[0], p[1], 1.5])).has_collision
                           for p, q in zip(xy, b)])
        np.testing.assert_array_equal(sc.transitions_legal(xy, b), slow_t)
        self.assertGreater(int((~slow_t).sum()), 200)
        # difficulty.py's replica and FastGeometry (the planner's legality) agree too
        yaw = rng.uniform(0, 2 * math.pi, 800)
        P = np.c_[xy[:800], np.full(800, C.CAMERA_HEIGHT_M)]
        R = np.stack([C.pose_from_xy_yaw(0, 0, t)[:3, :3] for t in yaw])
        Pt = P + np.c_[np.cos(yaw), np.sin(yaw), np.zeros(800)] * C.STEP_TRANSLATION_M
        rep = D._DetectorReplica(sc)
        np.testing.assert_array_equal(rep.poses_legal(P, R), slow[:800])
        np.testing.assert_array_equal(~rep.transitions_collide(P, Pt),
                                      sc.transitions_legal(P[:, :2], Pt[:, :2]))
        geom = D.FastGeometry(sc, validate_samples=64)
        np.testing.assert_array_equal(geom.poses_legal(P, R), slow[:800])
        # the clearance rule still measures from the room edges: 0.3 m from the outer wall is illegal for the
        # generator under both models; the wall between the rooms touches the door, so (as before) it is not a
        # clearance wall and poses near it or in the door are judged alike by both models
        for xy in ((0.3, 2.0), (3.0, 4.5), (1.5, 4.0)):
            self.assertEqual(sc.is_pose_legal(C.pose_from_xy_yaw(*xy, 0.0)), self.thin.is_pose_legal(C.pose_from_xy_yaw(*xy, 0.0)), xy)
        self.assertEqual(sc.is_pose_legal(C.pose_from_xy_yaw(0.3, 2.0, 0.0)), (False, "too_close_to_wall"))
        # inside the slab (0.05 m from its centre line): only the thick model objects
        self.assertEqual(sc.is_pose_legal(C.pose_from_xy_yaw(3.0, 4.15, 0.0)), (False, "collision_wall"))
        self.assertEqual(self.thin.is_pose_legal(C.pose_from_xy_yaw(3.0, 4.15, 0.0)), (True, "ok"))

    def test_generation_config_r10(self):
        fields = load_config(CONFIG_R10)
        self.assertEqual(fields["wall_model"], "thick")
        self.assertEqual(fields["visibility_rule"], "env")
        self.assertTrue(fields["require_structure_walls"] and fields["runtime_success_check"])
        cfg = RunConfig(**fields)
        self.assertEqual(cfg.wall_model, "thick")
        r6 = load_config(CONFIG_R10.with_name("generation_r6_20261014.json"))
        self.assertEqual({k: v for k, v in fields.items() if k != "wall_model"}, r6)  # only the wall model changed
        self.assertEqual(RunConfig().wall_model, "room_edges")


@unittest.skipUnless(HAVE_SCIPY, "the RL env needs scipy")
class ThickWallGenerationTest(unittest.TestCase):
    def test_generated_tasks_replay_in_the_thick_env(self):
        from data_gen.interiorgs_v10 import replay_env as RE
        from data_gen.interiorgs_v10.run_scene import _jsonable, generate_scene
        from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = write_synthetic_scenes(Path(tmp), with_walls=True)["two_rooms_walls"]
            cfg = RunConfig(target_tasks_per_scene=6, wall_model="thick", visibility_rule="env",
                            require_structure_walls=True)
            rows = _jsonable(generate_scene(scene_dir, scene_dir.name, cfg))["rows"]
            self.assertGreaterEqual(len(rows), 3)
            out = RE.replay(rows, scene_dir.parent, {"wall_model": "thick"})
            summary = RE.summarize(out)
            self.assertTrue(summary["all_ok"], summary["failures"])
            self.assertEqual(sorted(r["success_step"] for r in out),
                             sorted(int(r["difficulty"]["d_star"]) for r in rows))


if __name__ == "__main__":
    unittest.main()
