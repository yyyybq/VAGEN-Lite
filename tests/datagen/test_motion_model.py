"""Tests for the experimental horizon-locked motion model (contract / difficulty / scene)."""

import math
import unittest

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import difficulty as D
from test_difficulty import two_rooms_scene


class MotionModelTest(unittest.TestCase):
    def tearDown(self):
        C.set_motion_model("r1", 0.0)

    def test_r1_is_default_and_rejects_pitch(self):
        self.assertEqual(C.motion_model(), ("r1", 0.0))
        self.assertEqual(C.camera_forward_z(), 0.0)
        with self.assertRaises(ValueError):
            C.set_motion_model("r1", -15.0)
        with self.assertRaises(ValueError):
            C.set_motion_model("free_fly", 0.0)

    def test_horizon_locked_keeps_height_and_pitch(self):
        C.set_motion_model("horizon_locked", -15.0)
        p0 = C.pose_from_xy_yaw(1.0, 2.0, math.radians(30.0))
        fz = math.sin(math.radians(-15.0))
        self.assertAlmostEqual(p0[2, 2], fz, places=12)
        self.assertAlmostEqual(C.camera_forward_z(), fz, places=12)
        np.testing.assert_allclose(p0[:3, :3].T @ p0[:3, :3], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(p0[2, 0], 0.0, places=12)  # right axis stays horizontal
        rng = np.random.default_rng(0)
        P, R = D._split(p0)
        pose = p0
        for a in rng.choice(C.ACTIONS, 40):
            nxt = C.apply_action(pose, a)
            self.assertAlmostEqual(nxt[2, 3], C.CAMERA_HEIGHT_M, places=12)
            self.assertEqual(nxt[2, 2], pose[2, 2])  # turns about world Z leave forward z exact
            P, R = D.apply_action_batch(P, R, a)
            np.testing.assert_allclose(D.to_c2w(P, R)[0], nxt, atol=1e-12)
            pose = nxt
        fwd = C.apply_action(p0, "move_forward")
        self.assertAlmostEqual(math.hypot(*(fwd[:2, 3] - p0[:2, 3])), C.STEP_TRANSLATION_M, places=12)
        left = C.apply_action(p0, "turn_left")
        self.assertAlmostEqual(math.degrees(C.yaw_of(left) - C.yaw_of(p0)), 20.0, places=9)
        for a, inv in C.INVERSE_ACTION.items():
            back = C.apply_action(C.apply_action(p0, a), inv)
            np.testing.assert_allclose(back, p0, atol=1e-12)

    def test_r1_pose_unchanged(self):
        yaw = math.radians(37.0)
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        ref = np.eye(4)
        ref[:3, 0], ref[:3, 1], ref[:3, 2], ref[:3, 3] = r, np.cross(f, r), f, [1.0, 2.0, C.CAMERA_HEIGHT_M]
        self.assertTrue(np.array_equal(C.pose_from_xy_yaw(1.0, 2.0, yaw), ref))

    def test_legality_follows_the_active_pitch(self):
        scene = two_rooms_scene()
        flat = C.pose_from_xy_yaw(2.0, 2.0, 0.0)
        self.assertTrue(scene.is_pose_legal(flat)[0])
        C.set_motion_model("horizon_locked", -15.0)
        pitched = C.pose_from_xy_yaw(2.0, 2.0, 0.0)
        self.assertTrue(scene.is_pose_legal(pitched)[0])
        self.assertEqual(scene.is_pose_legal(flat), (False, "pitch_not_zero"))
        geom = D.FastGeometry(scene)
        Pp, Rp = D._split(pitched)
        Pf, Rf = D._split(flat)
        self.assertTrue(bool(geom.poses_legal(Pp, Rp)[0]))
        self.assertFalse(bool(geom.poses_legal(Pf, Rf)[0]))


if __name__ == "__main__":
    unittest.main()
