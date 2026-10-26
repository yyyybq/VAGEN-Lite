"""Tests for difficulty.py (exact d* certification) and initial_states.py."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import difficulty as D
from data_gen.interiorgs_v10 import initial_states as I
from data_gen.interiorgs_v10.sampling import floor_grid, fov_item, projective_item, score, yaw_set
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class DiskSuccess:
    """Synthetic success region: position within r of a centre and yaw within tol."""

    def __init__(self, cx, cy, r, yaw_deg, tol_deg):
        self.cx, self.cy, self.r, self.yaw, self.tol = cx, cy, r, yaw_deg, tol_deg

    def __call__(self, c2w):
        c2w = np.asarray(c2w)
        dy = abs((math.degrees(C.yaw_of(c2w)) - self.yaw + 180.0) % 360.0 - 180.0)
        return math.hypot(c2w[0, 3] - self.cx, c2w[1, 3] - self.cy) <= self.r and dy <= self.tol


_SCENE = None
_TMP = None


def two_rooms_scene():
    global _SCENE, _TMP
    if _SCENE is None:
        _TMP = tempfile.TemporaryDirectory()
        paths = write_synthetic_scenes(Path(_TMP.name))
        _SCENE = load_scene(paths["two_rooms"], "two_rooms")
    return _SCENE


def random_legal_pose(scene, rng):
    while True:
        x, y = rng.uniform(0.5, 9.5), rng.uniform(0.5, 4.5)
        p = C.pose_from_xy_yaw(x, y, math.radians(float(rng.choice(np.arange(0, 360, 10)))))
        if scene.is_pose_legal(p)[0]:
            return p


def scan_endpoints(scene, item, spacing=0.3, yaw_step=10.0):
    """Endpoints by scanning sampling.floor_grid x yaw_set with sampling.score
    (vectorised prefilter, every endpoint confirmed by the frozen scorer)."""
    grid = floor_grid(scene, spacing=spacing)
    poses = np.stack([C.pose_from_xy_yaw(x, y, t) for x, y in grid for t in yaw_set(yaw_step)])
    fast = D.CanonicalSuccess(item).batch(poses)
    return [p for p in poses[fast] if score(item, p)["success"]]


class DifficultyCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = two_rooms_scene()
        cls.objs = {o.id: o for o in cls.scene.objects}

    def test_batch_apply_and_keys_match_contract(self):
        rng = np.random.default_rng(0)
        off = 1 << 21
        for _ in range(60):
            p = C.pose_from_xy_yaw(rng.uniform(0, 10), rng.uniform(0, 5), rng.uniform(0, 2 * math.pi))
            P, R = D._split(p)
            for a in rng.choice(C.ACTIONS, 12):
                p = C.apply_action(p, a)
                P, R = D.apply_action_batch(P, R, a)
                np.testing.assert_allclose(D.to_c2w(P, R)[0], p, atol=1e-12)
                k = C.state_key(p)
                self.assertEqual(int(D.contract_keys(P, R)[0]), ((k[0] + off) << 31) | ((k[1] + off) << 9) | k[2])

    def test_fast_geometry_matches_scalar_reference(self):
        scene = self.scene
        geom = D.FastGeometry(scene, validate_samples=64)
        rng = np.random.default_rng(1)
        n = 1500
        xy = rng.uniform([-0.5, -0.5], [10.5, 5.5], size=(n, 2))
        yaw = rng.uniform(0, 2 * np.pi, n)
        P = np.c_[xy, np.full(n, C.CAMERA_HEIGHT_M)]
        R = np.stack([C.pose_from_xy_yaw(0, 0, t)[:3, :3] for t in yaw])
        Pt = P + np.c_[np.cos(yaw), np.sin(yaw), np.zeros(n)] * C.STEP_TRANSLATION_M
        ref_t = np.array([scene.transition_legal(D.to_c2w(P[i:i + 1], R[i:i + 1])[0],
                                                 D.to_c2w(Pt[i:i + 1], R[i:i + 1])[0])[0] for i in range(n)])
        ref_p = np.array([scene.is_pose_legal(D.to_c2w(P[i:i + 1], R[i:i + 1])[0])[0] for i in range(n)])
        np.testing.assert_array_equal(geom.transitions_legal(P, Pt, R), ref_t)
        np.testing.assert_array_equal(geom.poses_legal(P, R), ref_p)
        rep = D._DetectorReplica(scene)  # the local fallback must also be exact
        np.testing.assert_array_equal(~rep.transitions_collide(P, Pt), ref_t)
        np.testing.assert_array_equal(rep.poses_legal(P, R), ref_p)
        self.assertGreater(ref_t.mean(), 0.2)
        self.assertLess(ref_t.mean(), 0.95)

    def test_canonical_success_matches_frozen_score(self):
        o = self.objs
        items = [projective_item("two_rooms", o["2"], o["4"], "left"),
                 projective_item("two_rooms", o["5"], o["1"], "right"),
                 fov_item("two_rooms", o["10"], o["13"])]
        rng = np.random.default_rng(2)
        for item in items:
            cs = D.CanonicalSuccess(item)
            xy = rng.uniform([0.3, 0.3], [9.7, 4.7], size=(600, 2))
            yaws = rng.uniform(0, 2 * np.pi, 600)
            poses = np.stack([C.pose_from_xy_yaw(x, y, t) for (x, y), t in zip(xy, yaws)])
            fast = cs.batch(poses)
            ref = np.array([score(item, p)["success"] for p in poses])
            np.testing.assert_array_equal(fast, ref)
            self.assertGreater(ref.sum(), 5)

    def test_bfs_equals_brute_force_on_random_cases(self):
        scene = self.scene
        rng = np.random.default_rng(3)
        seen = []
        for _ in range(30):
            init = random_legal_pose(scene, rng)
            fn = DiskSuccess(init[0, 3] + rng.uniform(-0.9, 0.9), init[1, 3] + rng.uniform(-0.9, 0.9),
                             rng.uniform(0.12, 0.35), rng.uniform(0, 360), rng.uniform(10, 45))
            if fn(init):
                continue
            bf = D.brute_force_d_star(scene, fn, init, depth=4)
            res = D.certify_d_star(scene, fn, init, d_max=4)
            self.assertEqual(res["d_star"], bf)
            self.assertTrue(res["lower_bound_complete"])
            if bf is not None:
                self.assertEqual(len(res["certificate"]), bf)
                self.assertTrue(D.replay_certificate(scene, fn, init, res["certificate"])["ok"])
            seen.append(bf)
        # canonical Projective judge: inits a few random legal steps away from endpoints
        o = self.objs
        item = projective_item("two_rooms", o["2"], o["4"], "left")
        succ = D.make_success_fn(item)
        ends = scan_endpoints(scene, item, spacing=0.6, yaw_step=20.0)
        self.assertTrue(ends)
        for k in range(4):
            p = ends[int(rng.integers(len(ends)))]
            for a in rng.choice(C.ACTIONS, 2 + k % 2):
                q = C.apply_action(p, a)
                if scene.transition_legal(p, q)[0]:
                    p = q
            if succ.exact(p) or not scene.is_pose_legal(p)[0]:
                continue
            bf = D.brute_force_d_star(scene, succ, p, depth=4)
            res = D.certify_d_star(scene, succ, p, d_max=4)
            self.assertEqual(res["d_star"], bf)
            seen.append(bf)
        self.assertGreaterEqual(len(set(seen)), 3, seen)

    def test_hint_matches_plain_bfs_and_state_cap_is_reported(self):
        scene = self.scene
        o = self.objs
        item = projective_item("two_rooms", o["2"], o["4"], "left")
        succ = D.make_success_fn(item)
        init = C.pose_from_xy_yaw(3.0, 2.5, math.radians(10.0))
        plain = D.certify_d_star(scene, succ, init, d_max=8)
        self.assertIsNotNone(plain["d_star"])
        hinted = D.certify_d_star(scene, succ, init, d_max=8, hint_actions=plain["certificate"])
        self.assertEqual(hinted["d_star"], plain["d_star"])
        self.assertEqual(hinted["certificate_source"], "hint")
        self.assertLess(hinted["states"], plain["states"])
        capped = D.certify_d_star(scene, succ, init, d_max=8, max_states=500)
        self.assertIsNone(capped["d_star"])
        self.assertFalse(capped["lower_bound_complete"])
        self.assertEqual(capped["status"], "state_cap")
        # contract-key mode agrees here as well
        self.assertEqual(D.certify_d_star(scene, succ, init, d_max=8, key_mode="contract")["d_star"],
                         plain["d_star"])

    def test_chunk_turns(self):
        rng = np.random.default_rng(4)
        for n in range(0, 13):
            acts = [str(a) for a in rng.choice(C.ACTIONS, n)]
            turns = I.chunk_turns(acts)
            self.assertEqual(sum(turns, []), acts)
            self.assertEqual(len(turns), -(-n // 3))
            self.assertTrue(all(1 <= len(t) <= 3 for t in turns))
        self.assertEqual(I.chunk_turns(["turn_left"] * 2 + ["move_forward"] * 3 + ["move_right"] * 2),
                         [["turn_left"] * 2, ["move_forward"] * 3, ["move_right"] * 2])
        turns = I.chunk_turns(["move_left", "turn_left", "move_right", "move_forward"])
        self.assertFalse(any(I._contradictory(t) for t in turns))


class InitialStatesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = two_rooms_scene()
        o = {x.id: x for x in cls.scene.objects}
        specs = [("2", "4", "left"), ("5", "1", "right"), ("10", "13", "left")]
        cls.entries = []
        for a, b, rel in specs:
            item = projective_item("two_rooms", o[a], o[b], rel)
            cls.entries.append({"item": item, "endpoints": scan_endpoints(cls.scene, item)})
        fn, cls.obs_source = I.resolve_observability_fn(None)
        cls.obs_fn = staticmethod(fn)
        cls.results = []
        for k, e in enumerate(cls.entries):
            eps, funnel = I.propose_initial_states(cls.scene, e["item"], None, e["endpoints"],
                                                   quota_per_band=3, max_per_item=12, seed=k)
            cls.results.append((e, eps, funnel))

    def _check_episode(self, item, ep):
        scene = self.scene
        succ = D.make_success_fn(item)
        init = np.asarray(ep["init_c2w"])
        self.assertTrue(scene.is_pose_legal(init)[0])
        self.assertFalse(succ.exact(init))
        # replay with the contract motion model and the scalar runtime rule
        pose = init
        for i, a in enumerate(ep["certificate"], start=1):
            nxt = C.apply_action(pose, a)
            self.assertTrue(scene.transition_legal(pose, nxt)[0], (i, a))
            pose = nxt
            self.assertEqual(bool(score(item, pose)["success"]), i == len(ep["certificate"]), i)
        self.assertEqual(len(ep["certificate"]), ep["d_star"])
        self.assertTrue(ep["lower_bound_complete"])
        self.assertEqual(sum(ep["turns"], []), ep["certificate"])
        self.assertTrue(all(len(t) <= I.MAX_ACTIONS_PER_ASSISTANT_TURN for t in ep["turns"]))
        lo, hi = ep["band"]
        self.assertTrue(lo <= ep["d_star"] <= hi)
        # observability rule: >= 1 target identifiable initially, both at the terminal
        targets = I.item_targets(scene, item)
        n_init = sum(bool(self.obs_fn(scene, init, t)["identifiable"]) for t in targets)
        self.assertGreaterEqual(n_init, 1)
        self.assertEqual(ep["initial_observability"]["n_identifiable"], n_init)
        self.assertTrue(all(self.obs_fn(scene, pose, t)["identifiable"] for t in targets))

    def test_episodes_are_valid_diverse_and_span_bands(self):
        bands = set()
        n = 0
        for e, eps, funnel in self.results:
            counts = funnel.to_dict()["init_candidates"]
            self.assertEqual(counts.get("accepted", 0), len(eps))
            for ep in eps:
                self._check_episode(e["item"], ep)
                bands.add(ep["band_name"])
                n += 1
            for i in range(len(eps)):
                for j in range(i + 1, len(eps)):
                    a, b = np.asarray(eps[i]["init_c2w"]), np.asarray(eps[j]["init_c2w"])
                    close = math.hypot(a[0, 3] - b[0, 3], a[1, 3] - b[1, 3]) < 0.6
                    dyaw = abs((math.degrees(C.yaw_of(a) - C.yaw_of(b)) + 180.0) % 360.0 - 180.0)
                    self.assertFalse(close and dyaw < 30.0)
        self.assertGreaterEqual(len(bands), 3, bands)
        self.assertGreaterEqual(n, 10)

    def test_episodes_for_scene_allocates_and_reports(self):
        eps, funnel, report = I.episodes_for_scene(self.scene, self.entries, target_per_scene=12, seed=7)
        self.assertLessEqual(len(eps), 12)
        self.assertGreaterEqual(len(eps), 8)
        self.assertEqual(sum(report["band_counts"].values()), len(eps))
        self.assertEqual(report["missing_total"], 12 - len(eps))
        self.assertEqual(sum(report["band_targets"].values()), 12)
        self.assertGreaterEqual(report["band_targets"]["4-6"] + report["band_targets"]["7-8"], 7)
        for b, info in report["shortfall"].items():
            self.assertGreater(info["missing"], 0)
            self.assertTrue(info["item_stop_reasons"])
        by_key = {k: e["item"] for k, e in enumerate(self.entries)}
        for ep in eps:
            self._check_episode(by_key[ep["item_key"]], ep)
        self.assertIn("init_candidates", funnel.to_dict())


if __name__ == "__main__":
    unittest.main()
