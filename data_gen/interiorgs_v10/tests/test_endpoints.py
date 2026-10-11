"""Tests for workstream E: occlusion.py and endpoints.py (stdlib unittest)."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import endpoints as E
from data_gen.interiorgs_v10 import occlusion as O
from data_gen.interiorgs_v10 import sampling
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import _box, _write, write_synthetic_scenes


def _occlusion_scene(root: Path, with_tall_box: bool = True, tall_label: str = "wardrobe") -> Path:
    """Two closed rooms (no door): A [0,4]x[0,4], B [4,8]x[0,4]."""
    labels = [
        _box("1", "cabinet", 3.3, 1.7, 0.0, 3.7, 2.3, 0.5),        # target in room A
        _box("2", "armchair", 6.0, 1.6, 0.0, 6.8, 2.4, 0.9),       # target in room B (behind the wall)
        _box("3", "sofa", 0.6, 3.2, 0.0, 2.0, 3.7, 0.85),
    ]
    if with_tall_box:
        labels.append(_box("4", tall_label, 2.2, 1.4, 0.0, 2.6, 2.6, 2.2))
    rooms = [("living room", [(0, 0), (4, 0), (4, 4), (0, 4)]),
             ("bedroom", [(4, 0), (8, 0), (8, 4), (4, 4)])]
    name = f"synth_occlusion_{int(with_tall_box)}_{tall_label.replace(' ', '_')}"
    return _write(root / name, labels, rooms, [], y_sign=-1.0, truth={})


class OcclusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.scene_box = load_scene(_occlusion_scene(root, True))
        cls.scene_free = load_scene(_occlusion_scene(root, False))
        cls.scene_curtain = load_scene(_occlusion_scene(root, True, "curtain"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    @staticmethod
    def _obj(scene, oid):
        return next(o for o in scene.objects if o.id == oid)

    def test_hidden_behind_wall(self):
        s = self.scene_free
        self.assertGreater(len(s.detector.wall_segments), 0)
        cam = C.pose_from_xy_yaw(1.0, 2.0, 0.0)  # room A looking +x at the armchair in room B
        target = self._obj(s, "2")
        self.assertTrue(O.identifiable(s, cam, target)["inside_frame_fraction"] > 0.5)  # projects fine
        self.assertLess(O.visible_fraction(s, cam, target), 0.05)
        self.assertFalse(O.identifiable(s, cam, target)["identifiable"])

    def test_hidden_behind_tall_box(self):
        cam = C.pose_from_xy_yaw(0.8, 2.0, 0.0)
        vf_box = O.visible_fraction(self.scene_box, cam, self._obj(self.scene_box, "1"))
        vf_free = O.visible_fraction(self.scene_free, cam, self._obj(self.scene_free, "1"))
        self.assertLess(vf_box, 0.2)
        self.assertEqual(vf_free, 1.0)

    def test_non_occluding_label_is_ignored(self):
        cam = C.pose_from_xy_yaw(0.8, 2.0, 0.0)
        self.assertTrue(O.is_non_occluding("curtain"))
        self.assertTrue(O.is_non_occluding("Living Room Rug"))
        self.assertTrue(O.is_non_occluding("ceiling light"))
        self.assertFalse(O.is_non_occluding("floor lamp"))
        self.assertEqual(O.visible_fraction(self.scene_curtain, cam, self._obj(self.scene_curtain, "1")), 1.0)

    def test_out_of_frustum_is_zero_and_identifiable_schema(self):
        s = self.scene_free
        cam = C.pose_from_xy_yaw(1.0, 2.0, math.pi)  # facing away from the cabinet
        self.assertEqual(O.visible_fraction(s, cam, self._obj(s, "1")), 0.0)
        res = O.identifiable(s, C.pose_from_xy_yaw(0.8, 2.0, 0.0), self._obj(s, "1"))
        for key in ("inside_frame_fraction", "area_ratio", "visible_fraction", "identifiable"):
            self.assertIn(key, res)
        self.assertTrue(res["identifiable"])
        # dict entries (task rows) are accepted as well as Obj
        entry = self._obj(s, "1").target_object_entry()
        self.assertEqual(O.visible_fraction(s, C.pose_from_xy_yaw(0.8, 2.0, 0.0), entry), 1.0)


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        paths = write_synthetic_scenes(Path(cls._tmp.name))
        cls.scenes = {k: load_scene(p) for k, p in paths.items()}
        cls.two = cls.scenes["two_rooms"]
        cls.objs = {o.id: o for o in cls.two.objects}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _items(self):
        o = self.objs
        return [
            sampling.projective_item(self.two.scene_id, o["4"], o["5"], "left"),   # tea table / dining table
            sampling.projective_item(self.two.scene_id, o["4"], o["5"], "right"),
            sampling.projective_item(self.two.scene_id, o["6"], o["7"], "left"),   # duplicate chairs
            sampling.fov_item(self.two.scene_id, o["1"], o["4"]),                  # sofa + tea table
            sampling.projective_item(self.two.scene_id, o["10"], o["13"], "right"),  # bed / wardrobe
            sampling.fov_item(self.two.scene_id, o["11"], o["14"]),
        ]

    def test_vectorised_scorer_matches_canonical(self):
        rng = np.random.default_rng(0)
        for item in self._items()[:4]:
            n = 150
            xs, ys = rng.uniform(0.3, 5.7, n), rng.uniform(0.3, 4.7, n)
            p = item["target_region"]["params"]
            mid = (np.asarray(p["object_a_center"]) + np.asarray(p["object_b_center"])) / 2
            yaws = np.arctan2(mid[1] - ys, mid[0] - xs) + rng.normal(0, 0.4, n)
            ev = E.batch_evaluate(self.two, item, xs, ys, yaws)
            names = ev["gate_names"][:-1]
            for i in range(n):
                exact = sampling.score(item, C.pose_from_xy_yaw(xs[i], ys[i], yaws[i]))
                self.assertEqual(list(ev["gates"][i, :-1]), [exact["gates"][k] for k in names])
                self.assertAlmostEqual(float(ev["shaping_score"][i]), float(exact["shaping_score"]), places=9)

    def test_occlusion_memo_reset_keeps_results(self):
        # regression (real scene 0463_840754): the memo used to be cleared before this call's rows were read
        item = self._items()[0]
        rng = np.random.default_rng(1)
        xs, ys, yaws = rng.uniform(0.3, 5.7, 40), rng.uniform(0.3, 4.7, 40), rng.uniform(-math.pi, math.pi, 40)
        ref = E.batch_evaluate(self.two, item, xs, ys, yaws)
        old = E._OCCL_MEMO_MAX
        E._OCCL_MEMO_MAX = 5
        try:
            E._cache(self.two)["occl"].clear()
            for _ in range(2):  # first call overflows and resets, second refills from scratch
                ev = E.batch_evaluate(self.two, item, xs, ys, yaws)
                np.testing.assert_array_equal(ev["gates"], ref["gates"])
        finally:
            E._OCCL_MEMO_MAX = old

    def test_every_endpoint_rescores_as_success(self):
        found = 0
        for item in self._items():
            eps, funnel = E.sample_endpoints(self.two, item, mode="v10", max_endpoints=4)
            self.assertLessEqual(len(eps), 4)
            for ep in eps:
                c2w = np.asarray(ep["c2w"])
                self.assertTrue(sampling.score(item, c2w)["success"])
                self.assertTrue(self.two.is_pose_legal(c2w)[0])
                self.assertAlmostEqual(float(c2w[2, 2]), 0.0, places=9)
                self.assertAlmostEqual(float(c2w[2, 3]), C.CAMERA_HEIGHT_M, places=9)
                for t in item["target_object"]["objects"]:
                    self.assertGreaterEqual(O.visible_fraction(self.two, c2w, t), 0.5)
                self.assertTrue(E.success_fn(item)(c2w))
            found += bool(eps)
            # funnel accounting: every scored pose has exactly one first-fail entry
            counts = funnel.to_dict()
            scored = counts["endpoint_count"]["poses_scored"]
            self.assertEqual(sum(counts["endpoint_pose_first_fail"].values()), scored)
            self.assertEqual(sum(counts["endpoint_item"].values()), 1)
        self.assertGreaterEqual(found, 3)

    def test_endpoints_are_diverse(self):
        item = sampling.fov_item(self.two.scene_id, self.objs["1"], self.objs["4"])
        eps, _ = E.sample_endpoints(self.two, item, mode="v10", max_endpoints=6)
        self.assertGreaterEqual(len(eps), 2)
        keys = {tuple(ep["state_key"]) for ep in eps}
        self.assertEqual(len(keys), len(eps))
        xy = np.array([ep["position"][:2] for ep in eps])
        d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
        self.assertGreater(d.max(), 0.5)

    def test_v9_like_budget_and_yaw(self):
        item = self._items()[0]
        _, funnel = E.sample_endpoints(self.two, item, mode="v9_like", max_endpoints=1)
        counts = funnel.to_dict()
        self.assertLessEqual(counts["endpoint_position"]["legal"], 64)
        self.assertLessEqual(counts["endpoint_count"]["poses_scored"], 64 * 5)

    def test_success_fn_cache_batch_and_exact(self):
        item = self._items()[3]
        eps, _ = E.sample_endpoints(self.two, item, mode="v10", max_endpoints=2)
        self.assertTrue(eps)
        fn = E.success_fn(item, self.two)
        self.assertIs(fn, E.success_fn(item))  # resolved via the scene registry, cached
        c2w = np.asarray(eps[0]["c2w"])
        self.assertTrue(fn(c2w))
        self.assertIn(C.state_key(c2w), fn.cache)
        rng = np.random.default_rng(3)
        poses = [np.asarray(eps[0]["c2w"])] + [
            C.pose_from_xy_yaw(rng.uniform(0.5, 5.5), rng.uniform(0.5, 4.5), rng.uniform(-3, 3)) for _ in range(40)]
        batch = fn.batch(np.stack(poses))
        self.assertEqual(list(batch), [fn.exact(p) for p in poses])

    def test_v10_yield_ge_v9_like_on_synthetic(self):
        v9 = v10 = 0
        for name in ("two_rooms", "dense_duplicates"):
            scene = self.scenes[name]
            items = E.comparison_items(scene)[::5]
            for item in items:
                v9 += bool(E.sample_endpoints(scene, item, mode="v9_like", max_endpoints=1)[0])
                v10 += bool(E.sample_endpoints(scene, item, mode="v10", max_endpoints=1)[0])
        self.assertGreaterEqual(v10, v9)
        self.assertGreater(v10, 0)

    def test_prefilters_do_not_lose_items(self):
        """Analytic prefilters (relation side, distance band, yaw interval) are safe."""
        items = E.comparison_items(self.two)[::7]
        for item in items:
            with_pf = bool(E.sample_endpoints(self.two, item, mode="v10", max_endpoints=1)[0])
            audit = bool(E.sample_endpoints(self.two, item, mode="audit", max_endpoints=1)[0])
            if audit:
                self.assertTrue(with_pf, msg=str(item["task_description"]))


if __name__ == "__main__":
    unittest.main()
