"""Tests for the optional allocation features (reserve pool, family quotas), the
horizon-locked motion model, and that all of them are off by default."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import difficulty as D
from data_gen.interiorgs_v10 import initial_states as I
from data_gen.interiorgs_v10.contract import stable_hash
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.sampling import projective_item
from data_gen.interiorgs_v10.tests.test_difficulty import scan_endpoints, two_rooms_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

# generate_scene(two_rooms, RunConfig(target_tasks_per_scene=6)) recorded with the code of commit
# dd34f565 (before reserve / family quotas / motion models existed), numpy 2.2.4, Python 3.12.13.
GOLDEN_ROWS = "ac111143b9122bdc3ae98889785ab3e29a6caf291cb8e408d6a4c1d5af41b87b"
GOLDEN_FUNNEL = "71b006d302e5107e642c404d83e5c3f195c5a7a0ee3380c0e14b6d5e3a2f20cc"
GOLDEN_NUMPY = "2.2."


def _sig(eps):
    return json.dumps([[ep["item_key"], np.asarray(ep["init_c2w"]).round(9).tolist(), ep["d_star"],
                        ep["certificate"]] for ep in eps])


class AllocationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = two_rooms_scene()
        o = {x.id: x for x in cls.scene.objects}
        specs = [("2", "4", "left", "fam_a"), ("5", "1", "right", "fam_b"), ("10", "13", "left", "fam_c")]
        cls.entries = []
        for a, b, rel, fam in specs:
            item = projective_item("two_rooms", o[a], o[b], rel)
            eps = scan_endpoints(cls.scene, item)
            fake = copy.deepcopy(item)
            fake["task_type"] = fam  # the allocator only reads task_type to find the family
            cls.entries.append({"item": fake, "endpoints": eps, "success_fn": D.make_success_fn(item)})

    def _run(self, target, **kw):
        return I.episodes_for_scene(self.scene, self.entries, target_per_scene=target, seed=3, **kw)

    def test_defaults_equal_explicit_off(self):
        a, fa, _ = self._run(8)
        b, fb, _ = self._run(8, family_quotas=None, reserve_k=0, terminal_identifiable="all")
        self.assertEqual(_sig(a), _sig(b))
        self.assertEqual(fa.to_dict(), fb.to_dict())

    def test_parse_family_quotas(self):
        self.assertIsNone(I.parse_family_quotas(None))
        self.assertIsNone(I.parse_family_quotas(""))
        self.assertEqual(I.parse_family_quotas("a:1:3,b:2,c:0:*,d::4"),
                         {"a": (1, 3), "b": (2, None), "c": (0, None), "d": (0, 4)})
        self.assertEqual(I.parse_family_quotas({"a": (1, 2), "b": {"min": 2}}), {"a": (1, 2), "b": (2, None)})
        with self.assertRaises(ValueError):
            I.parse_family_quotas("a:3:2")
        with self.assertRaises(ValueError):
            self._run(4, family_quotas="fam_a:3,fam_b:2")  # minimums exceed the target

    def test_min_is_served_before_round_robin(self):
        fam_of = {k: e["item"]["task_type"] for k, e in enumerate(self.entries)}
        base, _, _ = self._run(6)
        n_c = sum(fam_of[ep["item_key"]] == "fam_c" for ep in base)
        alone, _, _ = I.episodes_for_scene(self.scene, self.entries[2:], target_per_scene=6, seed=3,
                                           max_per_item=4)  # what fam_c's item can supply by itself
        want = min(3, len(alone))
        self.assertGreater(want, n_c)  # the fixture must make the minimum bind
        eps, _, report = self._run(6, family_quotas={"fam_c": (3, None)})
        got = sum(fam_of[ep["item_key"]] == "fam_c" for ep in eps)
        self.assertGreaterEqual(got, want)  # the minimum overrides the fixed fill order
        self.assertEqual(report["family_counts"].get("fam_c", 0), got)

    def test_max_is_respected(self):
        eps, _, report = self._run(10, family_quotas="fam_a:0:1,fam_b:0:2")
        fam_of = {k: e["item"]["task_type"] for k, e in enumerate(self.entries)}
        counts = {}
        for ep in eps:
            counts[fam_of[ep["item_key"]]] = counts.get(fam_of[ep["item_key"]], 0) + 1
        self.assertLessEqual(counts.get("fam_a", 0), 1)
        self.assertLessEqual(counts.get("fam_b", 0), 2)
        self.assertEqual(report["family_counts"], dict(sorted(counts.items())))

    def test_soft_max_fills_only_when_other_families_cannot(self):
        fam_of = {k: e["item"]["task_type"] for k, e in enumerate(self.entries)}
        alone, _, _ = I.episodes_for_scene(self.scene, self.entries[:1], target_per_scene=6, seed=3)
        # one family only, max 2: the hard max leaves the scene short, the soft max fills it
        hard, _, _ = I.episodes_for_scene(self.scene, self.entries[:1], target_per_scene=6, seed=3,
                                          family_quotas="fam_a:0:2")
        soft, f, rep = I.episodes_for_scene(self.scene, self.entries[:1], target_per_scene=6, seed=3,
                                            family_quotas="fam_a:0:2", soft_max=True)
        self.assertLessEqual(len(hard), 2)
        self.assertEqual(len(soft), len(alone))
        self.assertGreater(len(soft), len(hard))
        self.assertEqual(f.to_dict()["init_scene_allocation"].get("family_soft_max_extra"), len(soft) - len(hard))
        self.assertLessEqual(len(soft), 6)
        # several families: when the others fill the target, the max still holds
        hard3, _, _ = self._run(6, family_quotas="fam_a:0:1")
        soft3, _, _ = self._run(6, family_quotas="fam_a:0:1", soft_max=True)
        if len(hard3) == 6:
            self.assertEqual(_sig(hard3), _sig(soft3))
            self.assertLessEqual(sum(fam_of[e["item_key"]] == "fam_a" for e in soft3), 1)
        # quotas off: soft_max changes nothing
        a, _, _ = self._run(6)
        b, _, _ = self._run(6, soft_max=True)
        self.assertEqual(_sig(a), _sig(b))

    def test_reserve_is_disjoint_banded_and_keeps_main(self):
        main0, _, _ = self._run(6)
        main, _, report = self._run(6, reserve_k=6)
        self.assertEqual(_sig(main), _sig(main0))  # the reserve pass never changes the main set
        res = report["reserve"]
        self.assertLessEqual(len(res), 6)
        self.assertGreater(len(res), 0)
        main_ids = {id(e) for e in main}
        self.assertFalse(any(id(e) in main_ids for e in res))
        targets = report["reserve_band_targets"]
        self.assertEqual(sum(targets.values()), 6)
        for name, n in report["reserve_band_counts"].items():
            self.assertLessEqual(n, targets[name])
        for ep in res:
            self.assertTrue(ep.get("reserve"))
            lo, hi = ep["band"]
            self.assertTrue(lo <= ep["d_star"] <= hi)
        # diversity holds between reserve and main episodes of the same item
        for r in res:
            for m in main:
                if r["item_key"] != m["item_key"]:
                    continue
                a, b = np.asarray(r["init_c2w"]), np.asarray(m["init_c2w"])
                close = math.hypot(a[0, 3] - b[0, 3], a[1, 3] - b[1, 3]) < 0.6
                dyaw = abs((math.degrees(C.yaw_of(a) - C.yaw_of(b)) + 180.0) % 360.0 - 180.0)
                self.assertFalse(close and dyaw < 30.0)

    def test_reserve_from_surplus_interleaves_d_star_within_band(self):
        class _Fake:
            item = {"task_type": "fam_a"}

            def __init__(self, surplus):
                self.surplus = surplus

            def fill(self, needs, max_new=None):
                return []

        def run(ds):
            surplus = [{"band": [2, 3], "band_name": "2-3", "d_star": d, "item_key": 0} for d in ds]
            funnel = I.Funnel()
            res, rep = I._fill_reserve([_Fake(surplus)], [], 14, 14, list(I.DEFAULT_BANDS),
                                       I.DEFAULT_BAND_WEIGHTS, None, funnel, 0.0, None, False)
            return res, rep, funnel.to_dict()["init_scene_allocation"]

        res, rep, f = run((3, 3, 3, 3, 3, 2, 2, 2, 2, 2))
        self.assertEqual(rep["reserve_band_targets"]["2-3"], 3)
        self.assertEqual(sorted(ep["d_star"] for ep in res), [2, 3, 3])  # not 3, 3, 3
        # one episode per d*: the per-d* turn counter must not clip the pass or the accounting
        res, rep, f = run((3, 2))
        self.assertEqual(sorted(ep["d_star"] for ep in res), [2, 3])
        self.assertEqual(rep["reserve_target"], 14)
        self.assertEqual(f["reserve_from_surplus"], 2)
        self.assertEqual(f["reserve_shortfall"], 12)

    def test_reserve_family_caps_scale_with_k(self):
        self.assertEqual(I._scaled_quotas({"a": (2, 4), "b": (1, None)}, 7, 14), {"a": (1, 2), "b": (0, None)})
        _, _, report = self._run(6, family_quotas="fam_a:0:2", reserve_k=6)
        self.assertLessEqual(report["reserve_family_counts"].get("fam_a", 0), 2)

    def test_family_terminal_roles(self):
        from data_gen.interiorgs_v10.families import absolute_positioning

        self.assertIsNone(I._family_terminal_roles(self.entries[0]["item"]))
        self.assertEqual(I._family_terminal_roles({"task_type": "occlusion_alignment"}), (1,))
        self.assertEqual(I._family_terminal_roles({"task_type": "absolute_positioning"}),
                         tuple(absolute_positioning.TERMINAL_IDENTIFIABLE_ROLES))


class DefaultOutputTest(unittest.TestCase):
    """generate_scene with every new option off == the pre-feature generator."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = write_synthetic_scenes(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_default_matches_golden_and_reserve_keeps_main_rows(self):
        default = _jsonable(generate_scene(self.paths["two_rooms"], "two_rooms", RunConfig(target_tasks_per_scene=6)))
        self.assertEqual(set(default), {"rows", "funnel", "summary", "timings"})
        self.assertNotIn("n_reserve", default["summary"])
        if np.__version__.startswith(GOLDEN_NUMPY):
            self.assertEqual(stable_hash(default["rows"]), GOLDEN_ROWS)
            self.assertEqual(stable_hash(default["funnel"]), GOLDEN_FUNNEL)
        with_reserve = _jsonable(generate_scene(self.paths["two_rooms"], "two_rooms",
                                                RunConfig(target_tasks_per_scene=6, reserve_k=3)))
        self.assertEqual(stable_hash(with_reserve["rows"]), stable_hash(default["rows"]))
        res = with_reserve["reserve_rows"]
        self.assertTrue(all(r["reserve"] for r in res))
        self.assertFalse({r["task_id"] for r in res} & {r["task_id"] for r in default["rows"]})
        self.assertEqual(with_reserve["summary"]["n_reserve"], len(res))
        self.assertEqual(C.motion_model(), ("r1", 0.0))


class MotionRowTest(unittest.TestCase):
    def tearDown(self):
        C.set_motion_model("r1", 0.0)

    def test_rows_from_other_models_are_marked(self):
        from data_gen.interiorgs_v10.run_scene import _task_row

        scene = two_rooms_scene()
        o = {x.id: x for x in scene.objects}
        item = projective_item("two_rooms", o["2"], o["4"], "left")
        ep = {"init_c2w": C.pose_from_xy_yaw(2.0, 2.0, 0.0).tolist(), "difficulty": {"d_star": 1}}
        flat = _task_row(scene, item, ep)
        self.assertNotIn("motion_model", flat)
        self.assertEqual(flat["action_protocol_version"], C.ACTION_PROTOCOL_VERSION)
        C.set_motion_model("horizon_locked", -15.0)
        ep = {"init_c2w": C.pose_from_xy_yaw(2.0, 2.0, 0.0).tolist(), "difficulty": {"d_star": 1}}
        row = _task_row(scene, item, ep)
        self.assertEqual(row["motion_model"], {"model": "horizon_locked", "pitch_deg": -15.0, "env_supported": False})
        self.assertNotEqual(row["action_protocol_version"], C.ACTION_PROTOCOL_VERSION)
        self.assertNotEqual(row["task_id"], flat["task_id"])


if __name__ == "__main__":
    unittest.main()
