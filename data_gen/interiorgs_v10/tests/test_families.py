"""Tests for workstream F (new task families + canonical v2 judges).

Run: cd <repo> && python3 -m unittest discover -s data_gen/interiorgs_v10/tests -t .
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as K
from data_gen.interiorgs_v10.families import canonical_v2 as V
from data_gen.interiorgs_v10.families import common as C
from data_gen.interiorgs_v10.families import judges_v2 as J
from data_gen.interiorgs_v10.families import make_patch
from data_gen.interiorgs_v10.sampling import fov_item, projective_item
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing import synthetic_scene as S

REPO = Path(__file__).resolve().parents[3]


def _occlusion_scene(root: Path) -> Path:
    """Living room with tall solid occluders and objects behind them."""
    labels = [
        S._box("1", "refrigerator", 2.7, 2.2, 0.0, 3.3, 2.8, 1.9),
        S._box("2", "armchair", 2.6, 3.6, 0.0, 3.4, 4.3, 0.9),
        S._box("3", "wardrobe", 4.5, 0.8, 0.0, 5.1, 2.4, 2.2),
        S._box("4", "floor lamp", 5.4, 1.4, 0.0, 5.7, 1.7, 1.6),
        S._box("5", "cabinet", 0.6, 0.2, 0.0, 1.8, 0.6, 0.9),
    ] + S._walls_for_rect(100, 0.0, 0.0, 6.0, 5.0)
    return S._write(root / "synth_occlusion", labels, [("living room", [(0, 0), (6, 0), (6, 5), (0, 5)])], [],
                    y_sign=-1.0, truth={})


def _jsonable(metric):
    return json.loads(json.dumps(metric, sort_keys=True, allow_nan=False))


def _random_poses(scene, n, rng, margin=0.3):
    lo = np.min([r.polygon.min(axis=0) for r in scene.rooms], axis=0) + margin
    hi = np.max([r.polygon.max(axis=0) for r in scene.rooms], axis=0) - margin
    xy = rng.uniform(lo, hi, size=(n, 2))
    yaw = rng.uniform(-math.pi, math.pi, size=n)
    return [K.pose_from_xy_yaw(float(x), float(y), float(t)) for (x, y), t in zip(xy, yaw)]


class FamiliesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        paths = S.write_synthetic_scenes(root)
        paths["occlusion"] = _occlusion_scene(root)
        cls.scenes = {name: load_scene(path, name) for name, path in paths.items()}
        # (family, scene) -> list of (item, candidates); built once, reused.
        cls.cases = {}
        plan = {
            "absolute_positioning": ("two_rooms", "dense_duplicates"),
            "equidistance": ("two_rooms", "dense_duplicates"),
            "occlusion_alignment": ("occlusion",),
            "screen_occupancy": ("two_rooms", "dense_duplicates"),
            "size_distance_invariance": ("two_rooms", "dense_duplicates"),
            "centering": ("two_rooms", "dense_duplicates"),
        }
        for task_type, scene_names in plan.items():
            fam = V.FAMILIES[task_type]
            for sname in scene_names:
                scene = cls.scenes[sname]
                objs = [o for o in scene.objects if not C.is_structural(o)]
                found = []
                for tup in C.enumerate_tuples(fam, objs, seed=0, max_tuples=6, ordered=not fam.SYMMETRIC):
                    for params in fam.param_menu(tup)[:2]:
                        item = fam.build_item(sname, tup, params)
                        cands = fam.success_pose_candidates(scene, item, seed=1, max_candidates=4)
                        if cands:
                            found.append((item, cands))
                    if len(found) >= 3:
                        break
                cls.cases[(task_type, sname)] = found

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _family_cases(self, task_type):
        out = []
        for (tt, _), cases in self.cases.items():
            if tt == task_type:
                out.extend(cases)
        return out

    # ------------------------------------------------------------ v1 identity
    def test_v1_projective_fov_bit_identical_through_v2_dispatcher(self):
        from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task

        scene = self.scenes["two_rooms"]
        by_id = {o.id: o for o in scene.objects}
        pairs = [("4", "2"), ("6", "7"), ("1", "3"), ("10", "13")]
        rng = np.random.default_rng(7)
        poses = _random_poses(scene, 120, rng)
        n_success = 0
        for a, b in pairs:
            for v1_item in (projective_item("two_rooms", by_id[a], by_id[b], "left"),
                            projective_item("two_rooms", by_id[a], by_id[b], "right"),
                            fov_item("two_rooms", by_id[a], by_id[b])):
                v2_item = dict(v1_item, canonical_task_metric_version=K.CANONICAL_TASK_METRIC_V2)
                for pose in poses:
                    ref = score_canonical_task(v1_item, pose)
                    got = V.score_canonical_task_v2(v2_item, pose)
                    via_v1 = V.score_canonical_task_v2(v1_item, pose)
                    self.assertEqual(json.dumps(ref, sort_keys=True), json.dumps(got, sort_keys=True))
                    self.assertEqual(json.dumps(ref, sort_keys=True), json.dumps(via_v1, sort_keys=True))
                    n_success += bool(ref["success"])
        self.assertGreater(n_success, 0)

    def test_uses_canonical_backend_v2_semantics(self):
        self.assertFalse(V.uses_canonical_backend_v2({"task_type": "equidistance"}))
        self.assertTrue(V.uses_canonical_backend_v2(
            {"task_type": "equidistance", "canonical_task_metric_version": K.CANONICAL_TASK_METRIC_V2}))
        self.assertTrue(V.uses_canonical_backend_v2(
            {"task_type": "fov_inclusion", "canonical_task_metric_version": K.CANONICAL_TASK_METRIC_V2}))
        with self.assertRaises(ValueError):
            V.uses_canonical_backend_v2(
                {"task_type": "equidistance", "canonical_task_metric_version": K.CANONICAL_TASK_METRIC_V1})
        with self.assertRaises(ValueError):
            V.uses_canonical_backend_v2({"task_type": "equidistance", "canonical_task_metric_version": "x"})
        self.assertEqual(V.SUPPORTED_TASK_TYPES_V2, V.V1_TASK_TYPES | J.NEW_TASK_TYPES)

    # ------------------------------------------------------------ registry
    def test_registry_interface_and_items(self):
        required = ("TASK_TYPE", "REGION_TYPE", "N_OBJECTS", "ROLE_NAMES", "TERMINAL_IDENTIFIABLE_ROLES",
                    "PARAM_MENU", "GATES_DOC", "tuple_ok", "param_menu", "build_item", "describe", "judge",
                    "sampler", "success_pose_candidates", "anchor_xy", "necessary_mask")
        self.assertEqual(frozenset(V.FAMILIES), J.NEW_TASK_TYPES)
        for task_type, fam in V.FAMILIES.items():
            for attr in required:
                self.assertTrue(hasattr(fam, attr), f"{task_type} lacks {attr}")
            self.assertEqual(fam.TASK_TYPE, task_type)
            self.assertEqual(len(fam.ROLE_NAMES), fam.N_OBJECTS)
            self.assertEqual(len(J.ROLES[task_type]), fam.N_OBJECTS)
        for task_type in V.FAMILIES:
            cases = self._family_cases(task_type)
            self.assertTrue(cases, task_type)
            item = cases[0][0]
            json.dumps(item, allow_nan=False)
            for key in ("scene_id", "task_type", "task_description", "target_object", "target_region",
                        "init_camera", "canonical_task_metric_version", "camera_model_version",
                        "action_protocol_version", "generator_version"):
                self.assertIn(key, item)
            self.assertEqual(item["canonical_task_metric_version"], K.CANONICAL_TASK_METRIC_V2)
            self.assertEqual(len(item["target_object"]["objects"]), V.FAMILIES[task_type].N_OBJECTS)
            for entry in item["target_object"]["objects"]:
                self.assertIn(f"the {entry['label']}", item["task_description"])
            self.assertEqual(item["task_description"], V.FAMILIES[task_type].describe(item))

    def test_descriptions_use_referring_expressions(self):
        scene = self.scenes["two_rooms"]
        obj = next(o for o in scene.objects if o.id == "6")
        obj = K.Obj(**{**obj.__dict__, "ref": "the dining chair closer to the sofa"})
        item = V.FAMILIES["absolute_positioning"].build_item("two_rooms", (obj,), {"distance_m": 2.0})
        self.assertIn("the dining chair closer to the sofa", item["task_description"])
        self.assertIn("2.0 meters", item["task_description"])

    # ------------------------------------------------------------ success regions
    def test_every_family_has_success_regions(self):
        for task_type in V.FAMILIES:
            cases = self._family_cases(task_type)
            self.assertGreaterEqual(len(cases), 2 if task_type != "occlusion_alignment" else 2,
                                    f"{task_type}: too few feasible tuples")
            for item, cands in cases:
                self.assertTrue(cands)
                for cand in cands:
                    metric = V.score_canonical_task_v2(item, cand["c2w"])
                    self.assertTrue(metric["success"], task_type)
                    self.assertGreaterEqual(metric["shaping_score"], 0.5)
                    ok, _ = self.scenes[item["scene_id"]].is_pose_legal(cand["c2w"])
                    self.assertTrue(ok)
                # at least one candidate is not an isolated point of the lattice
                self.assertGreaterEqual(max(c["lattice_support"] for c in cands), 1, task_type)

    def test_success_region_reachable_on_action_lattice(self):
        # One item per family: the region contains many 0.3 m / 20 deg lattice poses.
        for task_type in ("absolute_positioning", "equidistance", "occlusion_alignment"):
            fam = V.FAMILIES[task_type]
            item, _ = self._family_cases(task_type)[0]
            scene = self.scenes[item["scene_id"]]
            stats = C.region_stats(scene, item, fam, spacing=0.15, yaw_step_deg=10.0, neighbour_samples=10)
            self.assertGreater(stats["expected_lattice_successes"], 3.0, (task_type, stats))
            self.assertGreaterEqual(stats["sublattice_hit_fraction"], 0.5, (task_type, stats))
            self.assertLess(stats["success_pose_fraction"], 0.25, (task_type, stats))

    # ------------------------------------------------------------ shaping
    def test_shaping_upper_half_reserved_for_success(self):
        rng = np.random.default_rng(3)
        for task_type in V.FAMILIES:
            n_succ = n_fail = 0
            for item, cands in self._family_cases(task_type)[:2]:
                scene = self.scenes[item["scene_id"]]
                poses = _random_poses(scene, 120, rng)
                for cand in cands:
                    poses.append(cand["c2w"])
                    poses.extend(K.apply_action(cand["c2w"], a) for a in K.ACTIONS)
                for pose in poses:
                    m = V.score_canonical_task_v2(item, pose)
                    s = m["shaping_score"]
                    self.assertTrue(0.0 <= s <= 1.0)
                    self.assertEqual(m["score"], s)
                    self.assertEqual(m["success"], all(m["gates"].values()))
                    if m["success"]:
                        n_succ += 1
                        self.assertGreaterEqual(s, 0.5, task_type)
                    else:
                        n_fail += 1
                        self.assertLess(s, 0.5, task_type)
                    _jsonable(m)
            self.assertGreater(n_succ, 0, task_type)
            self.assertGreater(n_fail, 0, task_type)

    def test_shaping_increases_toward_distance_band(self):
        scene = self.scenes["two_rooms"]
        tv_cabinet = next(o for o in scene.objects if o.id == "2")
        item = V.FAMILIES["absolute_positioning"].build_item("two_rooms", (tv_cabinet,), {"distance_m": 2.5})
        # walk from far (4.9 m) to the band along the facing ray: progress is monotone
        scores = []
        for d in (4.9, 4.3, 3.7, 3.1, 2.5):
            x = float(tv_cabinet.aabb_min[0]) - d
            m = V.score_canonical_task_v2(item, K.pose_from_xy_yaw(x, float(tv_cabinet.center[1]), 0.0))
            scores.append(m["shaping_score"])
        self.assertEqual(scores, sorted(scores))
        self.assertGreaterEqual(scores[-1], 0.5)

    # ------------------------------------------------------------ gate specifics
    def test_absolute_positioning_tolerance_band(self):
        scene = self.scenes["two_rooms"]
        tv = next(o for o in scene.objects if o.id == "3")
        fam = V.FAMILIES["absolute_positioning"]
        item = fam.build_item("two_rooms", (tv,), {"distance_m": 2.0})
        y = float(tv.center[1])
        x_face = float(tv.aabb_min[0])
        for d, expect in ((2.0, True), (2.24, True), (1.76, True), (2.3, False), (1.7, False)):
            m = fam.judge(item, K.pose_from_xy_yaw(x_face - d, y, 0.0))
            self.assertEqual(m["success"], expect, (d, m["distance_m"], m["gates"]))
        # facing away: distance gate passes, visibility fails
        m = fam.judge(item, K.pose_from_xy_yaw(x_face - 2.0, y, math.pi))
        self.assertTrue(m["gates"]["distance"])
        self.assertFalse(m["success"])

    def test_occlusion_requires_target_in_view_direction(self):
        item, cands = self._family_cases("occlusion_alignment")[0]
        pose = cands[0]["c2w"]
        self.assertTrue(V.score_canonical_task_v2(item, pose)["success"])
        # turning far away keeps the geometry but the target leaves the frame
        away = pose
        for _ in range(5):
            away = K.apply_action(away, "turn_left")
        m = V.score_canonical_task_v2(item, away)
        self.assertFalse(m["success"])
        self.assertFalse(m["gates"]["target_in_view_direction"] and m["gates"]["occluder_inside_frame"])

    def test_occlusion_episodes_end_hidden_and_in_frame(self):
        from data_gen.interiorgs_v10 import initial_states as IS
        from data_gen.interiorgs_v10 import occlusion as O

        item, cands = self._family_cases("occlusion_alignment")[0]
        scene = self.scenes[item["scene_id"]]
        fam = V.FAMILIES["occlusion_alignment"]
        entry = {"item": item, "endpoints": [c["c2w"] for c in cands],
                 "success_fn": lambda c2w: bool(fam.judge(item, c2w)["success"])}
        eps_all, _, _ = IS.episodes_for_scene(scene, [entry], target_per_scene=4, seed=0)
        self.assertEqual(eps_all, [])  # "every target identifiable" can never hold for a hidden target
        eps, _, _ = IS.episodes_for_scene(scene, [entry], target_per_scene=4, seed=0,
                                          terminal_identifiable="family")
        self.assertGreater(len(eps), 0)
        target, occluder = (C.scene_object(scene, C.role_entry(item, i)) for i in (0, 1))
        for ep in eps:
            pose = np.asarray(ep["init_c2w"])
            for a in ep["certificate"]:
                pose = K.apply_action(pose, a)
            m = fam.judge(item, pose)
            self.assertTrue(m["success"])
            self.assertTrue(m["gates"]["target_in_front"])
            self.assertGreaterEqual(m["target_in_frame_fraction"], 0.80)  # in the frame, not out of view
            self.assertGreaterEqual(m["target_hidden_fraction"], 0.85)    # and behind the occluder
            self.assertTrue(O.identifiable(scene, pose, occluder)["identifiable"])
            self.assertFalse(O.identifiable(scene, pose, target)["identifiable"])

    def test_necessary_mask_and_yaw_window_are_sound(self):
        rng = np.random.default_rng(11)
        for task_type, fam in V.FAMILIES.items():
            for item, cands in self._family_cases(task_type)[:2]:
                scene = self.scenes[item["scene_id"]]
                poses = _random_poses(scene, 150, rng) + [c["c2w"] for c in cands]
                ax, ay = fam.anchor_xy(item)
                for pose in poses:
                    if not fam.judge(item, pose)["success"]:
                        continue
                    x, y = float(pose[0, 3]), float(pose[1, 3])
                    self.assertTrue(bool(fam.necessary_mask(item, np.array([[x, y]]))[0]), task_type)
                    diff = (K.yaw_of(pose) - math.atan2(ay - y, ax - x) + math.pi) % (2 * math.pi) - math.pi
                    self.assertLessEqual(abs(math.degrees(diff)), 90.0, task_type)

    def test_param_menu_rejects_infeasible_pitch0_values(self):
        scene = self.scenes["two_rooms"]
        tea_table = next(o for o in scene.objects if o.id == "4")  # 0.45 m tall
        so = V.FAMILIES["screen_occupancy"].param_menu((tea_table,))
        self.assertNotIn({"occupancy_ratio": 0.6}, so)
        ap = V.FAMILIES["absolute_positioning"].param_menu((tea_table,))
        self.assertNotIn({"distance_m": 1.0}, ap)
        self.assertIn({"distance_m": 2.0}, ap)

    # ------------------------------------------------------------ patch
    def test_patch_is_up_to_date(self):
        current = make_patch.PATCH_PATH.read_text(encoding="utf-8")
        self.assertEqual(current, make_patch.build_patch(REPO), "rerun families/make_patch.py")

    def test_patch_applies_and_patched_runtime_matches(self):
        if shutil.which("patch") is None:
            self.skipTest("patch binary not available")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copytree(REPO / "vagen", tmp / "vagen", ignore=shutil.ignore_patterns("__pycache__"))
            patch_text = make_patch.PATCH_PATH.read_bytes()
            for extra in (["--dry-run"], []):
                res = subprocess.run(["patch", "-p1", "--batch", "--forward"] + extra, cwd=tmp, input=patch_text,
                                     capture_output=True)
                self.assertEqual(res.returncode, 0, res.stdout.decode() + res.stderr.decode())
            vendored = tmp / make_patch.V2_REL
            self.assertEqual(vendored.read_bytes(), (Path(J.__file__)).read_bytes())

            # items x poses -> expected metrics from this module (and unpatched v1)
            from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task

            rng = np.random.default_rng(5)
            cases = []
            for task_type in V.FAMILIES:
                for item, cands in self._family_cases(task_type)[:1]:
                    poses = _random_poses(self.scenes[item["scene_id"]], 15, rng) + [c["c2w"] for c in cands[:2]]
                    for pose in poses:
                        cases.append({"item": item, "pose": pose.tolist(),
                                      "expected": V.score_canonical_task_v2(item, pose)})
            scene = self.scenes["two_rooms"]
            by_id = {o.id: o for o in scene.objects}
            for v1_item in (projective_item("two_rooms", by_id["4"], by_id["2"], "left"),
                            fov_item("two_rooms", by_id["6"], by_id["7"])):
                for v in (K.CANONICAL_TASK_METRIC_V1, K.CANONICAL_TASK_METRIC_V2):
                    it = dict(v1_item, canonical_task_metric_version=v)
                    for pose in _random_poses(scene, 10, rng):
                        cases.append({"item": it, "pose": pose.tolist(),
                                      "expected": score_canonical_task(v1_item, pose)})
            (tmp / "cases.json").write_text(json.dumps(cases, sort_keys=True))
            script = (
                "import json, sys, numpy as np\n"
                "import vagen, vagen.envs.active_spatial.canonical_task_metrics as m\n"
                "assert vagen.__file__.startswith(sys.argv[1]), vagen.__file__\n"
                "cases = json.load(open(sys.argv[2]))\n"
                "out = []\n"
                "for c in cases:\n"
                "    assert m.uses_canonical_backend(c['item'])\n"
                "    out.append(m.score_canonical_task(c['item'], np.asarray(c['pose'])))\n"
                "legacy = {'task_type': 'equidistance'}\n"
                "assert not m.uses_canonical_backend(legacy)\n"
                "for bad in ({'task_type': 'equidistance', 'canonical_task_metric_version': 'canonical_spatial_task_h1_v1'},\n"
                "            {'task_type': 'fov_inclusion', 'canonical_task_metric_version': 'unknown'}):\n"
                "    try:\n"
                "        m.uses_canonical_backend(bad)\n"
                "        raise SystemExit('accepted ' + repr(bad))\n"
                "    except ValueError:\n"
                "        pass\n"
                "print(json.dumps(out, sort_keys=True))\n"
            )
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONPATH"] = str(tmp)
            res = subprocess.run([sys.executable, "-c", script, str(tmp), str(tmp / "cases.json")], cwd=tmp,
                                 env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr[-3000:])
            got = json.loads(res.stdout)
            self.assertEqual(len(got), len(cases))
            for case, metric in zip(cases, got):
                self.assertEqual(json.dumps(metric, sort_keys=True),
                                 json.dumps(_jsonable(case["expected"]), sort_keys=True),
                                 case["item"]["task_type"])


if __name__ == "__main__":
    unittest.main()
