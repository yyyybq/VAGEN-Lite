"""2026-10-14 success rules (vagen/envs/active_spatial/target_visibility.py): the 45-degree left/right rule,
reference objects that must be visible and the visibility thresholds set on the human labels; the generator
applies them exactly as the env does."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import endpoints as E
from data_gen.interiorgs_v10 import occlusion as O
from data_gen.interiorgs_v10.sampling import projective_item
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing import synthetic_scene as S
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task


def _kitchen(root: Path) -> Path:
    labels = [
        S._box("1", "refrigerator", 4.30, 0.50, 0.0, 5.00, 1.20, 1.90),
        S._box("3", "wardrobe", 2.60, 1.60, 0.0, 3.00, 3.00, 2.20),
        S._box("4", "dining table", 3.80, 1.40, 0.0, 4.60, 2.20, 0.75),
        S._box("6", "sofa", 0.20, 0.20, 0.0, 1.00, 2.00, 0.85),
    ] + S._walls_for_rect(100, 0.0, 0.0, 5.0, 4.0)
    return S._write(root / "r5_kitchen", labels, [("", [(0, 0), (5, 0), (5, 4), (0, 4)])], [],
                    y_sign=-1.0, truth={})


class SuccessRulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.kitchen = load_scene(_kitchen(root), "r5_kitchen")
        cls.occ = TV.OccluderSet.from_detector(cls.kitchen.detector)
        cls.paths = write_synthetic_scenes(root / "scenes")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_thresholds_from_the_labels(self):
        self.assertEqual(TV.VISIBILITY_RULE_VERSION, "face_visible_v2")
        self.assertEqual(TV.MIN_FACE_VISIBLE_FRACTION, 0.4)
        self.assertEqual(TV.MIN_FACE_AREA_FRACTION, 0.003)
        self.assertEqual(TV.LATERAL_MAX_ANGLE_DEG, 45.0)

    def test_lateral_rule(self):
        item = {"task_type": "projective_relations",
                "target_region": {"params": {"object_a_center": [2.0, 0.0, 0.5], "object_b_center": [0.0, 0.0, 0.5]}}}
        # B -> A points along +x; facing +y the line runs along the camera's left-right axis
        for yaw_from_y, ok in ((0.0, True), (40.0, True), (50.0, False), (90.0, False), (-44.0, True), (-46.0, False)):
            pose = C.pose_from_xy_yaw(1.0, -3.0, math.radians(90.0 + yaw_from_y))
            gate = TV.lateral_gate(item, pose)
            self.assertEqual(gate["ok"], ok, yaw_from_y)
            self.assertAlmostEqual(gate["angle_deg"], abs(yaw_from_y), places=6)
            self.assertEqual(TV.lateral_ok(item, pose[None, :3, :3]).tolist(), [ok])
            self.assertEqual(gate["progress"] == 1.0, ok)
        poses = np.stack([C.pose_from_xy_yaw(0.0, 0.0, a) for a in np.linspace(-math.pi, math.pi, 721)])
        single = [TV.lateral_gate(item, p)["ok"] for p in poses]
        self.assertEqual(TV.lateral_ok(item, poses[:, :3, :3]).tolist(), single)  # batch == one pose
        self.assertTrue(TV.lateral_gate({"task_type": "fov_inclusion"}, poses[0])["ok"])  # projective only

    def test_reference_object_must_be_visible(self):
        by_id = {o.id: o for o in self.kitchen.objects}
        fridge = by_id["1"].target_object_entry()
        base = {"task_type": "absolute_positioning", "init_camera": {"intrinsics": C.NATIVE_K},
                "target_region": {"params": {}}}
        pose = C.pose_from_xy_yaw(2.0, 0.85, 0.0)  # facing the fridge along +x
        cases = {}
        for anchor_id in (None, "4", "6"):  # none, the dining table (in view), the sofa (behind the camera)
            entry = dict(fridge)
            if anchor_id:
                a = by_id[anchor_id]
                entry["ref_anchor"] = {"id": a.id, "label": a.label, "bbox_min": a.aabb_min.tolist(),
                                       "bbox_max": a.aabb_max.tolist()}
            item = dict(base, target_object={"objects": [entry]})
            cases[anchor_id] = TV.visibility_gates(item, pose, self.occ)
        self.assertTrue(cases[None]["targets"]["a"]["visible"])
        self.assertEqual(cases[None]["references"], {})
        self.assertTrue(cases["4"]["references"]["a_ref"]["visible"])
        self.assertTrue(cases["4"]["visible"])
        self.assertFalse(cases["6"]["references"]["a_ref"]["visible"])
        self.assertFalse(cases["6"]["visible"])  # the target alone is visible: the reference decides

    def test_generator_and_env_agree_with_both_rules(self):
        scene = load_scene(self.paths["two_rooms"], "two_rooms")
        by_id = {o.id: o for o in scene.objects}
        item = projective_item("two_rooms", by_id["4"], by_id["2"], "left")
        a = by_id["1"]  # a reference object for A
        item["target_object"]["objects"][0]["ref_anchor"] = {"id": a.id, "label": a.label,
                                                              "bbox_min": a.aabb_min.tolist(),
                                                              "bbox_max": a.aabb_max.tolist()}
        occ = O.face_occluders_for(scene)
        rng = np.random.default_rng(5)
        poses = np.stack([C.pose_from_xy_yaw(x, y, math.atan2(2.5 - y, 3.5 - x) + rng.uniform(-1.0, 1.0))
                          for x, y in rng.uniform([0.3, 0.3], [5.7, 4.7], size=(400, 2))])
        previous = O.visibility_rule()
        O.set_visibility_rule("env")
        try:
            gate_ok, _ = E.face_gate(scene, item, poses[:, :3, :3], poses[:, :3, 3])
            fn = E.success_fn(item, scene)
            rt = E.runtime_success_fn(item, scene)
            rt_batch = rt.batch(poses)
            exact_gen = [bool(fn.exact(p)) for p in poses]
        finally:
            O.set_visibility_rule(previous)
        rules = [TV.rule_gates(item, p, occ) for p in poses]
        self.assertEqual(gate_ok.tolist(), [r["ok"] for r in rules])
        env = [TV.score_with_visibility(item, p, occ) for p in poses]
        exact = [bool(m["success"]) for m in env]
        self.assertEqual(exact_gen, exact)
        self.assertEqual(rt_batch.tolist(), exact)
        canonical = [bool(score_canonical_task(item, p)["success"]) for p in poses]
        # each new rule rejects some frames the canonical metric accepts
        self.assertTrue(any(c and not r["lateral"]["ok"] for c, r in zip(canonical, rules)))
        self.assertTrue(any(c and r["lateral"]["ok"] and r["visibility"]["targets"]["a"]["visible"]
                            and r["visibility"]["targets"]["b"]["visible"] and not r["visibility"]["visible"]
                            for c, r in zip(canonical, rules)))
        for m in env:
            self.assertEqual(m["shaping_score"] >= 0.5, bool(m["success"]))

    def test_face_v1_is_an_alias_of_env(self):
        previous = O.visibility_rule()
        try:
            O.set_visibility_rule("face_v1")
            self.assertEqual(O.visibility_rule(), "env")
            self.assertTrue(O.env_rule())
        finally:
            O.set_visibility_rule(previous)



if __name__ == "__main__":
    unittest.main()
