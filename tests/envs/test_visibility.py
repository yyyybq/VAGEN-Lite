"""The visibility rule (vagen/envs/active_spatial/target_visibility.py) shared by env and generator."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import endpoints as E
from data_gen.interiorgs_v10 import occlusion as O
from data_gen.interiorgs_v10.families.canonical_v2 import FAMILIES
from data_gen.interiorgs_v10.sampling import projective_item
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing import synthetic_scene as S
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def _kitchen(root: Path) -> Path:
    """Kitchen [0,5]x[0,4]: a fridge built into a full-height cabinet run whose box overlaps the fridge's
    back half (as merged kitchen boxes do), a wardrobe that can hide it, a cup on a far table."""
    labels = [
        S._box("1", "refrigerator", 4.30, 0.50, 0.0, 5.00, 1.20, 1.90),
        S._box("2", "cabinet", 4.60, 0.30, 0.0, 5.00, 3.60, 2.40),
        S._box("3", "wardrobe", 2.60, 0.20, 0.0, 3.00, 1.60, 2.20),
        S._box("4", "dining table", 0.40, 3.00, 0.0, 1.20, 3.80, 0.75),
        S._box("5", "cup", 0.75, 3.35, 0.75, 0.83, 3.43, 0.85),
    ] + S._walls_for_rect(100, 0.0, 0.0, 5.0, 4.0)
    return S._write(root / "synth_kitchen", labels, [("kitchen", [(0, 0), (5, 0), (5, 4), (0, 4)])], [],
                    y_sign=-1.0, truth={})


def _entry(scene, oid):
    return {o.id: o for o in scene.objects}[oid].target_object_entry()


class VisibilityRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.kitchen = load_scene(_kitchen(root), "synth_kitchen")
        cls.occ = TV.OccluderSet.from_detector(cls.kitchen.detector)
        cls.K = O.canonical_intrinsics()
        cls.paths = write_synthetic_scenes(root / "scenes")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _vis(self, x, y, yaw_deg, oid):
        pose = C.pose_from_xy_yaw(x, y, math.radians(yaw_deg))
        return TV.object_visibility(pose, _entry(self.kitchen, oid), self.occ, self.K, O.IMG_W, O.IMG_H)

    def test_face_geometry(self):
        g = TV.FaceGeometry([0, 0, 0], [1, 2, 3], n=5)
        self.assertEqual(g.points.shape, (150, 3))
        self.assertEqual(sorted(g.areas.tolist()), [2.0, 2.0, 3.0, 3.0, 6.0, 6.0])
        self.assertTrue(np.allclose(np.abs(g.normals).sum(axis=1), 1.0))

    def test_built_in_fridge_is_visible_by_its_front_face(self):
        fridge = {o.id: o for o in self.kitchen.objects}["1"]
        pose = C.pose_from_xy_yaw(3.2, 0.85, 0.0)  # past the wardrobe, facing the fridge
        # the old rule counts the box samples inside the overlapping cabinet box as hidden
        self.assertLess(O.visible_fraction(self.kitchen, pose, fridge), 0.5)
        v = self._vis(3.2, 0.85, 0.0, "1")
        self.assertTrue(v["visible"])
        self.assertEqual(v["face"], 0)  # the -x face, i.e. the front
        self.assertGreaterEqual(v["visible_fraction"], TV.MIN_FACE_VISIBLE_FRACTION)  # the bottom is below the frame

    def test_hidden_out_of_frame_and_too_small(self):
        behind = self._vis(1.6, 0.9, 0.0, "1")  # the wardrobe (x 2.6-3.0, y 0.2-1.6) stands in between
        self.assertFalse(behind["visible"])
        self.assertFalse(self._vis(3.2, 0.85, 180.0, "1")["visible"])  # looking away
        far = self._vis(4.0, 2.0, 160.0, "5")  # a 8 cm cup about 3.5 m away
        self.assertFalse(far["visible"])
        self.assertLess(far["area_fraction"], TV.MIN_FACE_AREA_FRACTION)

    def test_generator_and_env_agree(self):
        scene = load_scene(self.paths["two_rooms"], "two_rooms")
        by_id = {o.id: o for o in scene.objects}
        item = projective_item("two_rooms", by_id["4"], by_id["2"], "left")
        self.assertTrue(np.array_equal(TV.item_camera(item)[0], O.canonical_intrinsics()))
        occ = O.face_occluders_for(scene)
        rng = np.random.default_rng(3)
        poses = []
        for _ in range(300):
            x, y = rng.uniform(0.3, 5.7), rng.uniform(0.3, 4.7)
            yaw = math.atan2(2.5 - y, 3.5 - x) + rng.uniform(-0.6, 0.6)
            poses.append(C.pose_from_xy_yaw(x, y, yaw))
        poses = np.stack(poses)
        previous = O.visibility_rule()
        O.set_visibility_rule("env")
        try:
            batch = E.evaluate_c2ws(scene, item, poses)
            vis_ok, _ = E.face_gate(scene, item, poses[:, :3, :3], poses[:, :3, 3])
            fn = E.success_fn(item, scene)
            rt = E.runtime_success_fn(item, scene)
            rt_batch = rt.batch(poses)
        finally:
            O.set_visibility_rule(previous)
        env = [TV.score_with_visibility(item, p, occ) for p in poses]
        gates = [TV.rule_gates(item, p, occ)["ok"] for p in poses]  # visibility and the 45-degree rule
        self.assertEqual(vis_ok.tolist(), gates)
        exact = [bool(m["success"]) for m in env]
        self.assertEqual([bool(fn.exact(p)) for p in poses], exact)
        self.assertEqual(rt_batch.tolist(), exact)
        far_from_threshold = ~np.asarray(batch["near"], bool)
        self.assertEqual(np.asarray(batch["success"])[far_from_threshold].tolist(),
                         np.asarray(exact)[far_from_threshold].tolist())
        self.assertGreater(sum(exact), 0)
        self.assertGreater(sum(m["gates"].get("relation", False) and not m["success"] for m in env)
                           + sum(not g for g in gates), 0)

    def test_endpoint_sampler_verifies_with_the_face_rule(self):
        scene = load_scene(self.paths["two_rooms"], "two_rooms")
        by_id = {o.id: o for o in scene.objects}
        item = projective_item("two_rooms", by_id["4"], by_id["2"], "left")
        previous = O.visibility_rule()
        O.set_visibility_rule("env")
        try:
            eps, funnel = E.sample_endpoints(scene, item, max_endpoints=4, seed=0)
        finally:
            O.set_visibility_rule(previous)
        verify = funnel.counts.get("endpoint_verify", {})
        self.assertNotIn("mismatch_occlusion", verify)  # the old box test must not run under the env rule
        self.assertGreater(verify.get("ok", 0), 0)
        occ = O.face_occluders_for(scene)
        for ep in eps:
            self.assertTrue(TV.score_with_visibility(item, np.asarray(ep["c2w"]), occ)["success"])

    def test_shaping_is_at_least_half_exactly_on_success(self):
        scene = load_scene(self.paths["two_rooms"], "two_rooms")
        by_id = {o.id: o for o in scene.objects}
        items = [projective_item("two_rooms", by_id["4"], by_id["2"], "left"),
                 FAMILIES["equidistance"].build_item("two_rooms", (by_id["4"], by_id["1"]), {}, None),
                 FAMILIES["fov_inclusion"].build_item("two_rooms", (by_id["4"], by_id["2"]), {}, None)]
        occ = O.face_occluders_for(scene)
        rng = np.random.default_rng(4)
        for item in items:
            n_success = 0
            xy = rng.uniform([0.3, 0.3], [5.7, 4.7], size=(3000, 2))
            fam = FAMILIES.get(item["task_type"])
            if fam is not None:  # where the family can succeed at all
                xy = xy[np.asarray(fam.necessary_mask(item, xy), bool)]
            ax, ay = fam.anchor_xy(item) if fam is not None else (3.5, 2.5)
            for x, y in xy[:200]:
                pose = C.pose_from_xy_yaw(x, y, math.atan2(ay - y, ax - x) + rng.uniform(-0.3, 0.3))
                m = TV.score_with_visibility(item, pose, occ)
                self.assertEqual(m["shaping_score"] >= 0.5, bool(m["success"]), item["task_type"])
                n_success += bool(m["success"])
            self.assertGreater(n_success, 0, item["task_type"])

    def test_r1_legacy_switch(self):
        by_id = {o.id: o for o in self.kitchen.objects}
        item = projective_item("synth_kitchen", by_id["3"], by_id["1"], "left")
        pose = C.pose_from_xy_yaw(1.6, 2.6, math.radians(-40.0))
        occ = self.occ

        def metric(legacy):
            stub = SimpleNamespace(config=ActiveSpatialEnvConfig(r1_legacy_success=legacy), current_item=item,
                                   view_engine=SimpleNamespace(get_pose=lambda: pose),
                                   _visibility_occluders=lambda: occ)
            return ActiveSpatialEnv._calculate_canonical_metric(stub)

        self.assertFalse(ActiveSpatialEnvConfig().r1_legacy_success)
        legacy = metric(True)
        self.assertEqual(legacy, score_canonical_task(item, pose))
        self.assertNotIn("visibility", legacy)
        new = metric(False)
        self.assertEqual(new, TV.score_with_visibility(item, pose, occ))
        self.assertIn("visibility", new)
        self.assertEqual(new["success"], bool(legacy["success"]) and new["visibility"]["visible"])

    def test_fov_v2_judge_through_the_env(self):
        scene = load_scene(self.paths["two_rooms"], "two_rooms")
        by_id = {o.id: o for o in scene.objects}
        item = FAMILIES["fov_inclusion"].build_item("two_rooms", (by_id["4"], by_id["2"]), {}, None)
        self.assertEqual(item["canonical_task_metric_version"], C.CANONICAL_TASK_METRIC_V2)
        m = score_canonical_task(item, C.pose_from_xy_yaw(0.8, 2.5, 0.0))
        self.assertEqual(m["task_metric"], "fov_inclusion_h1_v2")
        self.assertIn("fully_in_frame", m["gates"])
        self.assertIn("both", item["task_description"])


if __name__ == "__main__":
    unittest.main()
