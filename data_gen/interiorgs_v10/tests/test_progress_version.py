"""Round 12 freeze (the user's decision of 2026-10-22, docs/dense_reward_dcond.md): the D-cond progress version.
The default ("product") keeps every metric bit for bit; "grouped_gated" carries 0.25 x frame + 0.25 x face + 0.5 x
relation, the relation group only once every frame condition is met; success never depends on the version."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import occlusion as O
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.families.canonical_v2 import FAMILIES
from data_gen.interiorgs_v10.sampling import projective_item
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes
from vagen.envs.active_spatial import target_visibility as TV
from vagen.envs.active_spatial.canonical_task_metrics import score_canonical_task
from vagen.envs.active_spatial.canonical_task_metrics_v2 import FAILING_PROGRESS_CAP
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def old_apply_visibility(metric, vis, lateral=None):
    """apply_visibility as it was before the progress versions (commit 2207a920), for the bit-for-bit check."""
    out = dict(metric)
    gates = dict(out.get("gates") or {})
    for letter, t in vis["targets"].items():
        gates[f"target_{letter}_clearly_visible"] = bool(t["visible"])
    for key, r in (vis.get("references") or {}).items():
        gates[f"reference_{key[0]}_clearly_visible"] = bool(r["visible"])
    gates["required_targets_found"] = bool(vis["found"])
    lat_ok, lat_progress = True, 1.0
    if lateral is not None and lateral.get("applies"):
        lat_ok, lat_progress = bool(lateral["ok"]), float(lateral["progress"])
        gates["lateral_within_45deg"] = lat_ok
    success = bool(metric.get("success")) and bool(vis["visible"]) and lat_ok
    base = metric.get("continuous_gate_progress")
    if base is None:
        base = float(metric.get("shaping_score", 0.0) or 0.0)
    progress = float(min(1.0, max(0.0, float(base)))) * float(vis["progress"]) * lat_progress
    if not success:
        progress = min(progress, FAILING_PROGRESS_CAP)
    out.update(success=success, gates=gates, continuous_gate_progress=progress,
               shaping_score=0.5 * progress + 0.5 * float(success), visibility=vis)
    if lateral is not None and lateral.get("applies"):
        out["lateral"] = lateral
    return out


def vis_of(progress, visible=True):
    return {"targets": {"a": {"visible": visible}}, "references": {}, "found": True, "visible": visible,
            "progress": progress}


class ProgressVersionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = write_synthetic_scenes(Path(cls._tmp.name) / "scenes")
        cls.scene = load_scene(cls.paths["two_rooms"], "two_rooms")
        cls.occ = O.face_occluders_for(cls.scene)
        by_id = {o.id: o for o in cls.scene.objects}
        cls.items = [projective_item("two_rooms", by_id["4"], by_id["2"], "left"),
                     FAMILIES["equidistance"].build_item("two_rooms", (by_id["4"], by_id["1"]), {}, None),
                     FAMILIES["fov_inclusion"].build_item("two_rooms", (by_id["4"], by_id["2"]), {}, None)]
        rng = np.random.default_rng(5)
        cls.poses = []
        for _ in range(120):
            x, y = rng.uniform(0.3, 5.7), rng.uniform(0.3, 4.7)
            cls.poses.append(C.pose_from_xy_yaw(x, y, math.atan2(2.5 - y, 3.5 - x) + rng.uniform(-1.2, 1.2)))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_default_is_bit_for_bit_the_old_metric(self):
        for item in self.items:
            for pose in self.poses:
                metric = score_canonical_task(item, pose)
                vis = TV.visibility_gates(item, pose, self.occ)
                lat = TV.lateral_gate(item, pose)
                old = old_apply_visibility(metric, vis, lat)
                self.assertEqual(TV.apply_visibility(metric, vis, lat), old)
                self.assertEqual(TV.apply_visibility(metric, vis, lat, progress_version="product"), old)
                self.assertEqual(TV.score_with_visibility(item, pose, self.occ), old)

    def test_grouped_gated_on_real_tasks(self):
        n_gate_off = n_success = 0
        for item in self.items:
            for pose in self.poses:
                old = TV.score_with_visibility(item, pose, self.occ)
                new = TV.score_with_visibility(item, pose, self.occ, progress_version="grouped_gated")
                self.assertEqual(new["success"], old["success"])
                self.assertEqual(new["gates"], old["gates"])
                self.assertEqual(new["shaping_score"] >= 0.5, bool(new["success"]))
                g = new["progress_groups"]
                want = 0.25 * g["frame"] + 0.25 * g["face"] + 0.5 * g["relation"] * (1.0 if g["frame_met"] else 0.0)
                if not new["success"]:
                    want = min(want, FAILING_PROGRESS_CAP)
                self.assertAlmostEqual(new["continuous_gate_progress"], want, places=12)
                self.assertEqual(new["progress_version"], "grouped_gated")
                n_gate_off += not g["frame_met"]
                n_success += bool(new["success"])
        self.assertGreater(n_gate_off, 0)
        self.assertGreater(n_success, 0)

    def test_the_documented_examples(self):
        # only the relation is missing (screen occupancy far off): 0.25 + 0.25 + 0.5 x 0.074
        frame = {"objects_present": 1.0, "in_front": 1.0, "visible": 1.0, "min_area": 1.0, "inside_frame": 1.0}
        metric = {"success": False, "continuous_gate_progress": 0.074, "gates": {**{k: True for k in frame},
                                                                                 "occupancy": False},
                  "shaping_components": {**frame, "occupancy": 0.074}}
        out = TV.apply_visibility(metric, vis_of(1.0), progress_version="grouped_gated")
        self.assertAlmostEqual(out["continuous_gate_progress"], 0.25 + 0.25 + 0.5 * 0.074, places=12)
        self.assertAlmostEqual(TV.apply_visibility(metric, vis_of(1.0))["continuous_gate_progress"], 0.074, places=12)
        # the relation met but the objects not in front of the camera: the hard gate keeps the relation out
        metric4 = {"success": False, "continuous_gate_progress": 0.0,
                   "gates": {"in_front": False, "visible": True, "equidistant": True},
                   "shaping_components": {"in_front": 0.0, "visible": 1.0, "equidistant": 0.95}}
        out4 = TV.apply_visibility(metric4, vis_of(0.4, visible=False), progress_version="grouped_gated")
        self.assertAlmostEqual(out4["continuous_gate_progress"], 0.25 * 0.4, places=12)
        self.assertFalse(out4["progress_groups"]["frame_met"])
        # the left/right 45-degree rule belongs to the relation group
        lat = {"applies": True, "ok": False, "progress": 0.5}
        out5 = TV.apply_visibility(metric, vis_of(1.0), lat, progress_version="grouped_gated")
        self.assertAlmostEqual(out5["continuous_gate_progress"], 0.5 + 0.5 * 0.074 * 0.5, places=12)
        with self.assertRaises(ValueError):
            TV.apply_visibility(metric, vis_of(1.0), progress_version="sum")

    def test_env_switch_and_preset(self):
        self.assertEqual(ActiveSpatialEnvConfig().progress_version, "product")
        item, pose = self.items[1], self.poses[0]
        for version in TV.PROGRESS_VERSIONS:
            stub = SimpleNamespace(config=ActiveSpatialEnvConfig(progress_version=version), current_item=item,
                                   view_engine=SimpleNamespace(get_pose=lambda: pose),
                                   _visibility_occluders=lambda: self.occ)
            self.assertEqual(ActiveSpatialEnv._calculate_canonical_metric(stub),
                             TV.score_with_visibility(item, pose, self.occ, progress_version=version))
        self.assertEqual(TP.reward_overrides("D_cond"), {**TP.reward_overrides("dense"),
                                                         "progress_version": "grouped_gated"})
        self.assertNotIn("progress_version", TP.reward_overrides("dense"))
        self.assertEqual(TP.load_standards()["v10_p1_Dcond"]["reward_preset"], "D_cond")


if __name__ == "__main__":
    unittest.main()
