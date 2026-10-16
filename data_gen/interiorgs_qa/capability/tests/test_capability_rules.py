"""CPU checks of environment reuse, ambiguous-state rejection and answer independence."""
import copy
import json
import unittest

import numpy as np

from data_gen.interiorgs_qa.capability import rules, templates
from data_gen.interiorgs_v10.contract import NATIVE_K, pose_from_xy_yaw
from vagen.envs.active_spatial import target_visibility as environment


def task_fixture(kind="absolute_positioning"):
    objects = [{"id": "a", "label": "cabinet", "ref": "the cabinet next to the TV",
                "bbox_min": [-.3, 2.5, .2], "bbox_max": [.3, 2.6, 1.4]}]
    if kind != "absolute_positioning":
        objects[0]["bbox_min"] = [-1.2, 3.8, .2]
        objects[0]["bbox_max"] = [-.8, 4.2, 1.4]
        objects.append({"id": "b", "label": "chair", "ref": "the chair",
                        "bbox_min": [.8, 3.8, .2], "bbox_max": [1.2, 4.2, 1.4]})
    params = {"distance_m": 2.5, "relation": "left"}
    for role, obj in zip("ab", objects):
        params[f"object_{role}_id"] = obj["id"]
        params[f"object_{role}_center"] = ((np.asarray(obj["bbox_min"])+obj["bbox_max"])/2).tolist()
    return {"task_id": "test-task", "task_type": kind,
            "canonical_task_metric_version": "canonical_spatial_task_h1_v2",
            "camera_model_version": "canonical_camera_h1_resize_v1",
            "init_camera": {"intrinsics": NATIVE_K},
            "target_object": {"objects": objects}, "target_region": {"params": params}}


def measured_fixture(kind):
    task = task_fixture(kind)
    objs = []
    for role, obj in zip("ab", task["target_object"]["objects"]):
        objs.append({"id": role, "ref": obj["ref"], "references": [],
                     "visibility": {"visible": True}, "distance_m": 2.5,
                     "center_distance_m": 2.55,
                     "center_camera": [-1 if role == "a" else 1, .5, 3],
                     "projection": {"all_corners_in_front": True, "visible": True,
                                    "area_ratio": .05, "inside_fraction": 1.,
                                    "center_uv": [80 if role == "a" else 170, 120],
                                    "bbox_raw": [40, 50, 120, 180]}})
    measured = {"source_task_id": task["task_id"], "task_type": kind,
                "camera_valid": True, "pose_c2w": pose_from_xy_yaw(0, 0, np.pi/2).tolist(),
                "metric": {"success": False}, "objects": objs,
                "comparison_metric": {"equidistance_tolerance_m": .25},
                "lateral": {"angle_deg": 0., "max_angle_deg": 45.},
                "camera": {"width": 256, "height": 256}}
    return task, measured


class EnvironmentMeasurementTests(unittest.TestCase):
    def test_actual_environment_and_camera_used(self):
        task = task_fixture()
        pose = pose_from_xy_yaw(0, 0, np.pi/2)
        occ = environment.OccluderSet.from_boxes([])
        measured = rules.measure(task, pose, occ)
        direct = environment.score_with_visibility(task, pose, occ)
        self.assertEqual(measured["metric"], direct)
        self.assertAlmostEqual(measured["objects"][0]["distance_m"], 2.5)
        self.assertAlmostEqual(measured["objects"][0]["center_distance_m"], 2.55)
        self.assertEqual(measured["objects"][0]["ref"], "the cabinet next to the TV")
        self.assertAlmostEqual(measured["camera"]["K_effective"][0][0], 128.)
        self.assertAlmostEqual(measured["camera"]["K_effective"][1][1], 320*256/480)
        self.assertFalse(measured["rgb_verified"])
        self.assertFalse(measured["rendered_depth_verified"])
        json.dumps(measured, allow_nan=False)
        drafts, reasons = rules.propose(task, measured)
        self.assertEqual(reasons, [])
        self.assertEqual(len(drafts), 1)
        self.assertFalse(drafts[0]["extra"]["final_eligible"])

    def test_occluder_scene_and_current_camera_required(self):
        task, pose = task_fixture(), pose_from_xy_yaw(0, 0, 0)
        with self.assertRaises(ValueError):
            rules.measure(task, pose, None)
        pose[2, 3] = 1.7
        with self.assertRaises(ValueError):
            rules.measure(task, pose, environment.OccluderSet.from_boxes([]))

    def test_actual_blocker_changes_visibility_and_rejects_draft(self):
        task, pose = task_fixture(), pose_from_xy_yaw(0, 0, np.pi/2)
        occ = environment.OccluderSet.from_boxes([("blocker", "cabinet", [-1, 1, 0], [1, 1.3, 2])])
        measured = rules.measure(task, pose, occ)
        self.assertFalse(measured["objects"][0]["visibility"]["visible"])
        self.assertEqual(rules.propose(task, measured), ([], ["target_not_identifiable"]))


class DraftRulesTests(unittest.TestCase):
    def test_distance_inside_formal_range_uses_fixed_menu_for_every_seed(self):
        task, measured = measured_fixture("absolute_positioning")
        ranks = set()
        for distance in (1.5, 2., 2.5, 3.):
            measured["objects"][0]["distance_m"] = distance
            measured["objects"][0]["center_distance_m"] = distance + .05
            for seed in range(25):
                drafts, reasons = rules.propose(task, measured, {"seed": seed})
                self.assertFalse(reasons)
                draft = drafts[0]
                values = draft["fields"]["distance_options_m"]
                self.assertEqual(values, [1.5, 2., 2.5, 3.])
                idx = draft["option_keys"].index(draft["answer_key"])
                self.assertEqual(values[idx], distance)
                ranks.add(idx)
        self.assertEqual(ranks, {0, 1, 2, 3})

    def test_outside_distance_range_extends_actual_interval_without_requested_rank(self):
        task, measured = measured_fixture("absolute_positioning")
        measured["objects"][0]["distance_m"] = 4.5
        measured["objects"][0]["center_distance_m"] = 4.55
        ranks = set()
        for seed in range(100):
            draft = rules.propose(task, measured, {"seed": seed})[0][0]
            self.assertEqual(rules.propose(task, measured, {"seed": seed})[0][0], draft)
            values = draft["fields"]["distance_options_m"]
            self.assertEqual(values, sorted(set(values)))
            self.assertEqual([values[i+1]-values[i] for i in range(3)], [.5]*3)
            idx = draft["option_keys"].index(draft["answer_key"])
            self.assertEqual(values[idx], 4.5)
            ranks.add(idx)
        self.assertEqual(ranks, {0, 1, 2, 3})

    def test_distance_grid_ambiguity_not_hidden_by_window_edge(self):
        task, measured = measured_fixture("absolute_positioning")
        measured["objects"][0]["distance_m"] = 2.25
        measured["objects"][0]["center_distance_m"] = 2.25
        for seed in range(20):
            self.assertEqual(rules.propose(task, measured, {"seed": seed}), ([], ["distance_grid_ambiguous"]))
        measured["objects"][0]["distance_m"] = 2.5
        measured["objects"][0]["center_distance_m"] = 3.
        self.assertEqual(rules.propose(task, measured)[1], ["distance_side_center_disagree_or_ambiguous"])

    def test_equidistance_uses_environment_tolerance_with_buffer(self):
        task, measured = measured_fixture("equidistance")
        self.assertEqual(rules.propose(task, measured)[0][0]["answer_key"], "same")
        measured["objects"][1]["distance_m"] = 2.78
        measured["objects"][1]["center_distance_m"] = 2.83
        self.assertEqual(rules.propose(task, measured)[1], ["comparison_ambiguous"])
        measured["objects"][1]["distance_m"] = 4.
        measured["objects"][1]["center_distance_m"] = 4.05
        self.assertEqual(rules.propose(task, measured)[0][0]["answer_key"], "a_closer")

    def test_projective_excludes_45_degree_boundary_and_small_image_gap(self):
        task, measured = measured_fixture("projective_relations")
        self.assertEqual(rules.propose(task, measured)[0][0]["answer_key"], "left")
        for angle in (35., 45., 55.):
            measured["lateral"]["angle_deg"] = angle
            self.assertEqual(rules.propose(task, measured)[1], ["direction_angular_ambiguity"])
        measured["lateral"]["angle_deg"] = 0.
        measured["objects"][0]["projection"]["center_uv"][0] = 160
        self.assertEqual(rules.propose(task, measured)[1], ["direction_screen_order_unclear"])

    def test_longitudinal_direction_requires_depth_side_and_center_agreement(self):
        task, measured = measured_fixture("projective_relations")
        measured["lateral"]["angle_deg"] = 80.
        measured["objects"][0]["center_camera"][2] = 2.
        measured["objects"][1]["distance_m"] = 4.
        measured["objects"][1]["center_distance_m"] = 4.05
        self.assertEqual(rules.propose(task, measured)[0][0]["answer_key"], "front")
        measured["objects"][0]["center_camera"][2] = 5.
        self.assertEqual(rules.propose(task, measured)[1], ["direction_depth_distance_ambiguous_or_disagree"])

    def test_occlusion_nearer_question_cannot_label_invisible_target(self):
        task, measured = measured_fixture("occlusion_alignment")
        measured["objects"][1]["distance_m"] = 4.
        measured["objects"][1]["center_distance_m"] = 4.05
        measured["objects"][1]["center_camera"][2] = 5.
        self.assertEqual(rules.propose(task, measured)[0][0]["answer_key"], "a_closer")
        measured["objects"][0]["visibility"]["visible"] = False
        self.assertEqual(rules.propose(task, measured)[1], ["target_not_identifiable"])

    def test_framing_draft_has_per_object_flags_and_requires_real_contour(self):
        task, measured = measured_fixture("fov_inclusion")
        measured["objects"][1]["projection"].update(inside_fraction=.7, bbox_raw=[-40, 50, 80, 180])
        draft = rules.propose(task, measured)[0][0]
        self.assertEqual(draft["answer_key"], "a_only")
        self.assertEqual(draft["extra"]["full_flags"], [True, False])
        self.assertTrue(draft["extra"]["requires_actual_contour_agreement"])
        self.assertFalse(draft["extra"]["final_eligible"])
        measured["objects"][1]["projection"]["inside_fraction"] = .96
        self.assertEqual(rules.propose(task, measured)[1], ["framing_geometry_ambiguous"])

    def test_unresolved_reference_and_measurement_mismatch_fail_closed(self):
        task, measured = measured_fixture("equidistance")
        measured["source_task_id"] = "another-task"
        self.assertEqual(rules.propose(task, measured)[1], ["measurement_task_mismatch"])
        measured["source_task_id"] = task["task_id"]
        measured["objects"][0]["references"] = [{"visibility": {"visible": False}}]
        self.assertEqual(rules.propose(task, measured)[1], ["reference_not_identifiable"])


class TemplateTests(unittest.TestCase):
    def test_answer_does_not_choose_wording_or_option_order(self):
        task, measured = measured_fixture("projective_relations")
        draft = rules.propose(task, measured)[0][0]
        template_ids, orders = set(), set()
        for seed in range(60):
            baseline = templates.render_draft(draft, {"task_id": "opaque", "pose_hash": "123"}, seed)
            template_ids.add(baseline["template_id"])
            orders.add(tuple(baseline["option_keys"]))
            self.assertTrue(all(isinstance(s, str) for s in baseline["options"]))
            for key in draft["option_keys"]:
                changed = copy.deepcopy(draft)
                changed["answer_key"] = key
                actual = templates.render_draft(changed, {"task_id": "opaque", "pose_hash": "123"}, seed)
                self.assertEqual(actual["question"], baseline["question"])
                self.assertEqual(actual["options"], baseline["options"])
                self.assertEqual(actual["template_id"], baseline["template_id"])
                self.assertEqual(actual["option_keys"][ord(actual["answer"])-65], key)
        self.assertEqual(len(template_ids), 3)
        self.assertGreater(len(orders), 10)

    def test_distance_order_and_formal_reference_case_preserved(self):
        task, measured = measured_fixture("absolute_positioning")
        draft = rules.propose(task, measured)[0][0]
        rendered = templates.render_draft(draft, "opaque", 3)
        self.assertIn("the cabinet next to the TV", rendered["question"])
        self.assertEqual([float(s.split()[0]) for s in rendered["options"]], draft["fields"]["distance_options_m"])
        task, measured = measured_fixture("equidistance")
        rendered = templates.render_draft(rules.propose(task, measured)[0][0], "opaque", 3)
        self.assertTrue(any("the cabinet next to the TV" in s for s in rendered["options"]))

    def test_two_view_occlusion_templates_and_answer_identity_rejection(self):
        draft = {"kind": "occlusion_state", "fields": {"a": "the chair", "b": "the bed"},
                 "answer_key": "fully_hidden",
                 "option_keys": ["fully_hidden", "partly_hidden", "not_hidden", "outside"]}
        seen = set()
        for seed in range(30):
            rendered = templates.render_draft(draft, {"scene_id": "s", "frame_hash": "opaque"}, seed)
            seen.add(rendered["template_id"])
            self.assertIn("first image", rendered["question"].lower())
            self.assertIn("second", rendered["question"].lower())
            self.assertEqual(len(rendered["options"]), 4)
        self.assertEqual(len(seen), 3)
        with self.assertRaises(ValueError):
            templates.render_draft(draft, {"answer": "fully_hidden"}, 0)
        with self.assertRaises(ValueError):
            templates.render_draft(draft, "answer=fully_hidden", 0)


if __name__ == "__main__":
    unittest.main()
