"""Round-6 rules (2026-10-14, after the round-5 check page): names relative to the start's room (referring rules
v3, start_context.py), stricter room types, no tasks in scenes without walls and occlusion-alignment starts
with the target visible."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import objects as OB
from data_gen.interiorgs_v10 import referring as R
from data_gen.interiorgs_v10 import start_context as SC
from data_gen.interiorgs_v10.contract import Obj
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.scene import point_in_polygon
from data_gen.interiorgs_v10.testing import synthetic_scene as S


def _box(i, label, x0, y0, z0, x1, y1, z1, room=None):
    return Obj(id=str(i), label=label, aabb_min=np.array([x0, y0, z0], float),
               aabb_max=np.array([x1, y1, z1], float), room_index=room)


def _room(i, x0, y0, x1, y1):
    return SimpleNamespace(index=i, name="", polygon=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))


def _scene(objs, rooms):
    def room_of(x, y):
        for r in rooms:
            if point_in_polygon(x, y, r.polygon):
                return r.index
        return None
    return SimpleNamespace(scene_id="r6", scene_dir=None, objects=objs, rooms=rooms, convention={}, room_of=room_of)


def _assign(scene, ids, preset="v10_r6"):
    cfg = OB.get_preset(preset)
    objs = [copy.copy(o) for o in scene.objects if o.id in ids]
    R.assign_referring_expressions(scene, objs, cfg.referring, pool=scene.objects,
                                   anchor_ok=lambda a: OB.semantic_exclusion_reason(a.label, cfg) is None)
    return {o.id: o for o in objs}


# living room (0) with a sofa and a chair, bedroom (1) with a bed and a chair, a second bedroom (2) with a bed,
# a wardrobe in each bedroom
ROOMS = [_room(0, 0, 0, 5, 5), _room(1, 5, 0, 10, 5), _room(2, 10, 0, 15, 5)]
OBJS = [_box("s", "sofa", 0.2, 0.2, 0, 2.2, 1.2, 0.9, 0), _box("c0", "chair", 3, 3, 0, 3.5, 3.5, 0.9, 0),
        _box("b1", "bed", 5.5, 0.5, 0, 7.5, 2.5, 0.5, 1), _box("c1", "chair", 9, 4, 0, 9.5, 4.5, 0.9, 1),
        _box("w1", "wardrobe", 8.0, 0.2, 0, 9.5, 0.8, 2.1, 1),
        _box("b2", "bed", 10.5, 0.5, 0, 12.5, 2.5, 0.5, 2), _box("w2", "wardrobe", 13.0, 0.2, 0, 14.5, 0.8, 2.1, 2)]


class ReferringV3Test(unittest.TestCase):
    def test_only_one_in_its_room_is_just_its_name(self):
        scene = _scene(OBJS, ROOMS)
        got = _assign(scene, {"s", "c0", "c1", "b1", "w1"})
        self.assertEqual(got["s"].ref, "the sofa")                       # unique in the scene
        self.assertEqual(got["s"].ref_method, R.METHOD_UNIQUE)
        for oid, room, outside, others in (("c0", 0, "the chair in the living room", ["c1"]),
                                           ("c1", 1, "the chair in the bedroom", ["c0"]),
                                           ("b1", 1, None, ["b2"]),       # two bedrooms: rooms do not tell
                                           ("w1", 1, None, ["w2"])):
            e = got[oid].target_object_entry()
            self.assertEqual(e["ref"], "the " + R.display_label(got[oid].label), oid)
            self.assertEqual(e["ref_method"], R.METHOD_ROOM_LOCAL, oid)
            self.assertEqual((e["ref_room"], e["ref_outside"], e["ref_confusers"]), (room, outside, others), oid)
            self.assertIsNone(e["ref_anchor"])
        # v2 is unchanged: the room form, then a reference object
        v2 = _assign(scene, {"c0", "b1"}, "v10_r5")
        self.assertEqual(v2["c0"].ref, "the chair in the living room")
        self.assertNotEqual(v2["b1"].ref, "the bed")

    def test_two_in_one_room_still_need_a_reference(self):
        objs = OBJS + [_box("c0b", "chair", 0.2, 3.5, 0, 0.7, 4.0, 0.9, 0)]
        got = _assign(_scene(objs, ROOMS), {"c0"})
        self.assertNotEqual(got["c0"].ref_method, R.METHOD_ROOM_LOCAL)

    def test_room_types_v3(self):
        self.assertEqual(R.infer_room_type_v3(["toilet", "washbasin"]), "bathroom")
        self.assertEqual(R.infer_room_type_v3(["washbasin", "mirror"]), "bathroom")
        # the round-5 "stool in the bathroom": a vanity cabinet in a room with a sofa and a dining table
        self.assertIsNone(R.infer_room_type_v3(["vanity cabinet", "sofa", "dining table", "refrigerator"]))
        self.assertEqual(R.infer_room_type(["vanity cabinet", "sofa", "dining table"]), "bathroom")  # v2
        self.assertIsNone(R.infer_room_type_v3(["toilet", "sofa"]))
        self.assertIsNone(R.infer_room_type_v3(["gas stove", "dining table"]))
        self.assertIsNone(R.infer_room_type_v3(["bed", "dining table"]))
        self.assertEqual(R.infer_room_type_v3(["bed", "desk", "sofa"]), "bedroom")
        self.assertEqual(R.infer_room_type_v3(["sofa", "dining table"]), "living room")
        self.assertIsNone(R.infer_room_type_v3(["chair"]))


class StartContextTest(unittest.TestCase):
    def setUp(self):
        self.scene = _scene(OBJS, ROOMS)
        got = _assign(self.scene, {"s", "c0", "b1"})
        self.item = {"task_type": "projective_relations",
                     "canonical_task_metric_version": C.CANONICAL_TASK_METRIC_V1,
                     "target_region": {"params": {"relation": "left"}},
                     "target_object": {"objects": [got["c0"].target_object_entry(), got["s"].target_object_entry()]}}
        self.item["task_description"] = SC.describe(self.item)
        self.seen = set()
        self.obs = lambda scene, c2w, obj: {"identifiable": obj.id in self.seen}

    def test_start_decides_the_text(self):
        self.assertEqual(self.item["task_description"],
                         "Move so that the chair appears to the left of the sofa in your view.")
        inside = C.pose_from_xy_yaw(2.0, 2.0, 0.0)
        outside = C.pose_from_xy_yaw(6.0, 3.0, 0.0)
        self.assertEqual(SC.start_refs(self.scene, self.item, inside, self.obs),
                         {"c0": {"ref": "the chair", "ref_method": R.METHOD_ROOM_LOCAL}})
        self.seen = {"c1"}  # the other chair can be recognised from the start: name the room
        refs = SC.start_refs(self.scene, self.item, inside, self.obs)
        self.assertEqual(refs["c0"]["ref"], "the chair in the living room")
        self.seen = set()
        refs = SC.start_refs(self.scene, self.item, outside, self.obs)
        self.assertEqual(refs, {"c0": {"ref": "the chair in the living room", "ref_method": R.METHOD_ROOM}})
        row = SC.apply_refs(copy.deepcopy(self.item), refs)
        self.assertEqual(row["task_description"],
                         "Move so that the chair in the living room appears to the left of the sofa in your view.")
        self.assertEqual(row["target_object"]["objects"][0]["ref_method"], R.METHOD_ROOM)
        # a target that has no room form cannot be described from another room
        bed = _assign(self.scene, {"b1"})["b1"].target_object_entry()
        item = dict(self.item, target_object={"objects": [bed, self.item["target_object"]["objects"][1]]})
        self.assertIsNone(SC.start_refs(self.scene, item, inside, self.obs))
        self.assertEqual(SC.start_refs(self.scene, item, C.pose_from_xy_yaw(7.0, 3.5, 0.0), self.obs),
                         {"b1": {"ref": "the bed", "ref_method": R.METHOD_ROOM_LOCAL}})
        self.assertEqual(SC.start_refs(self.scene, {"target_object": {"objects": []}}, inside, self.obs), {})

    def test_oa_target_visible(self):
        item = {"task_type": "occlusion_alignment", "target_region": {"params": {"occluded": "7", "occluder": "8"}}}
        obs = {"targets": {"7": {"identifiable": False}, "8": {"identifiable": True}}}
        self.assertFalse(SC.oa_target_visible(item, obs))
        obs["targets"]["7"]["identifiable"] = True
        self.assertTrue(SC.oa_target_visible(item, obs))
        self.assertTrue(SC.oa_target_visible({"task_type": "screen_occupancy"}, {"targets": {}}))


class RunOptionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.paths = S.write_synthetic_scenes(cls.root / "scenes")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_scene_without_walls_gets_no_tasks(self):
        src = self.paths["two_rooms"]
        dst = self.root / "no_walls"
        dst.mkdir()
        for name in ("labels.json", "_truth.json"):
            (dst / name).write_text((src / name).read_text())
        structure = json.loads((src / "structure.json").read_text())
        structure["walls"] = []
        (dst / "structure.json").write_text(json.dumps(structure))
        out = generate_scene(dst, "no_walls", RunConfig(target_tasks_per_scene=2, require_structure_walls=True))
        self.assertEqual(out["summary"]["status"], "no_structure_walls")
        self.assertEqual(out["rows"], [])

    def test_r6_rows_name_targets_after_the_start(self):
        """Two rooms, a chair in each: every row's text matches its start (room_local only from inside the
        target's room), and the description is the builders' own wording of the entries' refs."""
        labels = [S._box("1", "sofa", 0.3, 1.5, 0.0, 1.2, 3.7, 0.9), S._box("2", "chair", 3.0, 0.6, 0.0, 3.5, 1.1, 0.9),
                  S._box("3", "bed", 7.8, 1.2, 0.0, 9.8, 3.4, 0.6), S._box("4", "chair", 6.6, 3.9, 0.0, 7.1, 4.4, 0.9),
                  S._box("5", "wardrobe", 6.2, 0.2, 0.0, 7.4, 0.8, 2.2), S._box("6", "tv cabinet", 5.3, 1.6, 0.0, 5.8, 3.4, 0.55)]
        labels += S._walls_for_rect(100, 0.0, 0.0, 6.0, 5.0) + S._walls_for_rect(200, 6.0, 0.0, 10.0, 5.0)
        rooms = [("", [(0, 0), (6, 0), (6, 5), (0, 5)]), ("", [(6, 0), (10, 0), (10, 5), (6, 5)])]
        d = S._write(self.root / "r6_chairs", labels, rooms, [[(5.95, 2.0), (6.05, 2.0), (6.05, 3.0), (5.95, 3.0)]],
                     y_sign=-1.0, truth={})
        cfg = RunConfig(object_preset="v10_r6", target_tasks_per_scene=8, visibility_rule="env",
                        families=("projective_relations", "absolute_positioning", "screen_occupancy"),
                        oa_start_target_visible=True)
        out = _jsonable(generate_scene(d, "r6_chairs", cfg))
        rows = out["rows"]
        self.assertTrue(rows, out["summary"])
        from data_gen.interiorgs_v10.scene import load_scene

        scene = load_scene(d, "r6_chairs")
        seen = set()
        for r in rows:
            self.assertEqual(r["task_description"], SC.describe(r))
            pose = np.asarray(r["init_camera"]["extrinsics"], float)
            room = scene.room_of(pose[0, 3], pose[1, 3])
            for e in r["target_object"]["objects"]:
                seen.add(e["ref_method"])
                if e["ref_method"] == R.METHOD_ROOM_LOCAL:
                    self.assertEqual(room, e["ref_room"])
                    self.assertEqual(e["ref"], "the " + R.display_label(e["label"]))
                if e.get("ref_outside") and e["ref_method"] == R.METHOD_ROOM:
                    self.assertEqual(e["ref"], e["ref_outside"])
        self.assertIn(R.METHOD_ROOM_LOCAL, seen)


if __name__ == "__main__":
    unittest.main()
