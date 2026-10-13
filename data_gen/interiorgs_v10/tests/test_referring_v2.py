"""Referring rules v2 (referring.py, decisions of 2026-10-14): names only when needed, rooms inferred from their
furniture, reference objects large, unique in their room and apart from the target."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_v10 import objects as OB
from data_gen.interiorgs_v10 import referring as R
from data_gen.interiorgs_v10.contract import Obj


def _box(i, label, x0, y0, z0, x1, y1, z1, room=None):
    return Obj(id=str(i), label=label, aabb_min=np.array([x0, y0, z0], float),
               aabb_max=np.array([x1, y1, z1], float), room_index=room)


def _room(i, x0, y0, x1, y1):  # InteriorGS rooms have no name
    return SimpleNamespace(index=i, name="",
                           polygon=np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))


def _scene(objs, rooms):
    return SimpleNamespace(scene_id="r5", scene_dir=None, objects=objs, rooms=rooms, convention={})


def _assign(scene, ids):
    cfg = OB.get_preset("v10_r5")
    objs = [copy.copy(o) for o in scene.objects if o.id in ids]
    R.assign_referring_expressions(scene, objs, cfg.referring, pool=scene.objects,
                                   anchor_ok=lambda a: OB.semantic_exclusion_reason(a.label, cfg) is None)
    return {o.id: o for o in objs}


class ReferringV2Test(unittest.TestCase):
    def test_name_only_when_unique(self):
        scene = _scene([_box("s", "sofa", 0, 0, 0, 2, 1, 0.9, 0), _box("t", "coffee table", 0, 1.5, 0, 1, 2, 0.4, 0),
                        _box("b", "bed", 6, 0, 0, 8, 2, 0.5, 1)],
                       [_room(0, 0, 0, 5, 5), _room(1, 5, 0, 10, 5)])
        got = _assign(scene, {"s", "t"})
        self.assertEqual(got["s"].ref, "the sofa")
        self.assertEqual(got["s"].target_object_entry()["ref_method"], R.METHOD_UNIQUE)
        self.assertIsNone(got["s"].target_object_entry()["ref_anchor"])

    def test_room_inferred_from_furniture(self):
        self.assertEqual(R.infer_room_type(["bed", "wardrobe"]), "bedroom")
        self.assertEqual(R.infer_room_type(["closestool", "washbasin"]), "bathroom")
        self.assertEqual(R.infer_room_type(["sofa", "dining table", "TV"]), "living room")
        self.assertEqual(R.infer_room_type(["dining table", "chair"]), "dining room")
        self.assertIsNone(R.infer_room_type(["bed", "closestool"]))  # mixed
        self.assertIsNone(R.infer_room_type(["gas stove", "sofa"]))  # open-plan kitchen and living room
        self.assertIsNone(R.infer_room_type(["pet bed", "chair"]))
        rooms = [_room(0, 0, 0, 5, 5), _room(1, 5, 0, 10, 5)]
        objs = [_box("s", "sofa", 0, 0, 0, 2, 1, 0.9, 0), _box("c1", "chair", 3, 3, 0, 3.5, 3.5, 0.9, 0),
                _box("b", "bed", 6, 0, 0, 8, 2, 0.5, 1), _box("c2", "chair", 9, 4, 0, 9.5, 4.5, 0.9, 1)]
        got = _assign(_scene(objs, rooms), {"c1", "c2"})
        self.assertEqual(got["c1"].ref, "the chair in the living room")
        self.assertEqual(got["c2"].ref, "the chair in the bedroom")
        self.assertEqual(got["c2"].ref_extra["ref_method"], R.METHOD_ROOM)
        # two bedrooms: the room no longer tells the chairs apart
        rooms3 = rooms + [_room(2, 10, 0, 15, 5)]
        objs3 = objs + [_box("b2", "bed", 11, 0, 0, 13, 2, 0.5, 2), _box("c3", "chair", 14, 4, 0, 14.5, 4.5, 0.9, 2)]
        got3 = _assign(_scene(objs3, rooms3), {"c2"})
        self.assertNotIn("in the bedroom", got3["c2"].ref or "")

    def test_reference_rules(self):
        rooms = [_room(0, 0, 0, 5, 5), _room(1, 5, 0, 10, 5), _room(2, 10, 0, 15, 5)]
        objs = [
            # bedroom 1: a bed with a mattress on it, a clock, a wardrobe next to the bed
            _box("b1", "bed", 0.5, 0.5, 0, 2.5, 2.5, 0.5, 0), _box("m1", "mattress", 0.6, 0.6, 0.5, 2.4, 2.4, 0.7, 0),
            _box("k1", "clock", 2.7, 0.5, 1.4, 3.0, 0.55, 1.7, 0), _box("w1", "wardrobe", 3.0, 0.5, 0, 4.0, 1.1, 2.1, 0),
            # bedroom 2: a bed and a wardrobe far from it
            _box("b2", "bed", 5.5, 0.5, 0, 7.5, 2.5, 0.5, 1), _box("w2", "wardrobe", 9.0, 4.0, 0, 9.9, 4.9, 2.1, 1),
            # bedroom 3: a bed with a floor lamp next to it
            _box("b3", "bed", 10.5, 0.5, 0, 12.5, 2.5, 0.5, 2), _box("f3", "floor lamp", 12.7, 0.6, 0, 13.1, 1.0, 1.6, 2),
        ]
        got = _assign(_scene(objs, rooms), {"b1", "b2", "b3"})
        # b1: not the mattress (on it), not the clock (too small); the wardrobe is unique in its room and
        # "the bed next to the wardrobe" picks b1 only (b2's wardrobe is 1.5 m+ away from b2)
        self.assertEqual(got["b1"].ref, "the bed next to the wardrobe")
        anchor = got["b1"].target_object_entry()["ref_anchor"]
        self.assertEqual(anchor["id"], "w1")
        self.assertEqual(anchor["bbox_max"], [4.0, 1.1, 2.1])
        self.assertEqual(got["b3"].ref, "the bed next to the floor lamp")
        self.assertIsNone(got["b2"].ref)  # nothing usable next to it, three beds: no size comparison
        # if the second wardrobe also stood next to its bed, "the bed next to the wardrobe" would be ambiguous
        objs[5] = _box("w2", "wardrobe", 7.6, 0.5, 0, 8.6, 1.1, 2.1, 1)
        got = _assign(_scene(objs, rooms), {"b1", "b2"})
        self.assertNotEqual(got["b1"].ref, "the bed next to the wardrobe")
        self.assertNotEqual(got["b2"].ref, "the bed next to the wardrobe")

    def test_v1_entries_are_unchanged(self):
        rooms = [_room(0, 0, 0, 5, 5)]
        objs = [_box("s", "sofa", 0, 0, 0, 2, 1, 0.9, 0)]
        scene = _scene(objs, rooms)
        o = copy.copy(objs[0])
        R.assign_referring_expressions(scene, [o], OB.get_preset("v10_relaxed").referring, pool=objs)
        self.assertEqual(set(o.target_object_entry()), {"id", "label", "bbox_min", "bbox_max", "center", "ref"})

    def test_preset_excludes_unusable_labels(self):
        self.assertEqual(OB.semantic_exclusion_reason("edge combination", "v10_r5"), "blacklist")
        self.assertEqual(OB.semantic_exclusion_reason("wall panels", "v10_r5"), "blacklist")
        self.assertIsNone(OB.semantic_exclusion_reason("edge combination", "v10_relaxed"))



if __name__ == "__main__":
    unittest.main()
