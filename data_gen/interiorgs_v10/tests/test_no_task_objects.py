"""No-task object list (user decision 2026-10-13): listed boxes keep collision and occlusion but are never a
task target or a reference object (anchor of "next to", other object of a size comparison)."""

import re
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10.objects import candidate_objects
from data_gen.interiorgs_v10.occlusion import occluders_for
from data_gen.interiorgs_v10.referring import Resolver
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes

_ANCHOR = re.compile(r"^the (.+?) next to the (.+)$")
_SIZE = re.compile(r"^the (larger|smaller|taller|shorter) (.+)$")


def _referenced_ids(scene, kept):
    """{target id: ids of every other object its description refers to}."""
    resolver = Resolver(scene, list(scene.objects))
    out = {}
    for o in kept:
        ids = set()
        m = _ANCHOR.match(o.ref or "")
        if m:
            ids |= {a.id for a in resolver("the " + m.group(2))}
        m = _SIZE.match(o.ref or "")
        if m:
            ids |= {a.id for a in resolver("the " + m.group(2))} - {o.id}
        out[o.id] = ids
    return out


class NoTaskObjectsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = write_synthetic_scenes(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_listed_objects_keep_boxes_but_are_never_target_or_reference(self):
        checked_reference = False
        for name, sdir in sorted(self.paths.items()):
            free = load_scene(sdir, name)
            kept, _ = candidate_objects(free)
            if not kept:
                continue
            refs = _referenced_ids(free, kept)
            # list every candidate target and every object some description refers to
            listed = sorted({o.id for o in kept[:2]} | {i for ids in refs.values() for i in ids})
            scene = load_scene(sdir, name, no_task_ids=listed + ["no_such_id"])
            self.assertEqual(scene.no_task_object_ids, sorted(listed))
            self.assertEqual({o.id for o in scene.objects}, {o.id for o in free.objects})
            self.assertEqual(len(scene.detector.object_boxes), len(free.detector.object_boxes))  # still collide
            self.assertEqual(set(occluders_for(scene).ids), set(occluders_for(free).ids))  # still occlude
            kept2, funnel = candidate_objects(scene)
            self.assertFalse({o.id for o in kept2} & set(listed), name)
            self.assertEqual(funnel.counts["objects"]["reject_no_task_list"], len(listed))
            for oid, ids in _referenced_ids(scene, kept2).items():
                self.assertFalse(ids & set(listed), (name, oid))
            checked_reference |= any(refs.values())
        self.assertTrue(checked_reference, "no synthetic scene exercises anchor / size references")


if __name__ == "__main__":
    unittest.main()
