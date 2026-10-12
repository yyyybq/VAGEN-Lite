"""Ignored objects (confirmed annotation errors): generator side and the env patch."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10.patches import make_env_ignored_objects_patch as P
from data_gen.interiorgs_v10.run_scene import RunConfig, _task_row, ignored_object_ids
from data_gen.interiorgs_v10.sampling import projective_item
from data_gen.interiorgs_v10.scene import load_scene
from data_gen.interiorgs_v10.testing import synthetic_scene as S

REPO = Path(__file__).resolve().parents[3]
BIG = "77"  # one "cabinet" box over the whole room, as in 0127_840030


def _room_with_bogus_cabinet(root: Path) -> Path:
    labels = [
        S._box(BIG, "cabinet", 0.2, 0.2, 0.0, 5.8, 4.8, 2.5),
        S._box("1", "sofa", 0.6, 3.2, 0.0, 2.0, 3.7, 0.85),
        S._box("2", "dining table", 3.6, 1.0, 0.0, 4.6, 1.8, 0.75),
        S._box("3", "wardrobe", 5.0, 3.0, 0.0, 5.6, 4.6, 2.2),
    ] + S._walls_for_rect(100, 0.0, 0.0, 6.0, 5.0)
    return S._write(root / "bogus_cabinet", labels, [("living room", [(0, 0), (6, 0), (6, 5), (0, 5)])], [],
                    y_sign=-1.0, truth={})


class GeneratorSideTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.dir = _room_with_bogus_cabinet(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_ignored_box_leaves_collision_objects_and_occluders(self):
        from data_gen.interiorgs_v10 import occlusion as O

        full = load_scene(self.dir, "bogus_cabinet")
        fixed = load_scene(self.dir, "bogus_cabinet", ignored_ids=[BIG, "not_in_scene"])
        self.assertEqual(fixed.ignored_object_ids, [BIG])
        self.assertIn(BIG, {o.id for o in full.objects})
        self.assertNotIn(BIG, {o.id for o in fixed.objects})
        self.assertNotIn(BIG, {b.ins_id for b in fixed.detector.object_boxes})
        self.assertEqual(len(fixed.detector.object_boxes), len(full.detector.object_boxes) - 1)
        self.assertIn(BIG, set(O.occluders_for(full).ids))
        self.assertNotIn(BIG, set(O.occluders_for(fixed).ids))
        grid = np.stack(np.meshgrid(np.arange(0.6, 5.5, 0.3), np.arange(0.6, 4.5, 0.3)), -1).reshape(-1, 2)
        self.assertFalse(full.legal_mask(grid).any())  # the bogus box blocks every position
        self.assertTrue(fixed.legal_mask(grid).sum() > 20)
        self.assertEqual(full.convention_v2.get("sign"), fixed.convention_v2.get("sign"))

    def test_rows_carry_the_ids_for_the_env(self):
        fixed = load_scene(self.dir, "bogus_cabinet", ignored_ids=[BIG])
        o = {x.id: x for x in fixed.objects}
        item = projective_item("bogus_cabinet", o["1"], o["2"], "left")
        ep = {"init_c2w": C.pose_from_xy_yaw(3.0, 2.5, 0.0).tolist(), "difficulty": {"d_star": 1}}
        self.assertEqual(_task_row(fixed, item, ep)["collision_convention"]["ignored_object_ids"], [BIG])
        plain = load_scene(self.dir, "bogus_cabinet")
        self.assertNotIn("ignored_object_ids", _task_row(plain, item, ep)["collision_convention"])

    def test_list_file_uses_confirmed_entries_only(self):
        path = self.root / "ignored.json"
        path.write_text(json.dumps({"version": "interiorgs_ignored_objects_v1",
                                    "scenes": {"bogus_cabinet": [{"ins_id": BIG, "reason": "covers the room"}]},
                                    "uncertain": {"bogus_cabinet": [{"ins_id": "3", "reason": "?"}]}}))
        self.assertEqual(ignored_object_ids(str(path), "bogus_cabinet"), [BIG])
        self.assertEqual(ignored_object_ids(str(path), "other_scene"), [])
        self.assertEqual(ignored_object_ids(None, "bogus_cabinet"), [])
        self.assertIsNone(RunConfig().ignored_objects)


_PROBE = r"""
import json, sys
import numpy as np
root, scene_dir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
import vagen
assert vagen.__file__.startswith(root), vagen.__file__
from pathlib import Path
from vagen.envs.active_spatial.collision_detector import CollisionDetector
from vagen.envs.active_spatial.visibility_checker import VisibilityChecker
sid = "bogus_cabinet"
if mode == "ignore":
    det = CollisionDetector(ignored_object_ids={sid: ["77"]})
else:
    det = CollisionDetector()
det.load_scene(Path(scene_dir), scene_id=sid)
if mode == "drop_after_load":  # unpatched reference: the same detector with only that one box removed
    det.object_boxes = [b for b in det.object_boxes if str(b.ins_id) != "77"]
vis = VisibilityChecker()
if mode == "ignore":
    vis.ignored_object_ids[sid] = {"77"}
vis.load_scene(Path(scene_dir), scene_id=sid)
rng = np.random.default_rng(0)
out = []
for _ in range(400):
    a = np.array([rng.uniform(0.3, 5.7), rng.uniform(0.3, 4.7), 1.5])
    step = rng.normal(size=2); step = 0.3 * step / np.linalg.norm(step)
    b = a + np.array([step[0], step[1], 0.0])
    r = det.check_collision(b, previous_position=a)
    out.append([bool(r.has_collision), r.collision_type, getattr(r, "collision_object", None)])
print(json.dumps({"results": out, "n_boxes": len(det.object_boxes), "n_occluders": len(vis.occluder_boxes),
                  "boundary": None if det.room_boundary is None else
                  [det.room_boundary.min_point.tolist(), det.room_boundary.max_point.tolist()]}))
"""


class EnvPatchTest(unittest.TestCase):
    def test_patch_up_to_date_applies_and_only_drops_the_listed_box(self):
        if shutil.which("patch") is None:
            self.skipTest("patch binary not available")
        self.assertEqual(P.PATCH_PATH.read_text(), P.build(), "regenerate with make_env_ignored_objects_patch.py")
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            scene_dir = _room_with_bogus_cabinet(tmp)
            env = tmp / "env"
            shutil.copytree(REPO / "vagen", env / "vagen", ignore=shutil.ignore_patterns("__pycache__"))
            for patch in (P.BASE_PATCH, P.PATCH_PATH):
                for extra in (["--dry-run"], []):
                    res = subprocess.run(["patch", "-p1", "--batch", "--forward"] + extra, cwd=env,
                                         input=patch.read_bytes(), capture_output=True)
                    self.assertEqual(res.returncode, 0, patch.name + res.stdout.decode() + res.stderr.decode())

            def probe(root: Path, mode: str):
                res = subprocess.run([sys.executable, "-c", _PROBE, str(root), str(scene_dir), mode],
                                     capture_output=True, text=True, cwd=str(tmp))
                self.assertEqual(res.returncode, 0, res.stderr)
                return json.loads(res.stdout.strip().splitlines()[-1])

            original = probe(REPO, "plain")
            patched_plain = probe(env, "plain")
            patched_ignore = probe(env, "ignore")
            reference = probe(REPO, "drop_after_load")
            # the patch alone changes nothing
            self.assertEqual(patched_plain, original)
            # ignoring 77 == the unpatched detector with exactly that box removed (boundary kept)
            self.assertEqual(patched_ignore["results"], reference["results"])
            self.assertEqual(patched_ignore["boundary"], original["boundary"])
            self.assertEqual(patched_ignore["n_boxes"], original["n_boxes"] - 1)
            self.assertEqual(patched_ignore["n_occluders"], original["n_occluders"] - 1)
            blocked = [r for r in original["results"] if r[0]]
            freed = sum(1 for a, b in zip(original["results"], patched_ignore["results"]) if a[0] and not b[0])
            self.assertGreater(len(blocked), 300)  # the bogus box blocks almost every move
            self.assertGreater(freed, 200)
            self.assertFalse(any(r[2] == BIG for r in patched_ignore["results"]))


if __name__ == "__main__":
    unittest.main()
