"""The depth-map occlusion judge (vagen/envs/active_spatial/occlusion_depth.py) and its runner (occ_judge.py)."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.transition_qa import occ_judge as OJ
from vagen.envs.active_spatial import occlusion_depth as OD

W = H = 256
K = np.array([[128.0, 0, 128.0], [0, 128.0, 128.0], [0, 0, 1]])
C2W = np.eye(4)  # camera at the origin looking along +z (right, down, forward)
A = {"bbox_min": [-0.5, -0.5, 4.0], "bbox_max": [0.5, 0.5, 5.0], "label": "sofa"}
B = {"bbox_min": [-1.0, -1.0, 2.0], "bbox_max": [1.0, 1.0, 2.5], "label": "cabinet"}
ONES = np.ones((H, W))


def run(depth, alpha=ONES, target=A, occluder=B, **params):
    return OD.judge(depth, alpha, K, C2W, target, occluder, params)


class GeometryTest(unittest.TestCase):
    def test_rays_have_unit_depth(self):
        d = OD.pixel_rays(K, C2W, W, H)
        self.assertEqual(d.shape, (W * H, 3))
        self.assertTrue(np.allclose(d[:, 2], 1.0))
        centre = d[128 * W + 128]
        self.assertTrue(np.allclose(centre, [0.5 / 128, 0.5 / 128, 1.0]))

    def test_ray_box(self):
        dirs = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        near, far = OD.ray_box(np.zeros(3), dirs, np.array([-0.5, -0.5, 4.0]), np.array([0.5, 0.5, 5.0]))
        self.assertAlmostEqual(near[0], 4.0)
        self.assertAlmostEqual(far[0], 5.0)
        self.assertTrue(np.isnan(near[1]) and np.isnan(near[2]))  # misses; behind the camera
        near, far = OD.ray_box(np.zeros(3), dirs[:1], np.array([-1.0, -1, -1]), np.array([1.0, 1, 1]))
        self.assertEqual(near[0], 0.0)  # camera inside the box: the entry is clipped at 0
        self.assertAlmostEqual(far[0], 1.0)


class JudgeTest(unittest.TestCase):
    def test_full_none_wall_background(self):
        full = run(np.full((H, W), 2.0))  # B's front face everywhere
        self.assertAlmostEqual(full["hidden_by_b"], 1.0)
        self.assertEqual(full["label"], "full")
        self.assertGreater(full["pixels"], 1000)  # A spans about 32 x 32 pixels at 4 m
        none = run(np.full((H, W), 4.0))  # A's near face
        self.assertEqual((none["label"], none["visible"]), ("none", 1.0))
        wall = run(np.full((H, W), 3.0))  # something between, outside B's box
        self.assertAlmostEqual(wall["hidden_by_other"], 1.0)
        self.assertEqual((wall["label"], wall["no_question"]), (None, "hidden_by_other"))
        far = run(np.full((H, W), 9.0))  # behind A's box: background only
        self.assertEqual((far["pixels"], far["label"], far["no_question"]), (0, None, "too_few_pixels"))

    def test_partial_half_and_half(self):
        depth = np.full((H, W), 4.0)
        depth[:, :128] = 2.0
        r = run(depth)
        self.assertAlmostEqual(r["hidden_by_b"], 0.5, places=2)
        self.assertEqual(r["label"], "partial")

    def test_hollow_box_is_not_hiding(self):
        """B's box covers A, but nothing is rendered in it (an open wardrobe, a person's loose box): A is seen."""
        big = {"bbox_min": [-2.0, -2.0, 1.0], "bbox_max": [2.0, 2.0, 3.5], "label": "wardrobe"}
        r = run(np.full((H, W), 4.2), occluder=big)
        self.assertEqual((r["label"], r["hidden_by_b"]), ("none", 0.0))

    def test_margin_and_thin_labels(self):
        depth = np.full((H, W), 1.97)  # 3 cm in front of B's box
        self.assertEqual(run(depth)["label"], "full")  # widened by 5 cm
        thin = dict(B, label="Monitor")
        r = run(depth, occluder=thin)
        self.assertEqual((r["b_margin"], r["label"]), (0.0, None))  # not widened: hidden by something else
        self.assertAlmostEqual(r["hidden_by_other"], 1.0)

    def test_opacity_and_near_tolerance(self):
        self.assertEqual(run(np.full((H, W), 2.0), alpha=np.full((H, W), 0.2))["pixels"], 0)  # no surface
        fuzz = np.full((H, W), 3.98)  # A's own surface rendered 2 cm in front of its box
        self.assertEqual(run(fuzz)["no_question"], "hidden_by_other")
        self.assertEqual(run(fuzz, near_tol=0.05)["label"], "none")

    def test_label_bands(self):
        m = {"hidden_by_b": 0.8, "hidden_by_other": 0.0, "visible": 0.2, "pixels": 500}
        self.assertEqual(OD.label(m), (None, "between_bands"))
        self.assertEqual(OD.label(dict(m, hidden_by_b=0.70)), ("partial", ""))
        self.assertEqual(OD.label(dict(m, hidden_by_b=0.85)), ("full", ""))
        self.assertEqual(OD.label(dict(m, hidden_by_b=0.02, hidden_by_other=0.2, visible=0.4)), (None, "none_not_visible"))
        self.assertEqual(OD.label(dict(m, hidden_by_b=0.4, hidden_by_other=0.3, visible=0.3)), (None, "hidden_by_other"))
        self.assertEqual(OD.label(dict(m, pixels=99)), (None, "too_few_pixels"))
        self.assertTrue(OD.hidden(dict(m, hidden_by_b=0.85)))
        self.assertFalse(OD.hidden(m))

    def test_overlay(self):
        depth = np.full((H, W), 4.0)
        depth[:, :128] = 2.0
        cls = OD.classify(depth, ONES, K, C2W, A, B)
        img = OD.overlay(np.zeros((H, W, 3), np.uint8), cls)
        self.assertEqual((img.shape, img.dtype), ((H, W, 3), np.uint8))
        self.assertGreater(int(img[128, 120, 0]), int(img[128, 120, 1]))  # hidden by B: red
        self.assertGreater(int(img[128, 136, 1]), int(img[128, 136, 0]))  # A: green


class RunnerTest(unittest.TestCase):
    def pilot(self, tmp):
        p = Path(tmp)
        (p / "maps").mkdir()
        rows = []
        for sid, d, bank in (("s_full", 2.0, "full"), ("s_wrong", 4.0, "full"), ("s_none", 4.0, "none"),
                             ("s_wall", 3.0, "partial")):
            np.savez_compressed(p / "maps" / f"{sid}.npz", depth=np.full((H, W), d, np.float16),
                                alpha=np.full((H, W), 255, np.uint8))
            rows.append({"sample_id": sid, "qtype": "occluded_now", "answer_value": bank, "box_hidden": 0.9,
                         "a": A, "b": B, "K": K.tolist(), "c2w": C2W.tolist()})
        (p / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        verdicts = {"verdicts": [{"sample_id": "s_full", "verdict": "right"}, {"sample_id": "s_wrong", "verdict": "wrong"},
                                 {"sample_id": "s_none", "verdict": "right"}, {"sample_id": "s_wall", "verdict": "wrong",
                                                                               "note": "wall"}]}
        (p / "v.json").write_text(json.dumps(verdicts))
        return p

    def test_agrees(self):
        self.assertTrue(OJ.agrees("right", "full", "full"))
        self.assertFalse(OJ.agrees("right", "full", None))
        self.assertTrue(OJ.agrees("wrong", "full", None))
        self.assertFalse(OJ.agrees("wrong", "partial", "partial"))

    def test_calibrate_and_confusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.pilot(tmp)
            self.assertEqual(OJ.rejudge(p, json.loads((p / "results.jsonl").read_text().splitlines()[0]))["label"], "full")
            out = p / "cal.json"
            with contextlib.redirect_stdout(io.StringIO()):
                OJ.cmd_calibrate(argparse.Namespace(pilot=str(p), verdicts=[str(p / "v.json")], first_check="", set=[],
                                                    out=str(out)))
            s = json.loads(out.read_text())["summary"]
            self.assertEqual((s["checked"], s["box judge agrees"], s["depth judge agrees"]), (4, 2, 4))
            with contextlib.redirect_stdout(io.StringIO()):
                OJ.cmd_confusion(argparse.Namespace(pilot=str(p), set=["near_tol=0.0"], out=str(p / "conf.json")))
            t = json.loads((p / "conf.json").read_text())["tables"]["all"]
            self.assertEqual(t["full"], {"完全": 1, "部分": 0, "没有": 1, "不出题": 0})
            self.assertEqual(t["partial"]["不出题"], 1)
            self.assertEqual(len((p / "conf.labels.jsonl").read_text().splitlines()), 4)
            with self.assertRaises(SystemExit):
                OJ.parse_sets(["nonsense=1"])


if __name__ == "__main__":
    unittest.main()
