"""The open-loop check's helpers (model_matrix/downstream/openloop.py, the user's decision 34 of 2026-10-30)."""
import unittest

import numpy as np


class OpenLoopTest(unittest.TestCase):
    def test_ways(self):
        from PIL import Image

        from model_matrix.downstream import openloop as O

        a = [Image.new("RGB", (4, 3), (200, 10, 10)), np.full((2, 2, 3), 7, dtype=np.uint8)]
        b = [Image.new("RGB", (4, 3), (10, 200, 10)), np.full((2, 2, 3), 9, dtype=np.uint8)]
        batch = [{"image": a, "lang": "x", "action": np.zeros((2, 3))}, {"image": b, "lang": "y", "action": np.ones((2, 3))}]
        self.assertIs(O.with_images(batch, "images"), batch)
        black = O.with_images(batch, "black")
        self.assertEqual(black[0]["image"][0].size, (4, 3))
        self.assertEqual(black[0]["image"][0].getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(black[1]["image"][1].shape, (2, 2, 3))
        self.assertEqual(int(black[1]["image"][1].max()), 0)
        self.assertEqual(batch[0]["image"][0].getpixel((0, 0)), (200, 10, 10))  # the originals untouched
        self.assertEqual(black[0]["lang"], "x")
        shuffled = O.with_images(batch, "shuffled")
        self.assertIs(shuffled[0]["image"], b)
        self.assertIs(shuffled[1]["image"], a)
        self.assertEqual(shuffled[0]["lang"], "x")  # only the images move
        self.assertIs(shuffled[0]["action"], batch[0]["action"])
        with self.assertRaises(ValueError):
            O.with_images(batch[:1], "shuffled")
        with self.assertRaises(ValueError):
            O.with_images(batch, "noise")
        single = O.with_images([{"image": Image.new("L", (2, 2), 5)}], "black")  # one image, not a list
        self.assertEqual(single[0]["image"].getpixel((0, 0)), 0)

    def test_positions(self):
        from model_matrix.downstream import openloop as O

        p = O.positions(1000, 16, 1)
        self.assertEqual(p, sorted(set(p)))
        self.assertEqual(len(p), 16)
        self.assertEqual(p, O.positions(1000, 16, 1))  # the same in every invocation
        self.assertNotEqual(p, O.positions(1000, 16, 2))
        with self.assertRaises(ValueError):
            O.positions(10, 11, 1)

    def test_eval_data(self):
        import json
        import tempfile
        from pathlib import Path

        from model_matrix.downstream import openloop as O

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            spec = [["CloseFridge/lerobot", 1.0, "panda_omron_robocasa365"]]
            (folder / "mixture_spec.json").write_text(json.dumps(spec))
            with self.assertRaises(ValueError):  # no train/ yet
                O.eval_data(folder)
            (folder / "train").mkdir()
            self.assertEqual(O.eval_data(folder), (str(folder / "train"), spec))
            (folder / "mixture_spec.json").write_text(json.dumps([["CloseFridge/lerobot", 1.0]]))
            with self.assertRaises(ValueError):
                O.eval_data(folder)

    def test_errors(self):
        from model_matrix.downstream import openloop as O

        images, black = O.Errors(), O.Errors()
        t = np.zeros((2, 4, 3))
        images.add(np.full((2, 4, 3), 0.1), t)
        images.add(np.full((2, 4, 3), 0.3), t)
        black.add(np.full((2, 4, 3), 0.5), t)
        black.add(np.full((2, 4, 3), 0.5), t)
        s = O.summarize({"images": images, "black": black})
        self.assertAlmostEqual(s["images"]["mse"], (0.01 + 0.09) / 2)
        self.assertEqual((s["images"]["values"], s["images"]["batches"]), (48, 2))
        self.assertAlmostEqual(s["images"]["mse_score"], np.mean([np.sqrt(24 * 0.01) / 24, np.sqrt(24 * 0.09) / 24]))
        self.assertAlmostEqual(s["black"]["ratio_to_images"], 0.25 / 0.05)
        self.assertNotIn("ratio_to_images", s["images"])
        with self.assertRaises(ValueError):
            images.add(np.zeros((2, 4, 2)), t)


if __name__ == "__main__":
    unittest.main()
