"""reward/miss_shift.py: the key packing, the neighbour-cell fallback and the recorded lookups on a synthetic table,
and the summary's under- / over-paid sums and advantage sign flips.  A temporary directory only (no real paths)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_gen.interiorgs_v10 import distance_table as DT
from data_gen.interiorgs_v10.difficulty import _split, contract_keys
from reward import miss_shift as MS


def pose(x, y):
    c = np.eye(4)
    c[0, 3], c[1, 3], c[2, 3] = x, y, 1.5
    return c


def key(c2w):
    return int(contract_keys(*_split(c2w))[0])


class MissShiftTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        k = key(pose(1.234, -0.567))
        self.kx, self.ky, self.kyaw = MS.unpack(k)
        self.assertEqual(MS.pack(self.kx, self.ky, self.kyaw), k)
        cells = {MS.pack(self.kx + 1, self.ky, self.kyaw): 3,  # next to the missing pose's cell
                 MS.pack(self.kx, self.ky - 1, self.kyaw + 1): 5,  # next to it too
                 MS.pack(self.kx + 5, self.ky, self.kyaw): 1}  # in the table, not next to it
        keys = np.array(sorted(cells), dtype=np.int64)
        DT.save(self.root / "t1.npz", {"keys": keys, "dist": np.array([cells[int(k)] for k in keys], np.int32),
                                       "depth_of": np.zeros(len(keys), np.int32), "task_id": "t1"})
        self.item = {"task_id": "t1", "difficulty": {"d_star": 4}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_fallback_takes_the_smallest_neighbour(self):
        own, fb = MS.make_cache(self.root, False), MS.make_cache(self.root, True)
        p = pose(1.234, -0.567)
        self.assertEqual(own.progress(self.item, p, False)["optimal_progress"], 0.0)
        self.assertAlmostEqual(fb.progress(self.item, p, False)["optimal_progress"], 1 - 3 / 8)
        for c in (own, fb):
            self.assertEqual(c.events[-1], {"miss": True, "success": False, "kind": "neighbour", "d_nb": 3, "spread": 2,
                                            "phi_nb": 1 - 3 / 8})

    def test_present_key_and_success_unchanged(self):
        own, fb = MS.make_cache(self.root, False), MS.make_cache(self.root, True)
        p = pose((self.kx + 5) * 0.01, -0.567)
        self.assertEqual(key(p), MS.pack(self.kx + 5, self.ky, self.kyaw))
        for c in (own, fb):
            self.assertAlmostEqual(c.progress(self.item, p, False)["optimal_progress"], 1 - 1 / 8)
            self.assertFalse(c.events[-1]["miss"])
            self.assertEqual(c.progress(self.item, pose(1.234, -0.567), True)["optimal_progress"], 1.0)
            self.assertTrue(c.events[-1]["miss"] and c.events[-1]["success"])

    def test_no_neighbour(self):
        fb = MS.make_cache(self.root, True)
        self.assertEqual(fb.progress(self.item, pose(3.0, 3.0), False)["optimal_progress"], 0.0)
        self.assertEqual(fb.events[-1]["kind"], "none")

    def test_summary(self):
        def ep(own, fb, kept, under=0.0, over=0.0, misses=0):
            return {"logged": own, "kept": kept, "success": False, "return_own": own, "return_fb": fb, "turns": 4,
                    "same_turns": True, "lookups": 10, "misses": misses, "misses_not_success": misses,
                    "misses_phi_nb_pos": misses,
                    "kinds": {"neighbour": misses} if misses else {}, "spread_pos": 0, "under_paid": under,
                    "over_paid": over, "turns_changed": int(bool(under or over)), "progress_abs": 2.0,
                    "reward_abs": 3.0}

        res = [{"update": 26, "task_id": "t1", "task_type": "fov_inclusion",
                "episodes": [ep(1.0, 1.0, True), ep(0.0, 0.0, True), ep(0.4, 0.9, True, under=0.6, over=0.1, misses=2),
                             ep(0.2, 0.2, False)]}]
        s = MS.summarize(res)
        self.assertEqual((s["lookups"], s["misses"], s["miss_pct"], s["misses_fallback_phi_pos"]), (40, 2, 5.0, 2))
        self.assertEqual((s["reward"]["under_paid"], s["reward"]["over_paid"]), (0.6, 0.1))
        self.assertEqual(s["reward"]["shift_pct_of_progress"], 8.75)
        self.assertEqual(s["returns"]["episodes_changed"], 1)
        self.assertEqual(s["kept"]["advantage_sign_flips"], 1)  # the third episode: below the mean, then above it
        self.assertEqual(s["by_block"]["26-50"]["misses"], 2)


if __name__ == "__main__":
    unittest.main()
