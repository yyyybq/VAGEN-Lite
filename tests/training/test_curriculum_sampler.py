"""Round 12 batch: the easy-first training-task order (vagen/curriculum_sampler.py)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HAVE_VERL = all(importlib.util.find_spec(m) is not None for m in ("torch", "omegaconf", "verl", "tensordict"))


@unittest.skipUnless(HAVE_VERL, "needs the training env (torch, omegaconf, verl)")
class CurriculumSamplerTest(unittest.TestCase):
    def test_order(self):
        from vagen.curriculum_sampler import curriculum_order

        easy = [1, 3, 5, 7, 9, 11]
        first = curriculum_order(12, easy, 4, seed=5, epoch=0)
        self.assertEqual(sorted(first), list(range(12)))  # every task once per epoch
        self.assertTrue(set(first[:4]) <= set(easy))      # the first draws are easy tasks only
        self.assertEqual(first, curriculum_order(12, easy, 4, seed=5, epoch=0))  # fixed by seed and epoch
        self.assertNotEqual(first, curriculum_order(12, easy, 4, seed=6, epoch=0))
        later = curriculum_order(12, easy, 4, seed=5, epoch=1)
        self.assertEqual(sorted(later), list(range(12)))
        self.assertEqual(curriculum_order(5, [], 0, seed=1, epoch=0), curriculum_order(5, [], 0, seed=1, epoch=0))
        with self.assertRaises(ValueError):
            curriculum_order(12, easy, 7, seed=5, epoch=0)

    def test_sampler_reads_dstar_from_the_task_rows(self):
        from omegaconf import OmegaConf

        from vagen.curriculum_sampler import DifficultyCurriculumSampler

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            dstars = [1, 7, 2, 8, 3, 6, 1, 5]
            path.write_text("".join(json.dumps({"task_id": f"t{i}", "difficulty": {"d_star": d}}) + "\n"
                                    for i, d in enumerate(dstars)))

            class Data:
                items = [{"seed": i, "config": {"jsonl_path": str(path)}} for i in range(len(dstars))]

                def __len__(self):
                    return len(self.items)

            cfg = OmegaConf.create({"seed": 3, "train_batch_size": 2,
                                    "sampler": {"easy_updates": 2, "max_dstar": 3, "class_path": "x", "class_name": "y"}})
            s = DifficultyCurriculumSampler(Data(), cfg)
            self.assertEqual(s.easy, [0, 2, 4, 6])
            self.assertEqual(len(s), 8)
            order = list(iter(s))
            self.assertTrue(set(order[:4]) <= {0, 2, 4, 6})  # 2 updates x 2 tasks, all easy
            self.assertEqual(sorted(order), list(range(8)))
            self.assertEqual(sorted(list(iter(s))), list(range(8)))  # next epoch: all tasks


if __name__ == "__main__":
    unittest.main()
