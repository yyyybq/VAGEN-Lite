"""vagen/interiorgs_v10_ppo.py: which PPO checkpoints are kept (decision of 2026-10-14)."""

import importlib.util
import unittest

HAVE_RAY = importlib.util.find_spec("ray") is not None and importlib.util.find_spec("omegaconf") is not None


@unittest.skipUnless(HAVE_RAY, "needs ray and omegaconf (the training env)")
class RetainedStepsTest(unittest.TestCase):
    def test_every_50th_and_the_latest_two_others(self):
        from vagen.interiorgs_v10_ppo import retained_steps

        saved = list(range(25, 151, 25))  # a checkpoint every 25 updates up to 150
        self.assertEqual(retained_steps(saved, 50, 2), {50, 100, 150, 75, 125})
        self.assertEqual(retained_steps(saved[:3], 50, 2), {25, 50, 75})
        self.assertEqual(retained_steps(range(25, 251, 25), 50, 2), {50, 100, 150, 200, 250, 175, 225})
        self.assertEqual(retained_steps([1, 2, 3], 2, 1), {2, 3})  # the one-card resume check
        self.assertEqual(retained_steps([1, 2, 3], 0, 1), {3})


if __name__ == "__main__":
    unittest.main()
