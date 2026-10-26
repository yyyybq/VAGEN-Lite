"""fetch_interiorgs_metadata.aoss_jobs: which files of which scenes are fetched."""

import unittest

from data_gen.interiorgs_v10.fetch_interiorgs_metadata import AOSS_FILES, aoss_jobs


class FetchJobsTest(unittest.TestCase):
    def test_default_and_selected(self):
        scenes = ["0001_a", "0002_b", "0003_c"]
        self.assertEqual(len(aoss_jobs(scenes)), 3 * len(AOSS_FILES))
        jobs = aoss_jobs(scenes, ["labels.json", "3dgs_compressed.ply"], only=["0002_b"])
        self.assertEqual(jobs, ["0002_b/labels.json", "0002_b/3dgs_compressed.ply"])
        with self.assertRaises(ValueError):
            aoss_jobs(scenes, None, only=["9999_x"])


if __name__ == "__main__":
    unittest.main()
