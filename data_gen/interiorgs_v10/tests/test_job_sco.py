"""job_sco.py: one sco job running a mode of sco_interiorgs_v10_train.sh."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import job_sco as J


class JobScoTest(unittest.TestCase):
    def test_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            launch = J.prepare(run, 1, ["sft-trial", "/lf dir", "30"])
            text = launch.read_text()
            self.assertIn(f"{J.WORKTREE}/{J.SCRIPT} sft-trial {run} '/lf dir' 30", text)
            prepared = json.loads((run / "job" / "prepared.json").read_text())
            self.assertEqual(prepared["spec"], "N4lS.Iq.I80.1")
            self.assertEqual(J.SPECS[8], "N4lS.Iq.I80.8")
            with self.assertRaises(ValueError):
                J.prepare(run, 2, ["sft"])
            with self.assertRaises(ValueError):
                J.prepare(run, 1, [])
            # the render service and the trainer of one PPO run: two job folders on one RUN
            J.prepare(run, 1, ["renderer"], job_dir="job_renderer")
            J.prepare(run, 8, ["rl", "/rl", "150"], job_dir="job_rl")
            self.assertIn(f"rl {run} /rl 150", (run / "job_rl" / "launch.sh").read_text())
            self.assertEqual(json.loads((run / "job_rl" / "prepared.json").read_text())["spec"], "N4lS.Iq.I80.8")
            self.assertTrue((run / "job" / "launch.sh").exists())

    def test_jobs_run_from_the_cluster_checkout(self):
        """Round 6: the cluster checkout is the only code directory (no more VAGEN-Lite-r4)."""
        self.assertEqual(J.WORKTREE, "/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        self.assertIn("\nWORKTREE=/mnt/umm/users/yinbaiqiao/VAGEN-Lite\n", script)
        self.assertNotIn("VAGEN-Lite-r4", script)

    def test_outputs_outside_run_are_handed_back(self):
        """Round 8: the SFT model (LF/model) and evaluation results (EVAL/NAME) were left root-only (mode 600)."""
        script = (Path(__file__).resolve().parents[3] / J.SCRIPT).read_text()
        self.assertIn('chown -R "${OWNER}" "${RUN}" ${ALSO_OWNED:+"${ALSO_OWNED}"}', script)
        for mode, line in (("sft)", "ALSO_OWNED=${LF}"), ("eval)", "ALSO_OWNED=${EVAL}/${NAME}")):
            block = script.split(f"\n  {mode}\n", 1)[1].split(";;", 1)[0]
            self.assertIn(line, block)


if __name__ == "__main__":
    unittest.main()
