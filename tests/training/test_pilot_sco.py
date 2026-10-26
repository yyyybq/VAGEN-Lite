"""pilot_sco.py: the frozen source carries the repo PLY loader; delete removes the job and retires its receipt."""

import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_gen.interiorgs_v10 import pilot_sco as S


class PilotScoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.pilot = self.tmp / "pilot"
        self.pilot.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_prepare_freezes_the_repo_ply_loader(self):
        repo = self.tmp / "repo"
        for rel in ("data_gen/interiorgs_v10/pilot.py", "vagen/__init__.py", "ply_gaussian_loader.py"):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text("# " + rel)
        (repo / "vagen/__pycache__").mkdir()
        (repo / "vagen/__pycache__/x.pyc").write_text("")
        with mock.patch.object(S, "ROOT", repo):
            S.prepare(self.pilot, "/gs")
        with tarfile.open(self.pilot / "job" / "source.tar.gz") as tar:
            names = set(tar.getnames())
        self.assertIn("ply_gaussian_loader.py", names)  # renderer must not fall back to the slow env copy
        self.assertIn("data_gen/interiorgs_v10/pilot.py", names)
        self.assertFalse(any("__pycache__" in n for n in names))
        launch = (self.pilot / "job" / "launch.sh").read_text()
        self.assertIn('export PYTHONPATH="$SRC"', launch)
        self.assertIn("--gs-root /gs", launch)

    def test_delete_removes_a_suspended_job_and_retires_the_receipt(self):
        job = self.pilot / "job"
        job.mkdir()
        (job / "submission.json").write_text(json.dumps({"job_id": "pt-abc123"}))
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, "deleted", "")

        with mock.patch.object(S, "describe", return_value={"state": "SUSPENDED"}), \
                mock.patch.object(S.subprocess, "run", side_effect=fake_run):
            S.delete(self.pilot)
        self.assertEqual([c[2:4] for c in calls], [["jobs", "delete"]])  # suspended: no stop first
        self.assertFalse((job / "submission.json").exists())  # a later submit creates a new job
        self.assertTrue((job / "submission_pt-abc123_deleted.json").exists())

    def test_delete_stops_a_running_job_first_and_keeps_the_receipt_on_failure(self):
        job = self.pilot / "job"
        job.mkdir()
        (job / "submission.json").write_text(json.dumps({"job_id": "pt-abc123"}))
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 1 if cmd[3] == "delete" else 0, "", "boom")

        with mock.patch.object(S, "describe", return_value={"state": "RUNNING"}), \
                mock.patch.object(S.subprocess, "run", side_effect=fake_run):
            S.delete(self.pilot)
        self.assertEqual([c[3] for c in calls], ["stop", "delete"])
        self.assertTrue((job / "submission.json").exists())


if __name__ == "__main__":
    unittest.main()
