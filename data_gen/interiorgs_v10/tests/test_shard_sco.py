"""shard_sco.py: launch scripts for sharded generation jobs; delete removes every job and retires the receipt."""

import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_gen.interiorgs_v10 import shard_sco as S


class ShardScoTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_prepare_writes_one_launch_per_job_shard(self):
        repo = self.tmp / "repo"
        for rel in ("data_gen/interiorgs_v10/run_all.py", "vagen/__init__.py"):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text("# " + rel)
        out = self.tmp / "run"
        with mock.patch.object(S, "ROOT", repo):
            scripts = S.prepare(out, "data_gen/interiorgs_v10/configs/cfg.json", 5, [1, 2, 3, 4], workers=13)
        self.assertEqual(sorted(scripts), ["1", "2", "3", "4"])
        text = Path(scripts["3"]).read_text()
        self.assertIn("--shard 3 --num-shards 5", text)
        self.assertIn("--config data_gen/interiorgs_v10/configs/cfg.json", text)
        self.assertIn('--out "$OUT/output"', text)
        self.assertIn("chown -R", text)  # outputs handed back to the submitting user
        with tarfile.open(out / "job" / "source.tar.gz") as tar:
            self.assertIn("data_gen/interiorgs_v10/run_all.py", tar.getnames())
        with self.assertRaises(ValueError):
            S.prepare(out, "cfg.json", 2, [2])

    def test_delete(self):
        out = self.tmp / "run"
        (out / "job").mkdir(parents=True)
        (out / "job" / "submission.json").write_text(json.dumps({"jobs": {"1": "pt-a", "2": "pt-b"}}))
        states = {"pt-a": {"state": "RUNNING"}, "pt-b": {"state": "SUCCEEDED"}}
        calls = []

        def fake_run(cmd, **kw):
            calls.append([cmd[3], cmd[-1]])
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with mock.patch.object(S, "describe", side_effect=lambda j: states[j]), \
                mock.patch.object(S.subprocess, "run", side_effect=fake_run):
            codes = S.delete(out)
        self.assertEqual(codes, {"1": 0, "2": 0})
        self.assertEqual(calls, [["stop", "pt-a"], ["delete", "pt-a"], ["delete", "pt-b"]])
        self.assertTrue((out / "job" / "submission_deleted.json").exists())


if __name__ == "__main__":
    unittest.main()
