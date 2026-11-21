"""q38 (2026-11-21, user OK): qa_eval easi turns the stack-dump thread on for every qwen3_5 benchmark, not only SITE.

The 27B base's one-card mmsi_bench, mindcube_tiny and 3dsrbench jobs printed nothing for 11.5 h after loading the
weights (blink, embspatial, viewspatial and vsi_bench finished in 15-20 min); the P0 line's one-card SITE video runs
stalled the same way without the thread (5 of 5) and finished with it (5 of 5).  Everything else runs exactly as at
271931e9 (git show; Q38_BEFORE_GIT names a checkout of the branch when this tree has no .git), with the
subprocess replaced by a fake.

    PYTHONPATH=.:verl python -B -m unittest vagen.q38.tests.test_qa_stackdump
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
BEFORE = "271931e9"  # the q38 branch's HEAD before this change
PATH = "data_gen/interiorgs_v10/qa_eval.py"


def _old_qa_eval():
    for repo in (REPO, os.environ.get("Q38_BEFORE_GIT")):
        if not repo:
            continue
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BEFORE}:{PATH}"], capture_output=True, text=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        mod = types.ModuleType("_qa_eval_before_stackdump")
        mod.__file__ = str(REPO / PATH)
        mod.__package__ = ""
        sys.modules[mod.__name__] = mod
        try:
            exec(compile(src, mod.__name__, "exec"), mod.__dict__)
        finally:
            sys.modules.pop(mod.__name__, None)
        return mod
    return None


class QaEvalStackDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import qa_eval as new

        cls.new = new
        cls.old = _old_qa_eval()

    def _run(self, mod, benchmark, model_type, root: Path):
        out = root / "o"
        shutil.rmtree(out, ignore_errors=True)
        seen = []

        def fake_run(cmd, cwd=None, env=None):
            seen.append((list(cmd), cwd, dict(env or {})))
            res = out / "m" / "easi_results.json"
            res.parent.mkdir(parents=True, exist_ok=True)
            res.write_text(json.dumps({"scores": {benchmark: 0.5}, "subScores": {benchmark: {"x": 1}}}))
            return types.SimpleNamespace(returncode=0)

        argv = ["qa_eval", "easi", "--name", "m", "--model", str(root), "--out", str(out), "--benchmark", benchmark,
                "--model-type", model_type]
        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            rc = mod.main()
        done = out / "m" / "done.json"
        return rc, seen, done.read_text() if done.is_file() else None

    def test_qwen3_5_every_benchmark_dumps(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for b in ("mmsi_bench", "mindcube_tiny", "3dsrbench", "blink", "vsi_bench"):
                new = self._run(self.new, b, "qwen3_5", root)
                first = new[1][0][2]["PYTHONPATH"].split(os.pathsep)[0]
                site = (Path(first) / "sitecustomize.py").read_text()  # before the old run clears the folder
                old = self._run(self.old, b, "qwen3_5", root)
                (cmd_n, cwd_n, env_n), (cmd_o, cwd_o, env_o) = new[1][0], old[1][0]
                self.assertEqual((new[0], cmd_n, cwd_n, new[2]), (old[0], cmd_o, cwd_o, old[2]), b)
                self.assertTrue(Path(first).name.startswith("_stack_dump_"), first)
                self.assertEqual(env_n.pop("PYTHONPATH"), first + os.pathsep + env_o.pop("PYTHONPATH", ""))
                self.assertEqual(env_n, env_o, b)  # nothing else in the environment changes
                self.assertIn("faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)", site)

    def test_site_and_other_model_types_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for b, kind in (("site_video", "qwen3_5"), ("site_image", "qwen3_5"), ("mmsi_bench", "qwen3_vl"),
                            ("site_video", "qwen3_vl"), ("blink", "qwen3_vl")):
                self.assertEqual(self._run(self.new, b, kind, root), self._run(self.old, b, kind, root), (b, kind))


if __name__ == "__main__":
    unittest.main()
