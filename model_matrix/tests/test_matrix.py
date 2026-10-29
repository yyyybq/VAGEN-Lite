"""The model line's downstream matrix (model_matrix/downstream/matrix.py, the user's 3.4 of 2026-10-27)."""
import tempfile
import unittest
from pathlib import Path


class MatrixTest(unittest.TestCase):
    def test_cells(self):
        from model_matrix.downstream import matrix as M

        self.assertEqual(M.point_cell({"all": [328, 944], "spatial": [151, 447]}),
                         "总体 34.7%（328/944）<br>空间 33.8%（151/447）")
        self.assertEqual(M.diff_cell({"all": {"diff": 0.0148, "p": 0.3244}, "spatial": {"diff": 0.0022, "p": 1.0}}),
                         "总体 +1.5（p 0.32）<br>空间 +0.2（p 1）")
        self.assertEqual(M.rate(0, 0), "—")

    def test_train_state(self):
        from model_matrix.downstream import matrix as M

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "DS-X-100-s1"
            exp.mkdir()
            self.assertEqual(M.train_state(exp), "排队")
            (exp / "train.log").write_text("x\r 45%|...| 90123/200000 [1:00:00<1:00:00, 1.3it/s]\r")
            self.assertEqual(M.train_state(exp), "训练中：第 90,123 步")
            (exp / "train.log").write_text("x\r100%|...| 200000/200000 [1:00:00<0:00:00, 1.3it/s]\r")
            self.assertEqual(M.train_state(exp), "训完，任务内评测中")
            done = exp / "evals" / "steps_200000" / "out" / "episodes"
            done.mkdir(parents=True)
            (done / "summary.json").write_text("{}")
            self.assertEqual(M.train_state(exp), "训完、评完")

    def test_stopped(self):
        import json
        from model_matrix.downstream import matrix as M

        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp) / "DS-X-100-s1"
            exp.mkdir()
            self.assertEqual(M.stopped(exp), {})
            (exp / "train.log").write_text("x\r 75%|...| 149902/200000 [1:00:00<1:00:00, 1.3it/s]\r")
            (exp / "stopped.json").write_text(json.dumps({"step": 149902, "note": "按要求停"}, ensure_ascii=False),
                                              encoding="utf-8")
            self.assertEqual(M.train_state(exp), "停训：第 149,902 步（按要求停）")

    def test_rows(self):
        from model_matrix.downstream import matrix as M

        codes = [r[1] for r in M.ROWS]
        self.assertEqual(codes[:4], ["B", "S", "RS", "RD"])  # the downstream line's four starts first
        self.assertEqual(len(set(codes)), len(codes))
        for line, code, label, ref in M.ROWS:  # every non-raw start has its own line's raw start as reference
            if ref:
                self.assertIn(ref, ("B", "Q3B", "CSB"))


if __name__ == "__main__":
    unittest.main()
